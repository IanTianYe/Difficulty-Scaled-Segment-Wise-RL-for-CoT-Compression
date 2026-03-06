from typing import List, Any, Dict, Optional
import torch
import math
from trl import GRPOTrainer


class CoTSegmentedGRPOTrainer(GRPOTrainer):

    def __init__(
        self,
        *args,
        reward_zones: List[str] = None,
        think_boundary: str = "</think>\n",
        end_boundary: str = "<|im_end|>",
        # name of reward function that returns diff_weight column
        difficulty_weight: str = "reward_diff_weight",
        # diff_scale multiplies diff_weight
        diff_scale: float = 1.0,
        # whether to apply diff to think advantages
        apply_diff_to_think: bool = True,
        temp_schedule: str = "none",
        temp_start: float = None,
        temp_final: float = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if hasattr(self.processing_class, "tokenizer"):  # Processor
            self._tok = self.processing_class.tokenizer
        else:  # Tokenizer
            self._tok = self.processing_class

        if reward_zones is None:
            raise ValueError("reward_zones must be provided")
        if len(reward_zones) != len(self.reward_funcs):
            raise ValueError("reward_zones length must equal reward_funcs length")
        self.reward_zones = reward_zones
        assert set(self.reward_zones) <= {"all", "think", "answer", "none"}

        self.difficulty_reward_name = str(difficulty_weight)
        self.diff_scale = float(diff_scale)
        self.apply_diff_to_think = bool(apply_diff_to_think)
        # tokenize boundaries once
        self._think_boundary_ids = self._tok.encode(think_boundary, add_special_tokens=False)
        self._end_boundary_ids = self._tok.encode(end_boundary, add_special_tokens=False)
        # cache for gathered rewards (GLOBAL)
        self._cached_rewards_per_func = None
        # cache reward func names in order
        self._reward_func_names: List[str] = []
        for fn in self.reward_funcs:
            n = getattr(fn, "__name__", None)
            self._reward_func_names.append(n or "reward_func")

        self.temp_schedule = str(temp_schedule).lower()
        self.temp_start = float(temp_start)
        self.temp_final = float(temp_final)

    # -----------------------------
    # Temperature schedule: linear warmup -> cosine decay
    # -----------------------------
    def _temperature_at_step(self, step: int) -> float:
        sched = str(getattr(self, "temp_schedule", "none")).lower()

        if sched in ("none", "", "off", "false", "0"):
            return float(getattr(self.args, "temperature", 0.7))

        max_steps = int(getattr(self.args, "max_steps", 0) or 0)
        if max_steps <= 0:
            return float(getattr(self.args, "temperature", 0.7))

        base = float(getattr(self.args, "temperature", 0.7))
        t0 = float(self.temp_start) if getattr(self, "temp_start", None) is not None else base
        tf = float(self.temp_final) if getattr(self, "temp_final", None) is not None else t0

        # clamp step, progress in [0,1]
        step = max(0, min(int(step), max_steps))
        progress = float(step) / float(max_steps)

        # Linear Annealing: t0 -> tf
        if sched in ("linear", "lin"):
            return float(t0 + (tf - t0) * progress)

        # Cosine Annealing: t0 -> tf
        if sched in ("cosine", "cos"):
            # cos_factor: 1 -> 0
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(tf + (t0 - tf) * cos_factor)

        raise ValueError(f"Unknown temp_schedule: {sched}. Use one of: none | linear | cosine")

    def _apply_temperature(self, temperature: float) -> None:
        """Sync temperature to all places TRL/transformers/vLLM use."""
        temperature = float(max(1e-6, temperature))

        # 1) trainer field used by TRL (vLLM sampling + logit scaling)
        self.temperature = temperature

        # 2) args (optional, but keep consistent)
        try:
            self.args.temperature = temperature
        except Exception:
            pass

        # 3) transformers generation_config path
        gen_cfg = getattr(self, "generation_config", None)
        if gen_cfg is not None:
            try:
                gen_cfg.temperature = temperature
            except Exception:
                pass

        # 4) generation_kwargs dict (TRL uses it to override model.generation_config)
        gkwargs = getattr(self, "generation_kwargs", None)
        if isinstance(gkwargs, dict):
            gkwargs["temperature"] = temperature

    def _update_temperature_for_generation(self):
        # Use optimizer-step-based schedule (global_step)
        step = int(getattr(self.state, "global_step", 0) or 0)
        t = self._temperature_at_step(step)
        self._apply_temperature(t)
        return t


    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    @staticmethod
    def _find_last_subseq(haystack: List[int], needle: List[int]) -> int:
        if not needle or not haystack or len(needle) > len(haystack):
            return -1
        for i in range(len(haystack) - len(needle), -1, -1):
            if haystack[i : i + len(needle)] == needle:
                return i
        return -1

    def _idx_by_reward_name(self, name: str) -> int:
        for i, n in enumerate(self._reward_func_names):
            if n == name:
                return i
        raise RuntimeError(
            f"Cannot find reward func named '{name}'. Got names={self._reward_func_names}"
        )

    def _build_masks_batch(self, completion_ids: torch.Tensor):
        """
        returns:
          think_mask:  (B,T)  <= includes </think>\\n boundary
          answer_mask: (B,T)  after </think>\\n up to <|im_end|> (inclusive)
          valid_mask:  (B,T)  <= <|im_end|> (inclusive)
        """
        B, T = completion_ids.shape
        device = completion_ids.device

        think_mask = torch.zeros((B, T), device=device, dtype=torch.float32)
        answer_mask = torch.zeros((B, T), device=device, dtype=torch.float32)
        valid_mask = torch.zeros((B, T), device=device, dtype=torch.float32)

        pad_id = self._tok.pad_token_id

        for b in range(B):
            row = completion_ids[b].tolist()

            # strip right padding
            if pad_id is not None:
                while row and row[-1] == pad_id:
                    row.pop()
            if not row:
                continue

            # ---------- find end boundary ----------
            end_start = self._find_last_subseq(row, self._end_boundary_ids)
            if end_start != -1:
                end_pos = end_start + len(self._end_boundary_ids)
                valid_mask[b, :end_pos] = 1.0
            else:
                # no im_end -> invalid => keep all zeros
                continue

            # ---------- find think boundary ----------
            search_region = row[:end_pos]
            think_start = self._find_last_subseq(search_region, self._think_boundary_ids)

            if think_start == -1:
                # no </think> -> all answer
                answer_mask[b, :end_pos] = 1.0
                continue

            think_end = think_start + len(self._think_boundary_ids)

            # NOTE: includes the </think>\n boundary tokens
            think_mask[b, :think_end] = 1.0

            if think_end < end_pos:
                answer_mask[b, think_end:end_pos] = 1.0

        return think_mask, answer_mask, valid_mask

    def _group_advantages_from_rewards(self, rewards: torch.Tensor, num_generations: int):
        """
        rewards: (N_global,)
        returns:
          adv: (N_global,)
          is_std_zero: (N_global,) bool
        """
        if rewards.dim() != 1:
            raise ValueError(f"rewards must be 1D, got shape={tuple(rewards.shape)}")
        if num_generations <= 0:
            raise ValueError("num_generations must be > 0")
        if rewards.numel() % num_generations != 0:
            raise ValueError(
                f"rewards length {rewards.numel()} not divisible by num_generations={num_generations}"
            )

        grouped = rewards.view(-1, num_generations)  # (G, ng)

        mean_grouped = grouped.mean(dim=1, keepdim=True)                # (G, 1)
        mean_grouped = mean_grouped.repeat(1, num_generations).view(-1) # (N,)

        scale_mode = getattr(self, "scale_rewards", "group")
        if scale_mode in ["group", "none"]:
            if num_generations > 1:
                std_grouped = grouped.std(dim=1, unbiased=False, keepdim=True)  # (G,1)
                std_grouped = std_grouped.repeat(1, num_generations).view(-1)  # (N,)
            else:
                std_grouped = torch.zeros_like(rewards)
        elif scale_mode == "batch":
            if rewards.numel() > 1:
                std_grouped = rewards.std(unbiased=False).expand_as(rewards)
            else:
                std_grouped = torch.zeros_like(rewards)
        else:
            raise ValueError(
                f"Invalid scale_rewards: {scale_mode}. Expected one of ['group','batch','none']."
            )

        adv = rewards - mean_grouped
        if scale_mode != "none":
            adv = adv / (std_grouped + 1e-4)

        is_std_zero = torch.isclose(std_grouped, torch.zeros_like(std_grouped))
        return adv, is_std_zero

    # ------------------------------------------------------------
    # core override points
    # ------------------------------------------------------------
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards_per_func = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        # cache GLOBAL gathered rewards
        self._cached_rewards_per_func = rewards_per_func.detach()
        return rewards_per_func

    def _generate_and_score_completions(self, inputs: List[Dict[str, Any]]):
        """
        Override TRL pipeline:
          - keep TRL reward computation & logging
          - replace scalar advantages with token-level ones
          - multiply THINK advantages by diff_weight column
        """
        cur_t = self._update_temperature_for_generation()
        # (可选) 打印观察
        if self.accelerator.is_main_process:
            print(f"[TEMP] step={int(getattr(self.state,'global_step',0))} temperature={cur_t:.4f}")
        out = super()._generate_and_score_completions(inputs)

        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        completion_ids = out["completion_ids"]  # (B_local, T)

        # (N_global, K)
        rewards_per_func = self._cached_rewards_per_func
        if rewards_per_func is None:
            raise RuntimeError("cached rewards missing: _calculate_rewards did not run or cache was overwritten")
        if rewards_per_func.dim() != 2:
            raise RuntimeError(f"cached rewards must be (N_global,K), got {tuple(rewards_per_func.shape)}")
        if rewards_per_func.size(0) % num_generations != 0:
            raise RuntimeError(
                f"N_global={rewards_per_func.size(0)} not divisible by num_generations={num_generations}"
            )

        # --- pick which columns contribute to which zone sums
        w = self.reward_weights.to(device).unsqueeze(0)  # (1,K)
        zones = self.reward_zones

        idx_all = [i for i, z in enumerate(zones) if z == "all"]
        idx_thk = [i for i, z in enumerate(zones) if z == "think"]
        idx_ans = [i for i, z in enumerate(zones) if z == "answer"]

        def weighted_sum(cols: List[int]) -> torch.Tensor:
            if not cols:
                return torch.zeros(rewards_per_func.size(0), device=device, dtype=torch.float32)
            return (rewards_per_func[:, cols].to(device) * w[:, cols]).nansum(dim=1).to(torch.float32)

        r_all = weighted_sum(idx_all)
        r_thk = weighted_sum(idx_thk)
        r_ans = weighted_sum(idx_ans)

        # totals for advantage computation
        r_think_total = r_all + r_thk
        r_answer_total = r_all + r_ans

        adv_think_g, _ = self._group_advantages_from_rewards(r_think_total, num_generations)
        adv_answer_g, _ = self._group_advantages_from_rewards(r_answer_total, num_generations)

        # --- multiply think advantages by diff_weight column (GLOBAL)
        if self.apply_diff_to_think:
            diff_weight_col = self._idx_by_reward_name(self.difficulty_reward_name)
            diff_weight_g = rewards_per_func[:, diff_weight_col].to(device=device, dtype=torch.float32)
            diff_weight_g = diff_weight_g * self.diff_scale

            adv_think_g = adv_think_g.to(torch.float32)
            adv_think_g = torch.where(
                adv_think_g > 0,
                adv_think_g * diff_weight_g,
                adv_think_g
            )
        else:
            adv_think_g = adv_think_g.to(torch.float32)

        adv_answer_g = adv_answer_g.to(torch.float32)

        # --- slice global -> local
        B_local = int(completion_ids.size(0))
        bs = torch.tensor([B_local], device=device, dtype=torch.long)
        all_bs = self.accelerator.gather(bs).tolist()
        rank = self.accelerator.process_index
        start = int(sum(all_bs[:rank]))
        end = start + int(all_bs[rank])

        if rewards_per_func.size(0) != sum(all_bs):
            raise RuntimeError(
                f"gathered rewards rows={rewards_per_func.size(0)} "
                f"!= sum(local_completion_bs)={sum(all_bs)}; batch alignment broken."
            )

        adv_think = adv_think_g[start:end]    # (B_local,)
        adv_answer = adv_answer_g[start:end]  # (B_local,)

        # --- token masks
        think_mask, answer_mask, valid_mask = self._build_masks_batch(completion_ids)

        # --- token-level advantages
        advantages_tok = (
            adv_think.unsqueeze(1) * think_mask
            + adv_answer.unsqueeze(1) * answer_mask
        ) * valid_mask
        advantages_tok = advantages_tok.to(torch.float32)

        # clear cache
        self._cached_rewards_per_func = None
        out["advantages"] = advantages_tok
        return out
