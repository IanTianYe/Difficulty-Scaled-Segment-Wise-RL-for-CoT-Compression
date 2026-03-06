import os
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
import unsloth
import atexit
import argparse
import hashlib
import textwrap
import threading
from typing import List, Dict, Optional, Tuple, Any
from unsloth import FastLanguageModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from segment_grpo_trainer import CoTSegmentedGRPOTrainer
from rl_agents import AnswerJudger, RefAnswerProvider

from utils import (
    _hash_update,
    _safe_preview,
    load_config,
    text_token_len,
    extract_think_and_answer,
    group_indices_by_prompt,
    format_reward,
    eff_reward_minmax_group,
    r_len_align_reward,
    strip_before_think,
)


# -----------------------------
# Batch-shared reward computer (avoid duplicate API calls across multi rewards)
# -----------------------------
class RewardBatchComputer:

    def __init__(
        self,
        tokenizer,
        executor: ThreadPoolExecutor,
        judger: Optional[AnswerJudger],
        ref_provider: Optional[RefAnswerProvider],
        args,
    ):
        self.tok = tokenizer
        self.exec = executor
        self.judger = judger
        self.ref_provider = ref_provider
        self.args = args
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._order = deque()
        self._max_cache_batches = 16
        self._call_lock = threading.Lock()
        self._call_n = 0
        self._watch_jsonl_lock = threading.Lock()

    def _next_call_id(self) -> int:
        with self._call_lock:
            self._call_n += 1
            return int(self._call_n)

    def _make_batch_key(self, prompts: List[str], completion_texts: List[str], ground_truth: List[str]) -> str:
        h = hashlib.sha256()
        for p, c, g in zip(prompts, completion_texts, ground_truth):
            _hash_update(h, p)
            _hash_update(h, c)
            _hash_update(h, g)
        return h.hexdigest()

    def _get_f(self, debug: dict, key: str, i: int) -> Optional[float]:
        v: Any = debug.get(key, None)
        if isinstance(v, (list, tuple)) and 0 <= i < len(v):
            try:
                return float(v[i])
            except Exception:
                return None
        return None

    def _push(self, extras, label: str, val: Optional[float], fmt: str):
        if val is not None:
            extras.append(f"{label}={format(val, fmt)}")

    def _pack_cache_minimal(
            self,
            r_fmt: List[float],
            r_ans: List[float],
            r_eff_daar: List[float],
            r_len_align: List[float],
            diff_weight: List[float],
    ) -> Dict[str, Any]:
        return {
            "R_fmt_comp": [float(x) for x in r_fmt],
            "R_ans_comp": [float(x) for x in r_ans],
            "R_eff_comp": [float(x) for x in r_eff_daar],
            "R_len_comp": [float(x) for x in r_len_align],
            "diff_weight": [float(x) for x in diff_weight],  # NEW
        }

    # -----------------------------
    # Stage 1: Parse completions
    # -----------------------------
    def _parse_completions(self, completion_texts: List[str]) -> Dict[str, Any]:
        """
        Input: completion_texts (already strip_before_think)
        Output:
          - think_texts, answer_texts
          - think_lens
          - fmt_ok (0/1)
        """
        n = len(completion_texts)
        think_texts: List[str] = [""] * n
        answer_texts: List[str] = [""] * n
        think_lens: List[float] = [0.0] * n
        fmt_ok: List[int] = [0] * n
        for i, t in enumerate(completion_texts):
            think_txt, ans_txt = extract_think_and_answer(t)
            think_texts[i] = think_txt
            answer_texts[i] = ans_txt
            think_lens[i] = float(text_token_len(self.tok, think_txt))
            fmt_ok[i] = int(format_reward(t))
        return {
            "think_texts": think_texts,
            "answer_texts": answer_texts,
            "think_lens": think_lens,
            "fmt_ok": fmt_ok,
        }

    # -----------------------------
    # Stage 2: External signals (judge / cot-validate / ref)
    # -----------------------------
    def _gather_external_signals(
        self,
        prompts: List[str],
        answer_texts: List[str],
        ground_truth: List[str],
        fmt_ok: List[int],
    ) -> Dict[str, Any]:
        """
        Output:
          - ans_ok: per-sample 0/1 from judger (only if fmt_ok==1 else 0)
          - ref_ans_len_by_prompt: dict prompt -> ref answer token length
        """
        n = len(prompts)
        ans_ok: List[int] = [0] * n
        ref_ans_len_by_prompt: Dict[str, float] = {}

        fut2key: Dict[Any, Tuple[str, Any]] = {}

        # --- judger (only if fmt ok)
        if self.judger is not None:
            for i in range(n):
                if int(fmt_ok[i]) != 1:
                    continue
                fut = self.exec.submit(self.judger.judge, answer_texts[i], ground_truth[i])
                fut2key[fut] = ("ans", i)

        # --- ref provider (per unique prompt)
        if self.ref_provider is not None:
            for p in set(prompts):
                fut = self.exec.submit(self.ref_provider.get_ref_completion, p)
                fut2key[fut] = ("ref", p)

        for fut in as_completed(list(fut2key.keys())):
            typ, key = fut2key[fut]
            try:
                res = fut.result()
            except Exception:
                res = None

            if typ == "ans":
                i = int(key)
                ans_ok[i] = int(res) if res is not None else 0
            elif typ == "ref":
                p = str(key)
                if isinstance(res, str) and res:
                    _, ref_ans = extract_think_and_answer(res)
                    ref_ans_len_by_prompt[p] = float(text_token_len(self.tok, ref_ans))

        return {
            "ans_ok": ans_ok,
            "ref_ans_len_by_prompt": ref_ans_len_by_prompt,
        }

    # -----------------------------
    # Stage 3: Group aggregation + compute rewards + build debug
    # -----------------------------
    def _compute_rewards_and_debug(
        self,
        prompts: List[str],
        completion_texts: List[str],
        think_texts: List[str],
        fmt_ok: List[int],
        ans_ok: List[int],
        think_lens: List[float],
        answer_texts: List[str],
        ref_ans_len_by_prompt: Dict[str, float],
    ) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:

        n = len(prompts)
        groups = group_indices_by_prompt(prompts)  # prompt -> [idxs]

        # per-sample group_acc
        group_acc_per_sample: List[float] = [0.0] * n
        for _, idxs in groups.items():
            # acc = sum(int(ans_ok[j]) for j in idxs) / float(len(idxs))
            acc = sum(int(ans_ok[j]) * int(fmt_ok[j]) for j in idxs) / float(len(idxs))
            for j in idxs:
                group_acc_per_sample[j] = float(acc)

        # margin-based L_ok:
        # L_min/L_max computed from correct samples if available; fallback to group think lens
        lcomp_margin = max(0, int(getattr(self.args, "lcomp_margin", 0)))

        think_len_min_by_prompt: Dict[str, float] = {}
        think_len_max_by_prompt: Dict[str, float] = {}

        for p, idxs in groups.items():
            correct_vals = [
                float(think_lens[j]) for j in idxs
                if int(fmt_ok[j]) == 1 and int(ans_ok[j]) == 1 and float(think_lens[j]) > 0
            ]
            if len(correct_vals) > 2:
                L_min = float(min(correct_vals))
                L_max = float(max(correct_vals))
            else:
                # Too few correct samples (0 or 1) => trigger the fallback logic in the trainer when denom <= eps
                L_min = 1.0
                L_max = 1.0

            think_len_min_by_prompt[p] = max(1.0, L_min)
            think_len_max_by_prompt[p] = max(1.0, L_max)

        # knobs
        eff_eps = float(getattr(self.args, "eff_eps", 1e-6))
        lenref_floor = float(getattr(self.args, "lenref_floor", 16.0))

        # per-sample outputs
        r_fmt: List[float] = [0.0] * n
        r_ans: List[float] = [0.0] * n
        r_eff_daar: List[float] = [0.0] * n
        r_len_align: List[float] = [0.0] * n

        # debug extras
        diff_weight: List[float] = [0.0] * n
        eff_raw: List[float] = [0.0] * n
        L_min_list: List[float] = [0.0] * n
        L_max_list: List[float] = [0.0] * n
        ans_lens: List[float] = [0.0] * n
        ref_ans_lens: List[float] = [0.0] * n

        for i in range(n):
            p = prompts[i]
            gacc = float(group_acc_per_sample[i])
            W_diff = float(2.0 - gacc)

            L_min = float(think_len_min_by_prompt.get(p, 1.0))
            L_max = float(think_len_max_by_prompt.get(p, L_min))
            L_think = max(1.0, float(think_lens[i]))

            fmt = int(fmt_ok[i])
            ans = int(ans_ok[i])

            r_fmt[i] = float(fmt)
            r_ans[i] = float(ans) if fmt == 1 else 0.0

            # efficiency reward: ONLY when fmt & ans
            if fmt == 1 and ans == 1:
                raw_eff = float(eff_reward_minmax_group(
                    L=L_think, L_min=L_min, L_max=L_max, lcomp_margin=lcomp_margin,
                    fmt_r=1, ans_r=1, eps=eff_eps
                ))
            else:
                raw_eff = 0.0

            eff_raw[i] = raw_eff
            r_eff_daar[i] = raw_eff

            # len align: ONLY when fmt & ans & have ref
            a_len = float(text_token_len(self.tok, answer_texts[i]))
            r_len = float(ref_ans_len_by_prompt.get(p, 0.0))
            ans_lens[i] = a_len
            ref_ans_lens[i] = r_len
            if fmt == 1 and ans == 1 and r_len > 0:
                r_len_align[i] = float(r_len_align_reward(a_len, r_len, floor=lenref_floor))
            else:
                r_len_align[i] = 0.0

            diff_weight[i] = W_diff
            L_min_list[i] = L_min
            L_max_list[i] = L_max

        rewards = {
            "r_fmt": r_fmt,
            "r_ans": r_ans,
            "r_eff_daar": r_eff_daar,
            "r_len_align": r_len_align,
            "diff_weight": diff_weight,
        }

        debug = {
            "prompts": prompts,
            "completion_texts": completion_texts,
            "think_texts": think_texts,
            "answer_texts": answer_texts,
            "fmt_ok": fmt_ok,
            "ans_ok": ans_ok,
            "think_lens": think_lens,
            "ans_lens": ans_lens,
            "ref_ans_lens": ref_ans_lens,
            "group_acc": group_acc_per_sample,
            "diff_weight": diff_weight,
            "eff_raw": eff_raw,
            "L_min": L_min_list,
            "L_max": L_max_list,
            "r_fmt": r_fmt,
            "r_ans": r_ans,
            "r_eff_daar": r_eff_daar,
            "r_len_align": r_len_align,
            "knobs": {
                "lcomp_margin": lcomp_margin,
                "eff_eps": eff_eps,
                "lenref_floor": lenref_floor,
            },
        }
        return rewards, debug

    # -----------------------------
    # Watch: print ONLY the max-think-length sample in this batch
    # -----------------------------
    def _watch(self, call_id: int, debug: Dict[str, Any]):
        watch_every = int(getattr(self.args, "watch_every", 10))
        if not (bool(getattr(self.args, "watch_enable", 0)) and watch_every > 0 and (call_id % watch_every == 0)):
            return

        prompts: List[str] = debug["prompts"]
        completion_texts: List[str] = debug["completion_texts"]
        think_texts: List[str] = debug["think_texts"]
        answer_texts: List[str] = debug["answer_texts"]
        think_lens: List[float] = debug["think_lens"]
        fmt_ok: List[int] = debug["fmt_ok"]
        ans_ok: List[int] = debug["ans_ok"]
        r_fmt: List[float] = debug["r_fmt"]
        r_ans: List[float] = debug["r_ans"]
        r_eff: List[float] = debug["r_eff_daar"]
        r_len: List[float] = debug["r_len_align"]

        n = len(think_lens)
        if n <= 0:
            return

        i_max = max(range(n), key=lambda i: float(think_lens[i]))
        w_fmt, w_ans, w_eff, w_len, diff_weight = [float(x) for x in self.args.reward_weights_list]
        total = (
            w_fmt * float(r_fmt[i_max]) +
            w_ans * float(r_ans[i_max]) +
            w_eff * float(r_eff[i_max]) +
            w_len * float(r_len[i_max])
        )

        pv = int(getattr(self.args, "watch_preview_chars", 300))
        show_full_answer = int(getattr(self.args, "watch_print_full_answer", 0))
        show_full_think = int(getattr(self.args, "watch_print_full_think", 0))
        show_full_completion = int(getattr(self.args, "watch_print_full_completion", 0))

        prompt_prev = _safe_preview(prompts[i_max], pv).replace("\n", "\\n")
        think_prev = _safe_preview(think_texts[i_max] or "", pv if show_full_think else min(140, pv)).replace("\n", "\\n")
        ans_prev = (answer_texts[i_max] or "").replace("\n", "\\n") if show_full_answer else _safe_preview(answer_texts[i_max] or "", pv).replace("\n", "\\n")
        comp_prev = _safe_preview(completion_texts[i_max] or "", pv).replace("\n", "\\n") if show_full_completion else None

        print("\n" + "=" * 110)
        print(f"[WATCH] call={call_id}  n={n}  pick=argmax(think_len) -> i={i_max}")
        print(f"[WATCH] weights: w_fmt={w_fmt:.3f} w_ans={w_ans:.3f} w_eff={w_eff:.3f} w_len={w_len:.3f}")
        print(
            f"[WATCH] fmt={int(fmt_ok[i_max])} ans={int(ans_ok[i_max])} "
            f"think_len={float(think_lens[i_max]):.0f}  total={float(total):.4f}"
        )

        extras: list[str] = []
        self._push(extras, "gacc", self._get_f(debug, "group_acc", i_max), ".3f")
        self._push(extras, "W", self._get_f(debug, "diff_weight", i_max), ".3f")
        self._push(extras, "eff_raw", self._get_f(debug, "eff_raw", i_max), ".4f")
        self._push(extras, "Lmin", self._get_f(debug, "L_min", i_max), ".0f")
        self._push(extras, "Lmax", self._get_f(debug, "L_max", i_max), ".0f")
        self._push(extras, "ans_len", self._get_f(debug, "ans_lens", i_max), ".0f")
        self._push(extras, "ref_len", self._get_f(debug, "ref_ans_lens", i_max), ".0f")
        if extras:
            print("[WATCH] " + "  ".join(extras))

        print(
            f"[WATCH] comps: r_fmt={float(r_fmt[i_max]):.4f} "
            f"r_ans={float(r_ans[i_max]):.4f} "
            f"r_eff_daar={float(r_eff[i_max]):.4f} "
            f"r_len_align={float(r_len[i_max]):.4f}"
        )
        print("-" * 110 + "\n")
        print(f"prompt: {prompt_prev}\n")
        print(f"think:  {think_prev}\n")
        print(f"answer: {ans_prev}\n")
        if comp_prev is not None:
            print(f"completion: {comp_prev}")
        print("=" * 110 + "\n")

    # -----------------------------
    # Main entry
    # -----------------------------
    def get_metrics(self, prompts, completions, ground_truth, completion_ids=None) -> Dict[str, Any]:
        # decode -> completion_texts
        if completion_ids is not None:
            completion_texts = self.tok.batch_decode(completion_ids, skip_special_tokens=False)
        else:
            completion_texts = completions

        # normalize: strip garbage before <think>
        completion_texts = [strip_before_think(t) for t in completion_texts]

        # cache check
        batch_key = self._make_batch_key(prompts, completion_texts, ground_truth)
        with self._lock:
            cached = self._cache.get(batch_key)
        if cached is not None:
            return cached

        call_id = self._next_call_id()

        # -------- Stage 1: parse completion --------
        parsed = self._parse_completions(completion_texts)
        think_texts: List[str] = parsed["think_texts"]
        answer_texts: List[str] = parsed["answer_texts"]
        think_lens: List[float] = parsed["think_lens"]
        fmt_ok: List[int] = parsed["fmt_ok"]

        # -------- Stage 2: external signals --------
        ext = self._gather_external_signals(
            prompts=prompts,
            answer_texts=answer_texts,
            ground_truth=ground_truth,
            fmt_ok=fmt_ok,
        )
        ans_ok: List[int] = ext["ans_ok"]
        ref_ans_len_by_prompt: Dict[str, float] = ext["ref_ans_len_by_prompt"]

        # -------- Stage 3: compute rewards + debug --------
        rewards, debug = self._compute_rewards_and_debug(
            prompts=prompts,
            completion_texts=completion_texts,
            think_texts=think_texts,
            fmt_ok=fmt_ok,
            ans_ok=ans_ok,
            think_lens=think_lens,
            answer_texts=answer_texts,
            ref_ans_len_by_prompt=ref_ans_len_by_prompt,
        )

        # -------- Watch (max-think only) --------
        self._watch(call_id, debug)

        # -------- Minimal cache output --------
        metrics_min = self._pack_cache_minimal(
            r_fmt=rewards["r_fmt"],
            r_ans=rewards["r_ans"],
            r_eff_daar=rewards["r_eff_daar"],
            r_len_align=rewards["r_len_align"],
            diff_weight=rewards["diff_weight"],  # NEW: 直接用你已算好的
        )
        with self._lock:
            self._cache[batch_key] = metrics_min
            self._order.append(batch_key)
            while len(self._order) > self._max_cache_batches:
                k = self._order.popleft()
                self._cache.pop(k, None)

        return metrics_min


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # [1] Config / I/O
    ap.add_argument("--config", type=str, default="", help="Path to config file (.json/.yaml/.yml)")
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B-Thinking-2507")
    ap.add_argument("--output_dir", type=str, default="")

    # [2] Data / rollout batching
    ap.add_argument("--train_data", type=str, default="", help="Path to training parquet file")
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--prompts_per_step", type=int, default=1)

    # [3] Sequence lengths / generation
    ap.add_argument("--max_prompt_len", type=int, default=256)
    ap.add_argument("--max_completion_length", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    # [3.5] Temperature schedule (warmup -> cosine decay)
    ap.add_argument("--temp_schedule", type=str, default="none",
                    help="none | linear | cosine")
    ap.add_argument("--temp_start", type=float, default=None,
                    help="Start temperature (defaults to --temperature)")
    ap.add_argument("--temp_final", type=float, default=None,
                    help="Final temperature (defaults to temp_start)")

    # [4] Optimization / GRPO
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--beta_kl", type=float, default=0.0)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--token_level_grpo", type=int, default=0,
                    help="Use token-level GRPO (1) or vanilla TRL GRPO (0).")

    # [5] Reward knobs
    ap.add_argument("--reward_weights", type=str, default="1.0,1.0,1.0,1.0,0.0",
                    help="Comma-separated weights for [fmt, ans, eff, len_align, diff_weight].")
    ap.add_argument("--eff_eps", type=float, default=1e-6)
    ap.add_argument("--lcomp_margin", type=int, default=0,
                    help="L_ok = L_min + lcomp_margin (margin-only target).")
    ap.add_argument("--lenref_floor", type=int, default=16, help="Minimum L_ref used in R_len.")

    # [6] Unsloth / LoRA / vLLM memory
    ap.add_argument("--use_lora", type=int, default=1, help="1 = LoRA/PEFT finetuning, 0 = full-parameter finetuning")
    ap.add_argument("--use_4bit", type=int, default=1)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str,
                    default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.7,
                    help="Unsloth/vLLM GPU memory utilization.")

    # [7] Threading / external calls
    ap.add_argument("--api_workers", type=int, default=16,
                    help="max threads for API calls per step")

    # [8] Judger API
    ap.add_argument("--enable_judger", type=int, default=1)
    ap.add_argument("--judger_api_key", type=str, default=os.environ.get("DEEPSEEK_API_KEY", ""))
    ap.add_argument("--judger_base_url", type=str, default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    ap.add_argument("--judger_model", type=str, default=os.environ.get("DEEPSEEK_JUDGE_MODEL", os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")))
    ap.add_argument("--judger_timeout_s", type=int, default=60)

    # [9] Reference generation
    ap.add_argument("--ref_max_new_tokens", type=int, default=32768,
                    help="Max new tokens for ref generation.")

    # [10] Watch / debug
    ap.add_argument("--watch_enable", type=int, default=0)
    ap.add_argument("--watch_every", type=int, default=10)
    ap.add_argument("--watch_preview_chars", type=int, default=200)
    ap.add_argument("--watch_write_jsonl", type=str, default="")
    ap.add_argument("--watch_print_full_answer", type=int, default=0)
    ap.add_argument("--watch_print_full_think", type=int, default=0)
    ap.add_argument("--watch_print_full_completion", type=int, default=0)

    # load config overrides
    cfg_args, _ = ap.parse_known_args()
    if cfg_args.config:
        cfg = load_config(cfg_args.config)
        ap.set_defaults(**cfg)

    args = ap.parse_args()

    if not args.train_data:
        raise ValueError("train_data is required (set via --train_data or in --config file).")
    if not args.output_dir:
        raise ValueError("output_dir is required (set via --output_dir or in --config file).")

    try:
        w = [float(x.strip()) for x in str(args.reward_weights).split(",") if x.strip() != ""]
    except Exception:
        w = [1.0, 1.0, 1.0, 1.0, 0.0]
    if len(w) != 5:
        raise ValueError("reward_weights must have exactly 5 floats for [fmt, ans_daar, eff_daar, len_align, diff_weight].")
    args.reward_weights_list = w

    max_seq_length = int(args.max_prompt_len) + int(args.max_completion_length)

    fast_inference = True
    try:
        import vllm
    except Exception:
        fast_inference = False
        print("[WARN] vllm not found -> fast_inference=False (pip install vllm for faster rollout)")

    use_lora = int(args.use_lora) == 1

    if not use_lora and int(args.use_4bit) == 1:
        raise ValueError(
            "Full-parameter training does not support load_in_4bit=1. "
            "Set --use_4bit 0 when --use_lora 0."
        )

    if use_lora:
        model, tok = FastLanguageModel.from_pretrained(
            model_name=args.model_id,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=bool(int(args.use_4bit)),
            fast_inference=fast_inference,
            max_lora_rank=int(args.lora_r),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            full_finetuning=False,
        )
    else:
        model, tok = FastLanguageModel.from_pretrained(
            model_name=args.model_id,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=False,
            fast_inference=False,
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            full_finetuning=True,
        )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if use_lora:
        print("[INFO] Using LoRA / PEFT finetuning")
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            bias="none",
            target_modules=[x.strip() for x in str(args.target_modules).split(",") if x.strip()],
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
    else:
        print("[INFO] Using full-parameter finetuning")
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    SYSTEM_PROMPT = textwrap.dedent("""\
        Follow this format:
        <think>Do NOT restate the problem. Provide brief reasoning; focus only on the logic, not on formatting or presentation.</think>
        Then give a clear, complete, very detailed and user-friendly answer.
        Keep the think section as short as possible while staying correct.
    """)

    def build_prompt(q: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    ds = load_dataset("parquet", data_files=args.train_data, split="train")
    ds = ds.filter(lambda ex: ex["problem"] and ex["answer"] and ex["problem"].strip() != "" and ex["answer"].strip() != "")
    ds = ds.shuffle(seed=42)
    ds = ds.map(lambda ex: {"prompt": build_prompt(ex["problem"]), "ground_truth": ex["answer"]})

    EXECUTOR = ThreadPoolExecutor(max_workers=max(1, int(args.api_workers)))

    @atexit.register
    def _shutdown_executor():
        try:
            EXECUTOR.shutdown(wait=True, cancel_futures=False)
        except Exception:
            pass

    judger = None
    if int(getattr(args, "enable_judger", 0)) == 1 and args.judger_api_key:
        judger = AnswerJudger(
            api_key=args.judger_api_key,
            base_url=args.judger_base_url,
            model=args.judger_model,
            timeout_s=args.judger_timeout_s,
        )
    else:
        print("[WARN] Judge disabled (no api key).")

    ref_provider = RefAnswerProvider(model, tok, max_new_tokens=int(args.ref_max_new_tokens))

    batch_computer = RewardBatchComputer(tok, EXECUTOR, judger, ref_provider, args)

    def reward_fmt(prompts, completions, ground_truth, completion_ids=None, **kwargs):
        m = batch_computer.get_metrics(prompts, completions, ground_truth, completion_ids=completion_ids)
        return [float(x) for x in m["R_fmt_comp"]]

    def reward_ans(prompts, completions, ground_truth, completion_ids=None, **kwargs):
        m = batch_computer.get_metrics(prompts, completions, ground_truth, completion_ids=completion_ids)
        return [float(x) for x in m["R_ans_comp"]]

    def reward_eff_daar(prompts, completions, ground_truth, completion_ids=None, **kwargs):
        m = batch_computer.get_metrics(prompts, completions, ground_truth, completion_ids=completion_ids)
        return [float(x) for x in m["R_eff_comp"]]

    def reward_len_align_fn(prompts, completions, ground_truth, completion_ids=None, **kwargs):
        m = batch_computer.get_metrics(prompts, completions, ground_truth, completion_ids=completion_ids)
        return [float(x) for x in m["R_len_comp"]]

    def reward_diff_weight(prompts, completions, ground_truth, completion_ids=None, **kwargs):
        m = batch_computer.get_metrics(prompts, completions, ground_truth, completion_ids=completion_ids)
        return [float(x) for x in m["diff_weight"]]

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=int(args.prompts_per_step),
        gradient_accumulation_steps=int(args.grad_accum),
        num_generations=int(args.num_generations),
        max_prompt_length=int(args.max_prompt_len),
        max_completion_length=int(args.max_completion_length),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        beta=float(args.beta_kl),
        reward_weights=list(args.reward_weights_list),  # [w_fmt, w_ans, w_eff, w_len]
        scale_rewards="group",
        remove_unused_columns=False,
        logging_steps=1,
        save_steps=100,
        save_total_limit=3,
        report_to="none",
    )

    if int(args.token_level_grpo) == 1:
        print("[INFO] Using Token-level GRPO trainer")

        trainer = CoTSegmentedGRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=ds,
            reward_funcs=[reward_fmt, reward_ans, reward_eff_daar, reward_len_align_fn, reward_diff_weight],
            reward_zones=[
                "none",    # fmt
                "none",    # ans
                "think",  # eff_daar (think length)
                "answer", # len_align (answer length)
                "none" # diff_weight
            ],
            processing_class=tok,
            difficulty_weight="reward_diff_weight",
            diff_scale=1.0,
            temp_schedule=str(args.temp_schedule),
            temp_start=float(args.temp_start),
            temp_final=float(args.temp_final),
        )
    else:
        print("[INFO] Using vanilla TRL GRPO trainer")
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=ds,
            reward_funcs=[reward_fmt, reward_ans, reward_eff_daar, reward_len_align_fn],
            processing_class=tok,
        )

    try:
        trainer.model.config.use_cache = False
    except Exception:
        pass

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("Saved to:", args.output_dir)


if __name__ == "__main__":
    main()

