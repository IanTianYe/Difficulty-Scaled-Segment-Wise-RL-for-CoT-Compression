# -*- coding: utf-8 -*-
import os
import re
import json
import math
import hashlib
import threading
from typing import List, Dict, Optional, Tuple, Any, Union

# -----------------------------
# General & Math Utilities
# -----------------------------

def _hash_update(h: "hashlib._Hash", s: str):
    h.update((s or "").encode("utf-8", errors="ignore"))
    h.update(b"\0")

def clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def _safe_preview(s: str, n: int) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return s if len(s) <= n else (s[:n] + " ...")

# -----------------------------
# Configuration & IO
# -----------------------------
def load_config(path: str) -> Dict[str, Any]:
    path = (path or "").strip()
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext == ".json":
            obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
        if ext in [".yml", ".yaml"]:
            try:
                import yaml  # pip install pyyaml
            except Exception as e:
                raise RuntimeError("YAML config requires PyYAML: pip install pyyaml") from e
            obj = yaml.safe_load(f)
            return obj if isinstance(obj, dict) else {}
    raise ValueError(f"Unsupported config extension: {ext}")

# -----------------------------
# Parsing Logic (Think/Answer Extraction)
# -----------------------------
def strip_before_think(text: str) -> str:
    s = (text or "")
    m = re.search(r"<think\b[^>]*>", s, flags=re.I)
    if not m:
        return s
    return s[m.end():]

def extract_think_and_answer(full_generated: str) -> Tuple[str, str]:
    # Returns: (think_text, answer_text)
    if "</think>" in full_generated:
        pre, post = full_generated.split("</think>", 1)
    else:
        pre, post = "", full_generated

    if post.startswith("\n"):
        ans = post[1:]
    else:
        ans = post

    ans = ans.rstrip()
    if ans.endswith("<|im_end|>"):
        ans = ans[: -len("<|im_end|>")]

    return pre, ans

# -----------------------------
# Token Calculation
# -----------------------------
def text_token_len(tokenizer, text: str) -> int:
    s = text or ""
    if not s:
        return 0
    return len(tokenizer.encode(s, add_special_tokens=False))

def group_indices_by_prompt(prompts: List[str]) -> Dict[str, List[int]]:
    mp: Dict[str, List[int]] = {}
    for i, p in enumerate(prompts):
        mp.setdefault(p, []).append(i)
    return mp

# -----------------------------
# Reward Calculation Logic
# -----------------------------
def format_reward(text: str) -> int:
    if text.count("</think>") != 1:
        return 0

    if not text.endswith("<|im_end|>"):
        return 0

    think, ans = extract_think_and_answer(text)
    if not think.strip():
        return 0

    if not ans.strip():
        return 0

    return 1

def eff_reward_minmax_group(
    L: float,
    L_min: float,
    L_max: float,
    lcomp_margin: float,
    fmt_r: int,
    ans_r: int,
    eps: float = 1e-6,
) -> float:
    """
    R_eff = 1 - (L - L_min)/(L_max - L_min) mapped to [0,1]
    BUT if L <= lcomp_margin -> directly 1.
    Recommended gating: Only pay when fmt==1 and ans==1.
    """
    if int(fmt_r) != 1 or int(ans_r) != 1:
        return 0.0

    L = float(max(1.0, L))
    L_min = float(max(1.0, L_min))
    L_max = float(max(1.0, L_max))
    lcomp_margin = float(max(1.0, lcomp_margin))

    if L <= lcomp_margin:
        return 1.0

    denom = (L_max - L_min)
    if denom <= eps:
        return 1.0

    r = 1.0 - (L - L_min) / (denom + eps)
    return float(clip(r, 0.0, 1.0))

import math

def r_len_align_reward(L_ans: float, L_ref: float, floor: float = 16.0) -> float:
    L_ref = float(max(float(L_ref), float(floor)))
    L_ans = float(L_ans)
    floor = float(floor)

    if L_ans < L_ref:
        return float(math.exp(-abs(L_ans - L_ref) / L_ref))

    L_cap = L_ref + floor
    if (L_ref <= L_ans) and (L_ans <= L_cap):
        return 1.0

    return float(math.exp(-abs(L_ans - L_cap) / L_cap))
