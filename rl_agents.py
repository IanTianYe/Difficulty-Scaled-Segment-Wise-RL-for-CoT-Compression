import re
import time
import threading
from typing import Optional, Union
import torch
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from utils import text_token_len

# -----------------------------
# API judge for correctness - thread-local client
# -----------------------------
class AnswerJudger:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout_s: int = 60,
        max_tokens: int = 4,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = int(max_tokens) if max_tokens is not None else 8
        self._local = threading.local()

    def _get_client(self) -> Optional[OpenAI]:
        if not self.api_key:
            return None
        c = getattr(self._local, "client", None)
        if c is None:
            c = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self._local.client = c
        return c

    @staticmethod
    def _parse_bool(s: str) -> Optional[int]:
        if not s:
            return None
        t = s.strip().lower()
        m = re.search(r"\b(true|false)\b", t)
        if m:
            return 1 if m.group(1) == "true" else 0
        m = re.search(r"\b([01])\b", t)
        if m:
            return int(m.group(1))
        if t.startswith("true"):
            return 1
        if t.startswith("false"):
            return 0
        return None

    def judge(self, pred_answer: str, gold_answer: str) -> int:
        client = self._get_client()
        pred = pred_answer.strip()
        gold = gold_answer.strip()
        system = (
            "You are a strict answer equivalence judge.\n"
            "Compare the FINAL answers only. Ignore reasoning.\n"
            "Output must be exactly one token: true or false (lowercase).\n"
            "No punctuation, no extra words, no newlines."
        )
        user = (
            "GOLD:\n"
            f"{gold}\n\n"
            "PRED:\n"
            f"{pred}\n\n"
            "Output true if final answers match, else false:"
        )
        last = ""
        for _ in range(3):
            try:
                system_msg: ChatCompletionSystemMessageParam = {"role": "system", "content": system}
                user_msg: ChatCompletionUserMessageParam = {"role": "user", "content": user}
                messages: list[ChatCompletionMessageParam] = [system_msg, user_msg]
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout_s,
                )
                text = (resp.choices[0].message.content or "").strip()
                v = self._parse_bool(text)
                if v is None:
                    v = 0
                return int(v)
            except Exception as e:
                last = str(e)
                time.sleep(1.0)
        print("[WARN] AnswerJudge failed:", last)
        return 0

# -----------------------------
# Ref provider (VRAM-safe): reuse same model but disable LoRA adapters during ref generation
# -----------------------------
class RefAnswerProvider:
    def __init__(self, model, tokenizer, max_new_tokens: int = 32768):
        self.model = model
        self.tok = tokenizer
        self.max_new_tokens = int(max_new_tokens)

    @torch.inference_mode()
    def get_ref_completion(self, prompt: str) -> str:
        inputs = self.tok(prompt, return_tensors="pt")
        inputs = {kk: vv.to(self.model.device) for kk, vv in inputs.items()}
        with self.model.disable_adapter():
            out = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=self.max_new_tokens,
            )
        input_len = inputs["input_ids"].shape[1]
        gen_ids = out[0][input_len:]
        completion = self.tok.decode(gen_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return completion