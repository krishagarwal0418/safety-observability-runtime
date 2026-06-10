from __future__ import annotations

import time
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import constants as C


def resolve_device(requested: str) -> str:
    return "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"


class HFClassifier:
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        max_length: int = 128,
        fp16_on_cuda: bool = True,
    ) -> None:
        self.model_path = model_path
        self.device = resolve_device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        if self.device == "cuda":
            self.model = self.model.to("cuda")
            if fp16_on_cuda:
                self.model = self.model.half()
        cfg = getattr(self.model, "config", None)
        self.id2label = {int(k): v for k, v in getattr(cfg, "id2label", {}).items()} if cfg else {}

    def _raw_probs(self, text: str, sigmoid: bool = False) -> tuple[dict[str, float], float]:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        if self.device == "cuda":
            enc = {k: v.to("cuda") for k, v in enc.items()}
            torch.cuda.synchronize()
        started = time.time()
        with torch.inference_mode():
            logits = self.model(**enc).logits.float().detach().cpu()[0]
        if self.device == "cuda":
            torch.cuda.synchronize()
        probs = torch.sigmoid(logits) if sigmoid else torch.softmax(logits, dim=-1)
        return {
            self.id2label.get(i, f"LABEL_{i}"): float(probs[i])
            for i in range(len(probs))
        }, round((time.time() - started) * 1000, 3)


class PromptInjectionModel(HFClassifier):
    def classify(self, text: str) -> dict[str, Any]:
        raw, latency = self._raw_probs(text)
        score = 0.0
        for label, prob in raw.items():
            low = label.lower()
            if any(tok in low for tok in ("injection", "malicious", "attack", "unsafe")):
                score = max(score, prob)
            elif label == "LABEL_1":
                score = max(score, prob)
        return {"scores": {C.PROMPT_INJECTION: score}, "raw": raw, "latency_ms": latency}


class JailbreakModel(HFClassifier):
    def classify(self, text: str) -> dict[str, Any]:
        raw, latency = self._raw_probs(text)
        score = 0.0
        for label, prob in raw.items():
            low = label.lower()
            if any(tok in low for tok in ("jailbreak", "attack", "unsafe", "malicious")):
                score = max(score, prob)
            elif label == "LABEL_1":
                score = max(score, prob)
        return {"scores": {C.JAILBREAK: score}, "raw": raw, "latency_ms": latency}


class ModerationModel(HFClassifier):
    def classify(self, text: str) -> dict[str, Any]:
        raw, latency = self._raw_probs(text, sigmoid=True)
        harmful = 0.0
        sexual = 0.0
        for label, prob in raw.items():
            low = label.lower()
            if "sexual" in low or low in ("s", "s3"):
                sexual = max(sexual, prob)
            if (
                "harmful" in low
                or "hate" in low
                or "harassment" in low
                or "toxic" in low
                or "violence" in low
                or "self" in low
                or low in ("h", "h2", "hr", "sh", "v", "v2")
            ):
                harmful = max(harmful, prob)
            if label == "LABEL_0":
                harmful = max(harmful, prob)
            if label == "LABEL_1":
                sexual = max(sexual, prob)
        return {
            "scores": {C.HARMFUL_CONTENT: harmful, C.SEXUAL: sexual},
            "raw": raw,
            "latency_ms": latency,
        }
