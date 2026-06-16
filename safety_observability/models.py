from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import constants as C


def resolve_device(requested: str) -> str:
    return "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"


def resolve_onnx_providers(requested: str = "auto") -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    requested = (requested or "auto").lower()
    if requested in ("cuda", "gpu"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif requested == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [provider for provider in providers if provider in available] or ["CPUExecutionProvider"]


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


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


class MiniLMToxicSpamONNXModel:
    """Adapter for navodPeiris/minilm-toxic-spam-classifier.

    The model is a 3-class softmax classifier: safe, toxic, spam. It has no
    sexual head, so the runtime maps max(toxic, spam) to harmful_content and
    leaves sexual at 0.0.
    """

    def __init__(
        self,
        model_path: str,
        *,
        max_length: int = 128,
        onnx_provider: str = "auto",
        harmful_labels: tuple[str, ...] = ("toxic", "spam"),
        **_: Any,
    ) -> None:
        import onnxruntime as ort

        self.model_path = model_path
        self.max_length = max_length
        self.harmful_labels = {label.lower() for label in harmful_labels}
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        cfg = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
        self.id2label = {int(k): v for k, v in cfg.get("id2label", {}).items()}
        self.providers = resolve_onnx_providers(onnx_provider)
        self.session = ort.InferenceSession(
            str(Path(model_path) / "onnx" / "model.onnx"),
            providers=self.providers,
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    def classify(self, text: str) -> dict[str, Any]:
        enc = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        feed = {key: value for key, value in enc.items() if key in self.input_names}
        started = time.time()
        logits = self.session.run(None, feed)[0]
        probs = _softmax(logits)[0]
        latency = round((time.time() - started) * 1000, 3)
        raw = {
            self.id2label.get(i, f"LABEL_{i}"): float(probs[i])
            for i in range(len(probs))
        }
        harmful = max(
            (prob for label, prob in raw.items() if label.lower() in self.harmful_labels),
            default=0.0,
        )
        return {
            "scores": {C.HARMFUL_CONTENT: harmful, C.SEXUAL: 0.0},
            "raw": {"scores": raw, "providers": self.session.get_providers()},
            "latency_ms": latency,
        }
