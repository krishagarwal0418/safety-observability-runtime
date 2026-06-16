from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def resolve_device(requested: str) -> str:
    return "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"


def resolve_onnx_providers(requested: str = "auto") -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    requested = (requested or "auto").lower()
    if requested in ("cuda", "gpu"):
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif requested == "cpu":
        preferred = ["CPUExecutionProvider"]
    else:
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [provider for provider in preferred if provider in available] or ["CPUExecutionProvider"]


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class PromptInjectionModel:
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        max_length: int = 512,
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

    def classify(self, text: str) -> dict[str, Any]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        if self.device == "cuda":
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            torch.cuda.synchronize()
        started = time.time()
        with torch.inference_mode():
            logits = self.model(**encoded).logits.float().detach().cpu()[0]
        if self.device == "cuda":
            torch.cuda.synchronize()
        probs = torch.softmax(logits, dim=-1)
        raw = {self.id2label.get(i, f"LABEL_{i}"): float(probs[i]) for i in range(len(probs))}
        score = 0.0
        for label, prob in raw.items():
            low = label.lower()
            if "inject" in low or "attack" in low or "unsafe" in low or label == "LABEL_1":
                score = max(score, prob)
        return {
            "score": score,
            "raw": raw,
            "latency_ms": round((time.time() - started) * 1000, 3),
        }


class MiniLMToxicSpamONNXModel:
    def __init__(
        self,
        model_path: str,
        *,
        max_length: int = 512,
        onnx_provider: str = "auto",
        harmful_labels: tuple[str, ...] = ("toxic", "spam"),
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
        encoded = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        feed = {key: value for key, value in encoded.items() if key in self.input_names}
        started = time.time()
        logits = self.session.run(None, feed)[0]
        probs = _softmax(logits)[0]
        raw = {self.id2label.get(i, f"LABEL_{i}"): float(probs[i]) for i in range(len(probs))}
        score = max(
            (prob for label, prob in raw.items() if label.lower() in self.harmful_labels),
            default=0.0,
        )
        return {
            "score": score,
            "raw": {"scores": raw, "providers": self.session.get_providers()},
            "latency_ms": round((time.time() - started) * 1000, 3),
        }
