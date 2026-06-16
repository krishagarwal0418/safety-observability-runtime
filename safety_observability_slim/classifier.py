from __future__ import annotations

import time
from typing import Any

from .config import load_config, resolve_path
from .models import MiniLMToxicSpamONNXModel, PromptInjectionModel, resolve_device


class SafetyClassifier:
    def __init__(
        self,
        config_path: str | None = None,
        *,
        device: str | None = None,
        onnx_provider: str | None = None,
        max_length: int | None = None,
    ) -> None:
        self.config = load_config(config_path)
        runtime = self.config.setdefault("runtime", {})
        if device is not None:
            runtime["device"] = device
        if onnx_provider is not None:
            runtime["onnx_provider"] = onnx_provider
        if max_length is not None:
            runtime["max_length"] = max_length

        self.device = resolve_device(runtime.get("device", "cuda"))
        self.max_length = int(runtime.get("max_length", 512))
        self.onnx_provider = runtime.get("onnx_provider", "auto")
        self.fp16_on_cuda = bool(runtime.get("fp16_on_cuda", True))

        models = self.config["models"]
        self.prompt_injection = PromptInjectionModel(
            str(resolve_path(models["prompt_injection"]["local_path"])),
            device=self.device,
            max_length=self.max_length,
            fp16_on_cuda=self.fp16_on_cuda,
        )
        self.moderation = MiniLMToxicSpamONNXModel(
            str(resolve_path(models["moderation"]["local_path"])),
            max_length=self.max_length,
            onnx_provider=self.onnx_provider,
        )

    def classify(self, text: str, *, include_raw: bool = False) -> dict[str, Any]:
        started = time.time()
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        prompt = self.prompt_injection.classify(text)
        moderation = self.moderation.classify(text)
        thresholds = self.config["thresholds"]

        scores = {
            "prompt_injection": prompt["score"],
            "harmful_content": moderation["score"],
        }
        labels = []
        if scores["prompt_injection"] >= thresholds["prompt_injection_review"]:
            labels.append("prompt_injection")
        if scores["harmful_content"] >= thresholds["harmful_content_review"]:
            labels.append("harmful_content")

        result: dict[str, Any] = {
            "labels": labels,
            "scores": {key: round(value, 4) for key, value in scores.items()},
            "triggered_models": ["prompt_injection", "moderation"],
            "runtime": {
                "device": self.device,
                "onnx_provider": self.onnx_provider,
                "onnx_providers_active": moderation["raw"]["providers"],
                "max_length": self.max_length,
            },
            "latency_ms": round((time.time() - started) * 1000, 2),
            "model_latency_ms": {
                "prompt_injection": prompt["latency_ms"],
                "moderation": moderation["latency_ms"],
            },
        }
        if include_raw:
            result["raw"] = {
                "prompt_injection": prompt["raw"],
                "moderation": moderation["raw"],
            }
        return result
