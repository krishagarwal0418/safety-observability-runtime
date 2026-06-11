from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from . import constants as C
from .config import load_config, resolve_path
from .deterministic import ATTACK, MODERATION, evaluate as deterministic_gate
from .fasttext_router import FastTextRouter
from .models import ModerationModel, PromptInjectionModel, resolve_device
from .normalize import normalize


class SafetyObservabilityClassifier:
    def __init__(self, config_path: str | None = None) -> None:
        self.config = load_config(config_path)
        runtime = self.config.get("runtime", {})
        self.device = resolve_device(runtime.get("device", "cuda"))
        self.max_length = int(runtime.get("max_length", 128))
        self.fp16_on_cuda = bool(runtime.get("fp16_on_cuda", True))
        models = self.config["models"]
        self.fasttext = FastTextRouter(resolve_path(models["fasttext_router"]["local_path"]))
        common = {
            "device": self.device,
            "max_length": self.max_length,
            "fp16_on_cuda": self.fp16_on_cuda,
        }
        self.prompt_injection = PromptInjectionModel(str(resolve_path(models["prompt_injection"]["local_path"])), **common)
        self.moderation = ModerationModel(str(resolve_path(models["moderation"]["local_path"])), **common)

    def classify(self, text: str, *, full_scan: bool | None = None, include_raw: bool = False) -> dict[str, Any]:
        started = time.time()
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        thresholds = self.config["thresholds"]
        runtime = self.config.get("runtime", {})
        full_scan = runtime.get("full_scan_default", False) if full_scan is None else full_scan
        norm = normalize(text)
        rules = deterministic_gate(norm)
        ft = self.fasttext.predict(norm.detection_text)
        ft_scores = ft["scores"]

        run_attack = full_scan or ATTACK in rules.force_route or ft_scores["attack"] >= thresholds["attack_route"]
        run_moderation = full_scan or MODERATION in rules.force_route or ft_scores["moderation"] >= thresholds["moderation_route"]
        fasttext_direct_safe = (
            bool(runtime.get("fasttext_direct_safe_enabled", False))
            and not full_scan
            and rules.allow_fast_skip
            and ft_scores.get("safe", 0.0) >= thresholds["fasttext_direct_safe_score"]
            and ft_scores["attack"] < thresholds["fasttext_direct_safe_max_route"]
            and ft_scores["moderation"] < thresholds["fasttext_direct_safe_max_route"]
        )
        fast_allow = (
            fasttext_direct_safe
            or (
                not full_scan
                and rules.allow_fast_skip
                and ft_scores["attack"] < thresholds["fast_allow"]
                and ft_scores["moderation"] < thresholds["fast_allow"]
            )
        )

        scores = {label: 0.0 for label in C.PUBLIC_LABELS}
        raw: dict[str, Any] = {"fasttext": ft, "rules": {"routes": sorted(rules.force_route), "reasons": rules.reasons}}
        triggered: list[str] = ["fasttext_router"] if self.fasttext.loaded else []

        if run_attack and not fast_allow:
            pi = self.prompt_injection.classify(norm.model_text)
            scores.update(pi["scores"])
            triggered.append("prompt_injection")
            if include_raw:
                raw["prompt_injection"] = pi["raw"]
        if run_moderation and not fast_allow:
            mod = self.moderation.classify(norm.model_text)
            scores.update(mod["scores"])
            triggered.append("moderation")
            if include_raw:
                raw["moderation"] = mod["raw"]

        labels = []
        if scores[C.PROMPT_INJECTION] >= thresholds["prompt_injection_review"]:
            labels.append(C.PROMPT_INJECTION)
        if scores[C.HARMFUL_CONTENT] >= thresholds["harmful_content_review"]:
            labels.append(C.HARMFUL_CONTENT)
        if scores[C.SEXUAL] >= thresholds["sexual_review"]:
            labels.append(C.SEXUAL)

        return {
            "labels": labels,
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "fast_allow": fast_allow,
            "fasttext_direct_safe": fasttext_direct_safe,
            "triggered_models": triggered,
            "skipped_models": [
                name for name in ("prompt_injection", "moderation") if name not in triggered
            ],
            "routing": {
                "fasttext": {k: round(v, 4) for k, v in ft_scores.items()},
                "rule_reasons": rules.reasons,
                "run_attack": run_attack,
                "run_moderation": run_moderation,
                "fasttext_direct_safe_thresholds": {
                    "safe_score": thresholds["fasttext_direct_safe_score"],
                    "max_route": thresholds["fasttext_direct_safe_max_route"],
                },
            },
            "latency_ms": round((time.time() - started) * 1000, 2),
            "raw": raw if include_raw else None,
        }
