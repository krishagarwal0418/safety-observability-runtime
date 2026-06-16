from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from . import constants as C
from .config import load_config, resolve_path
from .deterministic import ATTACK, MODERATION, evaluate as deterministic_gate
from .fasttext_router import FastTextRouter
from .models import MiniLMToxicSpamONNXModel, ModerationModel, PromptInjectionModel, resolve_device
from .normalize import normalize


class SafetyObservabilityClassifier:
    def __init__(
        self,
        config_path: str | None = None,
        *,
        runtime_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.config = load_config(config_path)
        if runtime_overrides:
            self.config.setdefault("runtime", {}).update(
                {key: value for key, value in runtime_overrides.items() if value is not None}
            )
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
        moderation_spec = models["moderation"]
        moderation_cls = (
            MiniLMToxicSpamONNXModel
            if moderation_spec.get("backend") == "minilm_toxic_spam_onnx"
            else ModerationModel
        )
        moderation_kwargs = dict(common)
        if moderation_spec.get("backend") == "minilm_toxic_spam_onnx":
            moderation_kwargs["onnx_provider"] = runtime.get("onnx_provider", "auto")
        self.moderation = moderation_cls(str(resolve_path(moderation_spec["local_path"])), **moderation_kwargs)

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
        fasttext_direct_prompt = (
            bool(runtime.get("fasttext_direct_classification_enabled", False))
            and not full_scan
            and rules.allow_fast_skip
            and ft_scores["attack"] >= thresholds.get("fasttext_direct_prompt_injection_score", 1.1)
            and ft_scores["moderation"] <= thresholds.get("fasttext_direct_prompt_injection_max_moderation", 0.0)
        )
        fasttext_direct_harmful = (
            bool(runtime.get("fasttext_direct_classification_enabled", False))
            and not fasttext_direct_prompt
            and not full_scan
            and rules.allow_fast_skip
            and ft_scores["moderation"] >= thresholds.get("fasttext_direct_harmful_content_score", 1.1)
            and ft_scores["attack"] <= thresholds.get("fasttext_direct_harmful_content_max_attack", 0.0)
        )
        fasttext_direct_safe = (
            bool(runtime.get("fasttext_direct_safe_enabled", False))
            and not full_scan
            and not fasttext_direct_prompt
            and not fasttext_direct_harmful
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

        if fasttext_direct_prompt:
            scores[C.PROMPT_INJECTION] = 1.0
            triggered.append("fasttext_direct_prompt_injection")
        elif run_attack and not fast_allow:
            pi = self.prompt_injection.classify(norm.model_text)
            scores.update(pi["scores"])
            triggered.append("prompt_injection")
            if include_raw:
                raw["prompt_injection"] = pi["raw"]
        if fasttext_direct_harmful:
            scores[C.HARMFUL_CONTENT] = 1.0
            triggered.append("fasttext_direct_harmful_content")
        if run_moderation and not fast_allow:
            mod = self.moderation.classify(norm.model_text)
            scores.update(mod["scores"])
            triggered.append("moderation")
            if include_raw:
                raw["moderation"] = mod["raw"]
        if fasttext_direct_harmful:
            scores[C.HARMFUL_CONTENT] = max(scores[C.HARMFUL_CONTENT], 1.0)

        labels = []
        # Injection is vetoed when the moderation scorer marks the row strongly
        # harmful: the injection model confuses toxic content with attacks, but
        # real injections stay low on moderation. If moderation did not run,
        # scores[HARMFUL_CONTENT] is 0.0 and the veto is a no-op.
        max_harmful = thresholds.get("prompt_injection_max_harmful", 1.01)
        if (
            scores[C.PROMPT_INJECTION] >= thresholds["prompt_injection_review"]
            and scores[C.HARMFUL_CONTENT] < max_harmful
        ):
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
            "fasttext_direct_prompt_injection": fasttext_direct_prompt,
            "fasttext_direct_harmful_content": fasttext_direct_harmful,
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
                "fasttext_direct_classification_thresholds": {
                    "prompt_injection_score": thresholds.get("fasttext_direct_prompt_injection_score"),
                    "prompt_injection_max_moderation": thresholds.get("fasttext_direct_prompt_injection_max_moderation"),
                    "harmful_content_score": thresholds.get("fasttext_direct_harmful_content_score"),
                    "harmful_content_max_attack": thresholds.get("fasttext_direct_harmful_content_max_attack"),
                },
            },
            "latency_ms": round((time.time() - started) * 1000, 2),
            "raw": raw if include_raw else None,
        }
