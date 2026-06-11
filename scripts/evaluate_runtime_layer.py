#!/usr/bin/env python3
"""Evaluate the full runtime layer on a small labeled JSONL/CSV sample."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

from safety_observability import SafetyObservabilityClassifier
from safety_observability.constants import HARMFUL_CONTENT, PROMPT_INJECTION, PUBLIC_LABELS, SEXUAL


HARMFUL_ALIASES = {
    "harmful_content",
    "toxicity",
    "toxic",
    "hate",
    "harassment",
    "violence",
    "self_harm",
    "self-harm",
    "dangerous_information",
    "illegal_activity",
}
PROMPT_ALIASES = {
    "prompt_injection",
    "injection",
    "inject",
    "jailbreak",
    "attack",
    "malicious",
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("rows", [])
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    raise ValueError(f"Unsupported data format: {path}")


def pick_text(row: dict[str, Any]) -> str:
    for key in ("text", "prompt", "instruction", "content", "input"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def labels_for(row: dict[str, Any]) -> set[str]:
    raw = row.get("labels", row.get("label", []))
    if isinstance(raw, str):
        raw = [raw]
    labels = {str(x).lower().strip() for x in raw or []}
    out: set[str] = set()
    if labels & PROMPT_ALIASES:
        out.add(PROMPT_INJECTION)
    if labels & HARMFUL_ALIASES:
        out.add(HARMFUL_CONTENT)
    if "sexual" in labels:
        out.add(SEXUAL)

    for label in PUBLIC_LABELS:
        val = row.get(label)
        if str(val).lower() in ("1", "true", "yes", "positive"):
            out.add(label)
    return out


def binary_metrics(gold: list[int], pred: list[int]) -> dict[str, float]:
    tp = sum(1 for g, p in zip(gold, pred) if g and p)
    fp = sum(1 for g, p in zip(gold, pred) if not g and p)
    fn = sum(1 for g, p in zip(gold, pred) if g and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def pct(num: int, den: int) -> float:
    return round(100 * num / den, 2) if den else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return round(float(statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]), 2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/runtime.yaml")
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=3000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="reports/runtime_layer_eval.json")
    p.add_argument("--enable-direct-safe", action="store_true")
    p.add_argument("--full-scan", action="store_true")
    p.add_argument("--fast-allow", type=float, default=None)
    p.add_argument("--attack-route", type=float, default=None)
    p.add_argument("--moderation-route", type=float, default=None)
    p.add_argument("--direct-safe-score", type=float, default=None)
    p.add_argument("--direct-safe-max-route", type=float, default=None)
    args = p.parse_args()

    rows = [r for r in read_rows(Path(args.data)) if pick_text(r)]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit]

    clf = SafetyObservabilityClassifier(args.config)
    runtime = clf.config.setdefault("runtime", {})
    thresholds = clf.config.setdefault("thresholds", {})
    if args.enable_direct_safe:
        runtime["fasttext_direct_safe_enabled"] = True
    for attr, key in (
        ("fast_allow", "fast_allow"),
        ("attack_route", "attack_route"),
        ("moderation_route", "moderation_route"),
        ("direct_safe_score", "fasttext_direct_safe_score"),
        ("direct_safe_max_route", "fasttext_direct_safe_max_route"),
    ):
        val = getattr(args, attr)
        if val is not None:
            thresholds[key] = val

    gold = {label: [] for label in PUBLIC_LABELS}
    pred = {label: [] for label in PUBLIC_LABELS}
    latencies: list[float] = []
    bert_calls = 0
    routed_any = []
    unsafe_any = []
    direct_safe = 0
    fast_allow = 0
    unsafe_skipped_bert = 0
    unsafe_false_pass = 0
    model_counts = {"prompt_injection": 0, "moderation": 0}

    for idx, row in enumerate(rows, start=1):
        true_labels = labels_for(row)
        result = clf.classify(pick_text(row), full_scan=args.full_scan)
        got_labels = set(result["labels"])
        called_berts = [m for m in ("prompt_injection", "moderation") if m in result["triggered_models"]]
        for m in called_berts:
            model_counts[m] += 1
        bert_calls += len(called_berts)
        direct_safe += int(bool(result.get("fasttext_direct_safe")))
        fast_allow += int(bool(result.get("fast_allow")))
        if true_labels and not called_berts:
            unsafe_skipped_bert += 1
        if true_labels and not got_labels and not called_berts:
            unsafe_false_pass += 1
        routed_any.append(1 if called_berts else 0)
        unsafe_any.append(1 if true_labels else 0)
        latencies.append(float(result["latency_ms"]))
        for label in PUBLIC_LABELS:
            gold[label].append(1 if label in true_labels else 0)
            pred[label].append(1 if label in got_labels else 0)
        if idx % 100 == 0:
            print(f"[eval] {idx}/{len(rows)} rows")

    per_label = {label: binary_metrics(gold[label], pred[label]) for label in PUBLIC_LABELS}
    macro_f1 = sum(m["f1"] for m in per_label.values()) / len(per_label)
    report = {
        "rows": len(rows),
        "thresholds": {
            "attack_route": thresholds.get("attack_route"),
            "moderation_route": thresholds.get("moderation_route"),
            "fast_allow": thresholds.get("fast_allow"),
            "fasttext_direct_safe_enabled": runtime.get("fasttext_direct_safe_enabled"),
            "fasttext_direct_safe_score": thresholds.get("fasttext_direct_safe_score"),
            "fasttext_direct_safe_max_route": thresholds.get("fasttext_direct_safe_max_route"),
        },
        "label_metrics": per_label,
        "macro_f1": round(macro_f1, 4),
        "routing": {
            "bert_calls_total": bert_calls,
            "bert_calls_per_row": round(bert_calls / len(rows), 4) if rows else 0,
            "prompt_injection_calls": model_counts["prompt_injection"],
            "moderation_calls": model_counts["moderation"],
            "fast_allow_rows": fast_allow,
            "fast_allow_pct": pct(fast_allow, len(rows)),
            "fasttext_direct_safe_rows": direct_safe,
            "fasttext_direct_safe_pct": pct(direct_safe, len(rows)),
            "unsafe_skipped_bert_rows": unsafe_skipped_bert,
            "unsafe_skipped_bert_pct_of_unsafe": pct(unsafe_skipped_bert, sum(unsafe_any)),
            "unsafe_false_pass_rows": unsafe_false_pass,
            "unsafe_false_pass_pct_of_unsafe": pct(unsafe_false_pass, sum(unsafe_any)),
            "route_any_bert": binary_metrics(unsafe_any, routed_any),
        },
        "latency_ms": {
            "avg": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
