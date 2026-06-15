#!/usr/bin/env python3
"""Calibrate thresholds for the quantized CPU runtime stack."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evaluate_quantized_full_suite import (  # type: ignore[import-not-found]
    ATTACK,
    HARMFUL_CONTENT,
    MODERATION,
    PROMPT_INJECTION,
    PUBLIC_LABELS,
    SEXUAL,
    FastTextRouter,
    OnnxTextClassifier,
    balanced_sample,
    binary_metrics,
    bucket_for,
    deterministic_gate,
    load_config,
    moderation_scores,
    normalize,
    prompt_score,
    resolve_path,
)


WEAK_DEFAULTS = {"self_harm", "self-harm", "dangerous_information", "illegal_activity"}


def fbeta(precision: float, recall: float, beta: float) -> float:
    if precision <= 0 and recall <= 0:
        return 0.0
    beta2 = beta * beta
    return (1 + beta2) * precision * recall / ((beta2 * precision) + recall)


def pct(num: int, den: int) -> float:
    return round(100 * num / den, 3) if den else 0.0


def make_gold(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    gold = {label: [] for label in PUBLIC_LABELS}
    for row in rows:
        labels = set(row["labels"])
        for label in PUBLIC_LABELS:
            gold[label].append(1 if label in labels else 0)
    return gold


def choose_label_thresholds(
    gold: dict[str, list[int]],
    scores: dict[str, list[float]],
    *,
    min_recall: float,
    beta: float,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    grid = [round(x / 100, 2) for x in range(5, 96, 5)]
    for label in PUBLIC_LABELS:
        best = None
        for thr in grid:
            pred = [1 if s >= thr else 0 for s in scores[label]]
            m = binary_metrics(gold[label], pred)
            score = fbeta(m["precision"], m["recall"], beta)
            candidate = {
                "threshold": thr,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "fbeta": round(score, 4),
            }
            if m["recall"] >= min_recall:
                if best is None or (candidate["precision"], candidate["fbeta"]) > (best["precision"], best["fbeta"]):
                    best = candidate
            elif best is None:
                best = candidate
        out[label] = best or {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "fbeta": 0.0}
    return out


def route_metrics(rows: list[dict[str, Any]], routed: list[int]) -> dict[str, float]:
    unsafe = [1 if row["labels"] else 0 for row in rows]
    return binary_metrics(unsafe, routed)


def choose_routing_thresholds(
    rows: list[dict[str, Any]],
    ft_scores: list[dict[str, float]],
    rule_routes: list[set[str]],
    *,
    min_route_recall: float,
) -> dict[str, Any]:
    grid = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    best = None
    for attack_thr in grid:
        for mod_thr in grid:
            routed = []
            prompt_calls = 0
            mod_calls = 0
            for scores, forced in zip(ft_scores, rule_routes):
                run_attack = ATTACK in forced or scores["attack"] >= attack_thr
                run_mod = MODERATION in forced or scores["moderation"] >= mod_thr
                prompt_calls += int(run_attack)
                mod_calls += int(run_mod)
                routed.append(1 if run_attack or run_mod else 0)
            m = route_metrics(rows, routed)
            calls = prompt_calls + mod_calls
            candidate = {
                "attack_route": attack_thr,
                "moderation_route": mod_thr,
                "route_precision": m["precision"],
                "route_recall": m["recall"],
                "route_f1": m["f1"],
                "bert_calls": calls,
                "bert_calls_per_row": round(calls / len(rows), 4),
                "any_bert_rows": sum(routed),
            }
            if m["recall"] >= min_route_recall:
                if best is None or (candidate["bert_calls"], -candidate["route_precision"]) < (
                    best["bert_calls"],
                    -best["route_precision"],
                ):
                    best = candidate
            elif best is None or candidate["route_recall"] > best["route_recall"]:
                best = candidate
    return best or {}


def choose_safe_thresholds(
    rows: list[dict[str, Any]],
    ft_scores: list[dict[str, float]],
    rule_routes: list[set[str]],
    *,
    max_false_pass_pct: float,
) -> dict[str, Any]:
    safe_grid = [0.999, 0.995, 0.99, 0.98, 0.97, 0.95]
    route_grid = [0.001, 0.002, 0.005, 0.01, 0.02]
    unsafe_total = sum(1 for row in rows if row["labels"])
    candidates = []
    for safe_thr in safe_grid:
        for max_route in route_grid:
            passed = 0
            false_pass = 0
            for row, scores, forced in zip(rows, ft_scores, rule_routes):
                direct_safe = (
                    not forced
                    and scores.get("safe", 0.0) >= safe_thr
                    and scores["attack"] < max_route
                    and scores["moderation"] < max_route
                )
                if direct_safe:
                    passed += 1
                    false_pass += int(bool(row["labels"]))
            false_pass_pct = pct(false_pass, unsafe_total)
            candidates.append(
                {
                    "fasttext_direct_safe_score": safe_thr,
                    "fasttext_direct_safe_max_route": max_route,
                    "passed_rows": passed,
                    "passed_pct": pct(passed, len(rows)),
                    "unsafe_false_pass_rows": false_pass,
                    "unsafe_false_pass_pct_of_unsafe": false_pass_pct,
                }
            )
    valid = [c for c in candidates if c["unsafe_false_pass_pct_of_unsafe"] <= max_false_pass_pct]
    if valid:
        return max(valid, key=lambda c: (c["passed_rows"], -c["unsafe_false_pass_rows"]))
    return min(candidates, key=lambda c: (c["unsafe_false_pass_pct_of_unsafe"], -c["passed_rows"]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/runtime.yaml")
    p.add_argument("--data", action="append", required=True)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--prompt-onnx-dir", default=None)
    p.add_argument("--moderation-onnx-dir", default="models/onnx_int8/moderation")
    p.add_argument("--exclude-label", action="append", default=sorted(WEAK_DEFAULTS))
    p.add_argument("--min-label-recall", type=float, default=0.85)
    p.add_argument("--label-beta", type=float, default=1.5)
    p.add_argument("--min-route-recall", type=float, default=0.97)
    p.add_argument("--max-safe-false-pass-pct", type=float, default=1.0)
    p.add_argument("--output", default="reports/threshold_calibration_1k.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    max_length = int(args.max_length or cfg.get("runtime", {}).get("max_length", 128))
    prompt_dir = Path(args.prompt_onnx_dir) if args.prompt_onnx_dir else resolve_path(
        cfg["models"]["prompt_injection_onnx_int8"]["local_path"]
    )
    moderation_dir = Path(args.moderation_onnx_dir)
    excluded = {x.lower() for x in args.exclude_label}
    rows = balanced_sample([Path(x) for x in args.data], args.limit, args.seed, excluded)

    fasttext = FastTextRouter(resolve_path(cfg["models"]["fasttext_router"]["local_path"]))
    prompt_model = OnnxTextClassifier(prompt_dir, max_length=max_length, sigmoid_outputs=False)
    moderation_model = OnnxTextClassifier(moderation_dir, max_length=max_length, sigmoid_outputs=True)

    print("[calibrate] sampled labels:", {k: sum(1 for r in rows if bucket_for(set(r["labels"])) == k) for k in ("safe", PROMPT_INJECTION, HARMFUL_CONTENT, SEXUAL)})
    started = time.perf_counter()
    model_texts = []
    ft_scores = []
    rule_routes = []
    for idx, row in enumerate(rows, start=1):
        norm = normalize(row["text"])
        rules = deterministic_gate(norm)
        ft = fasttext.predict(norm.detection_text)
        model_texts.append(norm.model_text)
        ft_scores.append(ft["scores"])
        rule_routes.append(set(rules.force_route))
        if idx % 100 == 0:
            print(f"[calibrate] routed {idx}/{len(rows)}")

    prompt_raw, prompt_stats = prompt_model.predict_batches(model_texts, args.batch_size)
    mod_raw, mod_stats = moderation_model.predict_batches(model_texts, args.batch_size)
    scores = {label: [] for label in PUBLIC_LABELS}
    for pi_raw, m_raw in zip(prompt_raw, mod_raw):
        scores[PROMPT_INJECTION].append(prompt_score(pi_raw))
        ms = moderation_scores(m_raw)
        scores[HARMFUL_CONTENT].append(ms[HARMFUL_CONTENT])
        scores[SEXUAL].append(ms[SEXUAL])
    gold = make_gold(rows)

    label_thresholds = choose_label_thresholds(
        gold,
        scores,
        min_recall=args.min_label_recall,
        beta=args.label_beta,
    )
    routing_thresholds = choose_routing_thresholds(
        rows,
        ft_scores,
        rule_routes,
        min_route_recall=args.min_route_recall,
    )
    safe_thresholds = choose_safe_thresholds(
        rows,
        ft_scores,
        rule_routes,
        max_false_pass_pct=args.max_safe_false_pass_pct,
    )
    recommended_yaml = {
        "thresholds": {
            "attack_route": routing_thresholds.get("attack_route", 0.01),
            "moderation_route": routing_thresholds.get("moderation_route", 0.01),
            "fast_allow": 0.0,
            "fasttext_direct_safe_score": safe_thresholds["fasttext_direct_safe_score"],
            "fasttext_direct_safe_max_route": safe_thresholds["fasttext_direct_safe_max_route"],
            "prompt_injection_review": label_thresholds[PROMPT_INJECTION]["threshold"],
            "harmful_content_review": label_thresholds[HARMFUL_CONTENT]["threshold"],
            "sexual_review": label_thresholds[SEXUAL]["threshold"],
        },
        "runtime": {"fasttext_direct_safe_enabled": True},
    }
    report = {
        "rows": len(rows),
        "excluded_labels": sorted(excluded),
        "sample_distribution": {k: sum(1 for r in rows if bucket_for(set(r["labels"])) == k) for k in ("safe", PROMPT_INJECTION, HARMFUL_CONTENT, SEXUAL)},
        "targets": {
            "min_label_recall": args.min_label_recall,
            "label_beta": args.label_beta,
            "min_route_recall": args.min_route_recall,
            "max_safe_false_pass_pct": args.max_safe_false_pass_pct,
        },
        "recommended": recommended_yaml,
        "label_thresholds": label_thresholds,
        "routing_thresholds": routing_thresholds,
        "safe_thresholds": safe_thresholds,
        "latency_ms": {
            "wall_total": round((time.perf_counter() - started) * 1000, 3),
            "prompt_onnx": {k: v for k, v in prompt_stats.items() if k != "token_lengths"},
            "moderation_onnx": {k: v for k, v in mod_stats.items() if k != "token_lengths"},
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    yaml_out = out.with_suffix(".recommended.yaml")
    yaml_out.write_text(
        "\n".join(
            [
                "thresholds:",
                f"  attack_route: {recommended_yaml['thresholds']['attack_route']}",
                f"  moderation_route: {recommended_yaml['thresholds']['moderation_route']}",
                "  fast_allow: 0.0",
                f"  fasttext_direct_safe_score: {recommended_yaml['thresholds']['fasttext_direct_safe_score']}",
                f"  fasttext_direct_safe_max_route: {recommended_yaml['thresholds']['fasttext_direct_safe_max_route']}",
                f"  prompt_injection_review: {recommended_yaml['thresholds']['prompt_injection_review']}",
                f"  harmful_content_review: {recommended_yaml['thresholds']['harmful_content_review']}",
                f"  sexual_review: {recommended_yaml['thresholds']['sexual_review']}",
                "runtime:",
                "  fasttext_direct_safe_enabled: true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"[calibrate] wrote {out}")
    print(f"[calibrate] wrote {yaml_out}")


if __name__ == "__main__":
    main()
