#!/usr/bin/env python3
"""Calibrate all thresholds for the PyTorch GPU runtime stack on a large corpus.

Runs every row through FastText + both BERT models in one pass, then grid-searches
over all thresholds offline. Calibrates:

  Routing   : attack_route, moderation_route
  Fast-allow: fast_allow (skip both BERTs entirely for clearly-safe FastText scores)
  Direct-FT : fasttext_direct_prompt_injection_score/max_moderation
              fasttext_direct_harmful_content_score/max_attack
              fasttext_direct_safe_score/max_route
  BERT label: prompt_injection_review, harmful_content_review, sexual_review

Writes reports/calibration_pytorch_50k.{json,recommended.yaml}.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from evaluate_pytorch_gpu_suite import (  # type: ignore[import-not-found]
    TorchTextClassifier,
    load_tokenizer,
)
from evaluate_quantized_full_suite import (  # type: ignore[import-not-found]
    ATTACK,
    HARMFUL_CONTENT,
    MODERATION,
    PROMPT_INJECTION,
    PUBLIC_LABELS,
    SEXUAL,
    FastTextRouter,
    balanced_sample,
    binary_metrics,
    bucket_for,
    deterministic_gate,
    load_config,
    moderation_scores,
    normalize,
    pct,
    prompt_score,
    resolve_path,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

WEAK_DEFAULTS = {"self_harm", "self-harm", "dangerous_information", "illegal_activity", "unknown", "violence"}


def fbeta(precision: float, recall: float, beta: float) -> float:
    if precision <= 0 and recall <= 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / ((b2 * precision) + recall)


def make_gold(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    gold = {label: [] for label in PUBLIC_LABELS}
    for row in rows:
        labels = set(row["labels"])
        for label in PUBLIC_LABELS:
            gold[label].append(1 if label in labels else 0)
    return gold


# ──────────────────────────────────────────────────────────────────────────────
# Threshold choosers
# ──────────────────────────────────────────────────────────────────────────────

def choose_label_thresholds(
    gold: dict[str, list[int]],
    scores: dict[str, list[float]],
    *,
    min_recall: float,
    beta: float,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    grid = [round(x / 100, 2) for x in range(5, 96, 5)]
    for label in PUBLIC_LABELS:
        best = None
        for thr in grid:
            pred = [1 if s >= thr else 0 for s in scores[label]]
            m = binary_metrics(gold[label], pred)
            fb = fbeta(m["precision"], m["recall"], beta)
            candidate = {
                "threshold": thr,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "fbeta": round(fb, 4),
            }
            if m["recall"] >= min_recall:
                if best is None or (candidate["precision"], candidate["fbeta"]) > (best["precision"], best["fbeta"]):
                    best = candidate
            elif best is None:
                best = candidate
        out[label] = best or {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "fbeta": 0.0}
    return out


def choose_routing_thresholds(
    rows: list[dict[str, Any]],
    ft_scores: list[dict[str, float]],
    rule_routes: list[set[str]],
    *,
    min_route_recall: float,
) -> dict[str, Any]:
    grid = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
    unsafe = [1 if row["labels"] else 0 for row in rows]
    best = None
    for attack_thr in grid:
        for mod_thr in grid:
            routed, prompt_calls, mod_calls = [], 0, 0
            for scores, forced in zip(ft_scores, rule_routes):
                run_attack = ATTACK in forced or scores["attack"] >= attack_thr
                run_mod = MODERATION in forced or scores["moderation"] >= mod_thr
                prompt_calls += int(run_attack)
                mod_calls += int(run_mod)
                routed.append(1 if run_attack or run_mod else 0)
            m = binary_metrics(unsafe, routed)
            calls = prompt_calls + mod_calls
            c = {
                "attack_route": attack_thr,
                "moderation_route": mod_thr,
                "route_precision": m["precision"],
                "route_recall": m["recall"],
                "route_f1": m["f1"],
                "bert_calls": calls,
                "bert_calls_per_row": round(calls / max(1, len(rows)), 4),
                "any_bert_rows": sum(routed),
                "any_bert_pct": pct(sum(routed), len(rows)),
            }
            if m["recall"] >= min_route_recall:
                if best is None or (c["bert_calls"], -c["route_precision"]) < (best["bert_calls"], -best["route_precision"]):
                    best = c
            elif best is None or c["route_recall"] > best.get("route_recall", 0):
                best = c
    return best or {}


def choose_fast_allow_threshold(
    rows: list[dict[str, Any]],
    ft_scores: list[dict[str, float]],
    rule_routes: list[set[str]],
    *,
    max_false_pass_pct: float,
) -> dict[str, Any]:
    """Find the highest fast_allow score that lets us skip both BERTs safely."""
    grid = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2]
    unsafe_total = sum(1 for row in rows if row["labels"])
    candidates = []
    for thr in grid:
        passed = false_pass = 0
        for row, scores, forced in zip(rows, ft_scores, rule_routes):
            if forced:
                continue
            fast_allow = scores["attack"] < thr and scores["moderation"] < thr
            if fast_allow:
                passed += 1
                false_pass += int(bool(row["labels"]))
        fp_pct = pct(false_pass, unsafe_total)
        candidates.append({
            "fast_allow": thr,
            "passed_rows": passed,
            "passed_pct": pct(passed, len(rows)),
            "false_pass_rows": false_pass,
            "false_pass_pct_of_unsafe": fp_pct,
        })
    valid = [c for c in candidates if c["false_pass_pct_of_unsafe"] <= max_false_pass_pct]
    if valid:
        return max(valid, key=lambda c: (c["passed_rows"], -c["false_pass_rows"]))
    return min(candidates, key=lambda c: (c["false_pass_pct_of_unsafe"], -c["passed_rows"]))


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
            passed = false_pass = 0
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
            fp_pct = pct(false_pass, unsafe_total)
            candidates.append({
                "fasttext_direct_safe_score": safe_thr,
                "fasttext_direct_safe_max_route": max_route,
                "passed_rows": passed,
                "passed_pct": pct(passed, len(rows)),
                "unsafe_false_pass_rows": false_pass,
                "unsafe_false_pass_pct_of_unsafe": fp_pct,
            })
    valid = [c for c in candidates if c["unsafe_false_pass_pct_of_unsafe"] <= max_false_pass_pct]
    if valid:
        return max(valid, key=lambda c: (c["passed_rows"], -c["unsafe_false_pass_rows"]))
    return min(candidates, key=lambda c: (c["unsafe_false_pass_pct_of_unsafe"], -c["passed_rows"]))


def choose_direct_label_threshold(
    rows: list[dict[str, Any]],
    ft_scores: list[dict[str, float]],
    rule_routes: list[set[str]],
    *,
    score_key: str,
    target_label: str,
    max_other_key: str,
    min_precision: float,
) -> dict[str, Any]:
    score_grid = [0.999, 0.995, 0.99, 0.98, 0.97, 0.95, 0.9, 0.85, 0.8]
    other_grid = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    total_target = sum(1 for row in rows if target_label in row["labels"])
    candidates = []
    for score_thr in score_grid:
        for max_other in other_grid:
            pred, gold = [], []
            for row, scores, forced in zip(rows, ft_scores, rule_routes):
                hit = not forced and scores[score_key] >= score_thr and scores[max_other_key] <= max_other
                if hit:
                    pred.append(1)
                    gold.append(1 if target_label in row["labels"] else 0)
            m = binary_metrics(gold, pred) if pred else {k: 0 for k in ("precision", "recall", "f1", "tp", "fp", "fn", "tn")}
            candidates.append({
                "enabled": bool(pred),
                "score": score_thr,
                "max_other": max_other,
                "rows": len(pred),
                "rows_pct": pct(len(pred), len(rows)),
                "precision": m["precision"],
                "coverage_of_label": round(m["tp"] / total_target, 4) if total_target else 0.0,
                "tp": m["tp"],
                "fp": m["fp"],
            })
    valid = [c for c in candidates if c["precision"] >= min_precision and c["rows"] > 0]
    if valid:
        return max(valid, key=lambda c: (c["rows"], c["precision"]))
    best = max(candidates, key=lambda c: (c["precision"], c["rows"]))
    best["enabled"] = False
    return best


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate PyTorch GPU thresholds on a large corpus.")
    p.add_argument("--config", default="configs/runtime.yaml")
    p.add_argument("--data", action="append", required=True, help="JSONL/CSV/JSON source. Pass multiple times.")
    p.add_argument("--limit", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--fp32", action="store_true", help="Disable FP16 on CUDA.")
    p.add_argument("--prompt-model-dir", default=None)
    p.add_argument("--moderation-model-dir", default=None)
    p.add_argument("--exclude-label", action="append", default=sorted(WEAK_DEFAULTS))
    p.add_argument("--include-label", action="append", default=[])
    p.add_argument("--min-label-recall", type=float, default=0.85,
                   help="Minimum recall enforced when picking BERT label thresholds.")
    p.add_argument("--label-beta", type=float, default=1.5,
                   help="F-beta weight favouring recall over precision for label thresholds.")
    p.add_argument("--min-route-recall", type=float, default=0.97,
                   help="Minimum recall for routing (must route this fraction of unsafe rows to a BERT).")
    p.add_argument("--max-safe-false-pass-pct", type=float, default=0.5,
                   help="Max %% of unsafe rows allowed through the FastText direct-safe shortcut.")
    p.add_argument("--max-fast-allow-false-pass-pct", type=float, default=1.0,
                   help="Max %% of unsafe rows allowed through the fast-allow (skip-both-BERTs) path.")
    p.add_argument("--min-direct-precision", type=float, default=0.95,
                   help="Min precision for FastText direct classification shortcuts.")
    p.add_argument("--output", default="reports/calibration_pytorch_50k.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    runtime_cfg = cfg.get("runtime", {})
    max_length = int(args.max_length or runtime_cfg.get("max_length", 128))
    fp16 = not args.fp32 and bool(runtime_cfg.get("fp16_on_cuda", True))

    excluded = {x.lower() for x in args.exclude_label}
    included = {x.lower() for x in args.include_label}
    rows = balanced_sample([Path(x) for x in args.data], args.limit, args.seed, excluded, included)
    if not rows:
        raise SystemExit("No rows matched. Check --data paths and include/exclude labels.")
    print(f"[calibrate] {len(rows)} rows — distribution: {dict(Counter(bucket_for(set(r['labels'])) for r in rows))}")

    prompt_dir = Path(args.prompt_model_dir) if args.prompt_model_dir else resolve_path(
        cfg["models"]["prompt_injection"]["local_path"]
    )
    moderation_dir = Path(args.moderation_model_dir) if args.moderation_model_dir else resolve_path(
        cfg["models"]["moderation"]["local_path"]
    )
    fasttext = FastTextRouter(resolve_path(cfg["models"]["fasttext_router"]["local_path"]))
    model_common = {"max_length": max_length, "device": args.device, "fp16": fp16}
    prompt_model = TorchTextClassifier(prompt_dir, sigmoid_outputs=False, **model_common)
    moderation_model = TorchTextClassifier(moderation_dir, sigmoid_outputs=True, **model_common)
    print(f"[calibrate] prompt fp16={prompt_model.fp16}  moderation fp16={moderation_model.fp16}")

    # ── Pass 1: normalize + FastText (CPU, cheap) ────────────────────────────
    started = time.perf_counter()
    model_texts: list[str] = []
    ft_score_list: list[dict[str, float]] = []
    rule_route_list: list[set[str]] = []

    for idx, row in enumerate(rows, start=1):
        norm = normalize(row["text"])
        rules = deterministic_gate(norm)
        ft = fasttext.predict(norm.detection_text)
        model_texts.append(norm.model_text)
        ft_score_list.append(ft["scores"])
        rule_route_list.append(set(rules.force_route))
        if idx % 5000 == 0:
            print(f"[calibrate] FastText pass: {idx}/{len(rows)}")

    ft_elapsed = (time.perf_counter() - started) * 1000
    print(f"[calibrate] FastText pass done in {ft_elapsed/1000:.1f}s")

    # ── Pass 2: both BERTs on all rows in batches (GPU) ─────────────────────
    print(f"[calibrate] running prompt BERT on all {len(rows)} rows …")
    t0 = time.perf_counter()
    prompt_raw_all, prompt_stats = prompt_model.predict_batches(model_texts, args.batch_size)
    print(f"[calibrate] prompt BERT done in {(time.perf_counter()-t0):.1f}s")

    print(f"[calibrate] running moderation BERT on all {len(rows)} rows …")
    t0 = time.perf_counter()
    mod_raw_all, mod_stats = moderation_model.predict_batches(model_texts, args.batch_size)
    print(f"[calibrate] moderation BERT done in {(time.perf_counter()-t0):.1f}s")

    # ── Collect scores ───────────────────────────────────────────────────────
    bert_scores: dict[str, list[float]] = {label: [] for label in PUBLIC_LABELS}
    for pi_raw, m_raw in zip(prompt_raw_all, mod_raw_all):
        bert_scores[PROMPT_INJECTION].append(prompt_score(pi_raw))
        ms = moderation_scores(m_raw)
        bert_scores[HARMFUL_CONTENT].append(ms[HARMFUL_CONTENT])
        bert_scores[SEXUAL].append(ms[SEXUAL])

    gold = make_gold(rows)
    total_elapsed = (time.perf_counter() - started) * 1000

    # ── Calibrate ────────────────────────────────────────────────────────────
    print("[calibrate] grid-searching thresholds …")
    label_thr = choose_label_thresholds(gold, bert_scores, min_recall=args.min_label_recall, beta=args.label_beta)
    routing_thr = choose_routing_thresholds(rows, ft_score_list, rule_route_list, min_route_recall=args.min_route_recall)
    fast_allow_thr = choose_fast_allow_threshold(rows, ft_score_list, rule_route_list, max_false_pass_pct=args.max_fast_allow_false_pass_pct)
    safe_thr = choose_safe_thresholds(rows, ft_score_list, rule_route_list, max_false_pass_pct=args.max_safe_false_pass_pct)
    direct_prompt = choose_direct_label_threshold(
        rows, ft_score_list, rule_route_list,
        score_key="attack", target_label=PROMPT_INJECTION,
        max_other_key="moderation", min_precision=args.min_direct_precision,
    )
    direct_harmful = choose_direct_label_threshold(
        rows, ft_score_list, rule_route_list,
        score_key="moderation", target_label=HARMFUL_CONTENT,
        max_other_key="attack", min_precision=args.min_direct_precision,
    )

    # ── Recommended config ───────────────────────────────────────────────────
    recommended = {
        "thresholds": {
            "attack_route": routing_thr.get("attack_route", 0.01),
            "moderation_route": routing_thr.get("moderation_route", 0.0),
            "fast_allow": fast_allow_thr["fast_allow"],
            "fasttext_direct_safe_score": safe_thr["fasttext_direct_safe_score"],
            "fasttext_direct_safe_max_route": safe_thr["fasttext_direct_safe_max_route"],
            "prompt_injection_review": label_thr[PROMPT_INJECTION]["threshold"],
            "harmful_content_review": label_thr[HARMFUL_CONTENT]["threshold"],
            "sexual_review": label_thr[SEXUAL]["threshold"],
            "fasttext_direct_prompt_injection_score": direct_prompt["score"] if direct_prompt["enabled"] else 1.1,
            "fasttext_direct_prompt_injection_max_moderation": direct_prompt["max_other"],
            "fasttext_direct_harmful_content_score": direct_harmful["score"] if direct_harmful["enabled"] else 1.1,
            "fasttext_direct_harmful_content_max_attack": direct_harmful["max_other"],
        },
        "runtime": {
            "fasttext_direct_safe_enabled": safe_thr["passed_rows"] > 0,
            "fasttext_direct_classification_enabled": direct_prompt["enabled"] or direct_harmful["enabled"],
        },
    }

    report = {
        "rows": len(rows),
        "excluded_labels": sorted(excluded),
        "included_labels": sorted(included),
        "sample_distribution": dict(Counter(bucket_for(set(r["labels"])) for r in rows)),
        "raw_label_distribution": dict(Counter(
            label for row in rows for label in row.get("raw_labels", [])
        )),
        "device": args.device,
        "fp16": {"prompt": prompt_model.fp16, "moderation": moderation_model.fp16},
        "batch_size": args.batch_size,
        "max_length": max_length,
        "targets": {
            "min_label_recall": args.min_label_recall,
            "label_beta": args.label_beta,
            "min_route_recall": args.min_route_recall,
            "max_safe_false_pass_pct": args.max_safe_false_pass_pct,
            "max_fast_allow_false_pass_pct": args.max_fast_allow_false_pass_pct,
            "min_direct_precision": args.min_direct_precision,
        },
        "recommended": recommended,
        "label_thresholds": label_thr,
        "routing_thresholds": routing_thr,
        "fast_allow_thresholds": fast_allow_thr,
        "safe_thresholds": safe_thr,
        "direct_fasttext_thresholds": {
            "prompt_injection": direct_prompt,
            "harmful_content": direct_harmful,
        },
        "latency_ms": {
            "wall_total": round(total_elapsed, 3),
            "fasttext_pass": round(ft_elapsed, 3),
            "prompt_pytorch": {k: v for k, v in prompt_stats.items() if k != "token_lengths"},
            "moderation_pytorch": {k: v for k, v in mod_stats.items() if k != "token_lengths"},
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    thr = recommended["thresholds"]
    rt = recommended["runtime"]
    yaml_lines = [
        "thresholds:",
        f"  attack_route: {thr['attack_route']}",
        f"  moderation_route: {thr['moderation_route']}",
        f"  fast_allow: {thr['fast_allow']}",
        f"  fasttext_direct_safe_score: {thr['fasttext_direct_safe_score']}",
        f"  fasttext_direct_safe_max_route: {thr['fasttext_direct_safe_max_route']}",
        f"  prompt_injection_review: {thr['prompt_injection_review']}",
        f"  harmful_content_review: {thr['harmful_content_review']}",
        f"  sexual_review: {thr['sexual_review']}",
        f"  fasttext_direct_prompt_injection_score: {thr['fasttext_direct_prompt_injection_score']}",
        f"  fasttext_direct_prompt_injection_max_moderation: {thr['fasttext_direct_prompt_injection_max_moderation']}",
        f"  fasttext_direct_harmful_content_score: {thr['fasttext_direct_harmful_content_score']}",
        f"  fasttext_direct_harmful_content_max_attack: {thr['fasttext_direct_harmful_content_max_attack']}",
        "runtime:",
        f"  fasttext_direct_safe_enabled: {'true' if rt['fasttext_direct_safe_enabled'] else 'false'}",
        f"  fasttext_direct_classification_enabled: {'true' if rt['fasttext_direct_classification_enabled'] else 'false'}",
        "",
    ]
    yaml_out = out.with_suffix(".recommended.yaml")
    yaml_out.write_text("\n".join(yaml_lines), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"[calibrate] wrote {out}")
    print(f"[calibrate] wrote {yaml_out}")


if __name__ == "__main__":
    main()
