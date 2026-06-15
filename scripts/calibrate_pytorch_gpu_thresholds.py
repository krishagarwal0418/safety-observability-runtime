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


def choose_injection_veto_threshold(
    gold: list[int],
    inj_scores: list[float],
    mod_harmful: list[float],
    *,
    min_recall: float,
    beta: float,
) -> dict[str, Any]:
    """2D search: label injection iff inj_score >= t_inj AND mod_harmful < max_harmful.

    The injection BERT cannot separate real injections from toxic content by its
    own score (both saturate near 1.0). But the moderation BERT can: toxic rows
    score high on harmful, real injections score low. So we veto the injection
    label using the moderation harmful score. `max_harmful = 1.01` means the veto
    is off (the toxic rows are not suppressed).
    """
    inj_grid = [round(x / 100, 2) for x in range(50, 96, 5)] + [0.97, 0.99]
    veto_grid = [1.01, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    best = None
    for t_inj in inj_grid:
        for max_harmful in veto_grid:
            pred = [
                1 if (s >= t_inj and h < max_harmful) else 0
                for s, h in zip(inj_scores, mod_harmful)
            ]
            m = binary_metrics(gold, pred)
            fb = fbeta(m["precision"], m["recall"], beta)
            cand = {
                "threshold": t_inj,
                "max_harmful": max_harmful,
                "veto_enabled": max_harmful <= 1.0,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "fbeta": round(fb, 4),
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
            }
            if m["recall"] >= min_recall:
                if best is None or (cand["precision"], cand["fbeta"]) > (best["precision"], best["fbeta"]):
                    best = cand
            elif best is None or m["recall"] > best["recall"]:
                best = cand
    return best or {"threshold": 0.5, "max_harmful": 1.01, "veto_enabled": False,
                    "precision": 0.0, "recall": 0.0, "f1": 0.0, "fbeta": 0.0}


def choose_route_threshold(
    rows: list[dict[str, Any]],
    ft_scores: list[dict[str, float]],
    rule_routes: list[set[str]],
    *,
    score_key: str,
    force_key: str,
    target_labels: set[str],
    min_recall: float,
) -> dict[str, Any]:
    """Pick the highest threshold for ONE route that still catches its OWN class.

    Each BERT has a job: attack route exists to catch prompt_injection, moderation
    route to catch harmful_content/sexual. Calibrating a route against *any* unsafe
    label (the old behaviour) lets moderation become a catch-all at route=0 while
    attack does nothing useful. We instead target each route at the label(s) that
    route is responsible for, and pick the highest threshold (fewest rows routed)
    that still recalls `min_recall` of that class.
    """
    grid = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    target = [1 if (set(row["labels"]) & target_labels) else 0 for row in rows]
    total_target = sum(target)
    curve = []
    best = None
    for thr in grid:
        routed = [
            1 if (force_key in forced or sc[score_key] >= thr) else 0
            for sc, forced in zip(ft_scores, rule_routes)
        ]
        tp = sum(1 for t, r in zip(target, routed) if t and r)
        recall = tp / total_target if total_target else 0.0
        n_routed = sum(routed)
        cand = {
            "threshold": thr,
            "target_recall": round(recall, 4),
            "routed_rows": n_routed,
            "routed_pct": pct(n_routed, len(rows)),
            "target_total": total_target,
            "target_routed": tp,
            "route_precision": round(tp / n_routed, 4) if n_routed else 0.0,
        }
        curve.append(cand)
        if recall >= min_recall:
            # Meets recall — prefer the threshold that routes the fewest rows.
            if best is None or n_routed < best["routed_rows"]:
                best = cand
        elif best is None or recall > best["target_recall"]:
            best = cand
    best = dict(best or {"threshold": 0.0})
    best["curve"] = curve
    return best


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
    # Calibration always runs FP32: we score ALL rows through both BERTs (not
    # just routing-selected rows), and FP16 autocast saturates scores on
    # out-of-distribution content (e.g. DeBERTa v3 injection scores → 1.0 for
    # hate/sexual rows), producing useless precision numbers.
    model_common = {"max_length": max_length, "device": args.device, "fp16": False}
    prompt_model = TorchTextClassifier(prompt_dir, sigmoid_outputs=False, **model_common)
    moderation_model = TorchTextClassifier(moderation_dir, sigmoid_outputs=True, **model_common)
    print(f"[calibrate] running FP32 (autocast disabled — calibration requires stable scores across all content types)")

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

    # ── Routing calibration first (FastText only, no BERT needed) ───────────
    # Each route is calibrated against the class it is responsible for:
    #   attack route      → prompt_injection recall
    #   moderation route  → harmful_content / sexual recall
    print("[calibrate] computing per-route thresholds from FastText scores …")
    attack_route_cal = choose_route_threshold(
        rows, ft_score_list, rule_route_list,
        score_key="attack", force_key=ATTACK,
        target_labels={PROMPT_INJECTION}, min_recall=args.min_route_recall,
    )
    mod_route_cal = choose_route_threshold(
        rows, ft_score_list, rule_route_list,
        score_key="moderation", force_key=MODERATION,
        target_labels={HARMFUL_CONTENT, SEXUAL}, min_recall=args.min_route_recall,
    )
    attack_thr = attack_route_cal["threshold"]
    mod_thr = mod_route_cal["threshold"]
    routing_thr = {
        "attack_route": attack_thr,
        "moderation_route": mod_thr,
        "attack_detail": attack_route_cal,
        "moderation_detail": mod_route_cal,
    }
    def _print_curve(name: str, cal: dict) -> None:
        print(f"[calibrate] {name} route recall/efficiency tradeoff "
              f"(min_route_recall={args.min_route_recall}):")
        print(f"            {'thr':>6} {'recall':>8} {'routed%':>9} {'route_P':>8}")
        for c in cal.get("curve", []):
            mark = " <- chosen" if c["threshold"] == cal["threshold"] else ""
            print(f"            {c['threshold']:>6} {c['target_recall']:>8} "
                  f"{c['routed_pct']:>9} {c['route_precision']:>8}{mark}")

    _print_curve("ATTACK (injection)", attack_route_cal)
    _print_curve("MODERATION (harmful/sexual)", mod_route_cal)
    print(f"[calibrate] routing: attack_route={attack_thr} "
          f"(injection recall {attack_route_cal['target_recall']}, "
          f"{attack_route_cal['routed_pct']}% routed)  "
          f"moderation_route={mod_thr} "
          f"(harmful/sexual recall {mod_route_cal['target_recall']}, "
          f"{mod_route_cal['routed_pct']}% routed)")
    if attack_thr == 0.0 or mod_thr == 0.0:
        print("[calibrate] WARNING: a route collapsed to 0.0 (routes 100%). The "
              "FastText head cannot hit the recall target above threshold 0. "
              "Lower --min-route-recall to route a subset, or accept full routing "
              "(max quality, lower throughput). See the tradeoff curve above.")

    # ── Filter rows to those that would actually be routed to each BERT ──────
    # Prompt injection BERT only sees high-attack-score rows in production.
    # Calibrating its threshold on hate/toxic/safe content causes false-positive
    # flooding — those rows never reach this model at inference time.
    prompt_indices = [
        i for i, (sc, forced) in enumerate(zip(ft_score_list, rule_route_list))
        if ATTACK in forced or sc["attack"] >= attack_thr
    ]
    mod_indices = [
        i for i, (sc, forced) in enumerate(zip(ft_score_list, rule_route_list))
        if MODERATION in forced or sc["moderation"] >= mod_thr
    ]
    prompt_texts = [model_texts[i] for i in prompt_indices]
    mod_texts = [model_texts[i] for i in mod_indices]
    print(f"[calibrate] prompt BERT rows: {len(prompt_texts)}  moderation BERT rows: {len(mod_texts)}")

    # ── Pass 2: BERTs on routed subsets only ────────────────────────────────
    t0 = time.perf_counter()
    print(f"[calibrate] running prompt BERT on {len(prompt_texts)} routed rows …")
    prompt_raw_routed, prompt_stats = prompt_model.predict_batches(prompt_texts, args.batch_size)
    print(f"[calibrate] prompt BERT done in {(time.perf_counter()-t0):.1f}s")

    t0 = time.perf_counter()
    print(f"[calibrate] running moderation BERT on {len(mod_texts)} routed rows …")
    mod_raw_routed, mod_stats = moderation_model.predict_batches(mod_texts, args.batch_size)
    print(f"[calibrate] moderation BERT done in {(time.perf_counter()-t0):.1f}s")

    # ── Collect scores aligned to routed subsets ─────────────────────────────
    gold_all = make_gold(rows)

    # Prompt injection: calibrate only on attack-routed rows
    prompt_gold = {label: [gold_all[label][i] for i in prompt_indices] for label in PUBLIC_LABELS}
    prompt_bert_scores: dict[str, list[float]] = {label: [0.0] * len(prompt_indices) for label in PUBLIC_LABELS}
    for j, raw in enumerate(prompt_raw_routed):
        prompt_bert_scores[PROMPT_INJECTION][j] = prompt_score(raw)

    # Moderation: calibrate only on moderation-routed rows. Also index the
    # harmful score by original row id so the injection veto can look it up.
    mod_gold = {label: [gold_all[label][i] for i in mod_indices] for label in PUBLIC_LABELS}
    mod_bert_scores: dict[str, list[float]] = {label: [0.0] * len(mod_indices) for label in PUBLIC_LABELS}
    mod_harmful_by_idx: dict[int, float] = {}
    for j, raw in enumerate(mod_raw_routed):
        ms = moderation_scores(raw)
        mod_bert_scores[HARMFUL_CONTENT][j] = ms[HARMFUL_CONTENT]
        mod_bert_scores[SEXUAL][j] = ms[SEXUAL]
        mod_harmful_by_idx[mod_indices[j]] = ms[HARMFUL_CONTENT]

    total_elapsed = (time.perf_counter() - started) * 1000

    # ── Calibrate ────────────────────────────────────────────────────────────
    print("[calibrate] grid-searching thresholds …")
    # Baseline (no veto) injection threshold, kept for comparison in the report.
    pi_label_thr = choose_label_thresholds(
        prompt_gold, prompt_bert_scores, min_recall=args.min_label_recall, beta=args.label_beta
    )
    # Injection veto: search injection-score × max-moderation jointly. Uses the
    # moderation harmful score for each attack-routed row (0.0 if that row was
    # not moderation-routed, which means moderation never flagged it).
    prompt_inj_scores = prompt_bert_scores[PROMPT_INJECTION]
    prompt_gold_inj = prompt_gold[PROMPT_INJECTION]
    prompt_mod_harmful = [mod_harmful_by_idx.get(i, 0.0) for i in prompt_indices]
    inj_veto = choose_injection_veto_threshold(
        prompt_gold_inj, prompt_inj_scores, prompt_mod_harmful,
        min_recall=args.min_label_recall, beta=args.label_beta,
    )
    print(f"[calibrate] injection: no-veto P={pi_label_thr[PROMPT_INJECTION]['precision']} "
          f"R={pi_label_thr[PROMPT_INJECTION]['recall']}  →  "
          f"veto(max_harmful={inj_veto['max_harmful']}) "
          f"P={inj_veto['precision']} R={inj_veto['recall']} F1={inj_veto['f1']}")
    mod_label_thr = choose_label_thresholds(
        mod_gold, mod_bert_scores, min_recall=args.min_label_recall, beta=args.label_beta
    )
    label_thr = {
        PROMPT_INJECTION: {
            "threshold": inj_veto["threshold"],
            "max_harmful": inj_veto["max_harmful"],
            "precision": inj_veto["precision"],
            "recall": inj_veto["recall"],
            "f1": inj_veto["f1"],
            "fbeta": inj_veto["fbeta"],
        },
        HARMFUL_CONTENT: mod_label_thr[HARMFUL_CONTENT],
        SEXUAL: mod_label_thr[SEXUAL],
    }
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
            "prompt_injection_max_harmful": label_thr[PROMPT_INJECTION]["max_harmful"],
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
        "fp16": False,
        "batch_size": args.batch_size,
        "max_length": max_length,
        "bert_routing": {
            "prompt_routed_rows": len(prompt_indices),
            "prompt_routed_pct": round(100 * len(prompt_indices) / len(rows), 2),
            "moderation_routed_rows": len(mod_indices),
            "moderation_routed_pct": round(100 * len(mod_indices) / len(rows), 2),
        },
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
        "injection_veto": {
            "no_veto_baseline": pi_label_thr[PROMPT_INJECTION],
            "with_veto": inj_veto,
        },
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
        f"  prompt_injection_max_harmful: {thr['prompt_injection_max_harmful']}",
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
