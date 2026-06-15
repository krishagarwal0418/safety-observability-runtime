#!/usr/bin/env python3
"""Batched PyTorch evaluation of the full runtime stack.

This is the pragmatic GPU path: use the downloaded transformer repos directly
with CUDA and FP16 instead of converting the models through ONNX.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    percentile,
    prompt_score,
    resolve_path,
    token_bucket,
)


def load_tokenizer(path: str | Path):
    try:
        return AutoTokenizer.from_pretrained(str(path), fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(str(path))


class TorchTextClassifier:
    def __init__(
        self,
        model_dir: Path,
        *,
        max_length: int,
        sigmoid_outputs: bool,
        device: str,
        fp16: bool,
    ) -> None:
        self.model_dir = model_dir
        self.max_length = max_length
        self.sigmoid_outputs = sigmoid_outputs
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.tokenizer = load_tokenizer(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        self.model.to(self.device)
        cfg = getattr(self.model, "config", None)
        self.id2label = {int(k): v for k, v in getattr(cfg, "id2label", {}).items()} if cfg else {}
        # DeBERTa v1 attention masking overflows FP16 — only enable autocast for v2/v3+
        model_type = getattr(cfg, "model_type", "")
        self.fp16 = bool(fp16 and self.device == "cuda" and model_type != "deberta")

    def token_count(self, text: str) -> int:
        return len(self.tokenizer(text, truncation=True, max_length=self.max_length)["input_ids"])

    def predict_batches(self, texts: list[str], batch_size: int) -> tuple[list[dict[str, float]], dict[str, Any]]:
        outputs: list[dict[str, float]] = []
        batch_latencies: list[float] = []
        rows_per_batch: list[int] = []
        token_lengths: list[int] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
            token_lengths.extend(int(x) for x in enc["attention_mask"].sum(dim=1).tolist())
            enc = {k: v.to(self.device) for k, v in enc.items()}
            if self.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                logits = self.model(**enc).logits.float()
                probs = torch.sigmoid(logits) if self.sigmoid_outputs else torch.softmax(logits, dim=-1)
                probs = probs.detach().cpu()
            if self.device == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000
            batch_latencies.append(elapsed)
            rows_per_batch.append(len(batch))
            for row in probs:
                outputs.append({self.id2label.get(i, f"LABEL_{i}"): float(row[i]) for i in range(len(row))})
        per_row = [
            batch_ms / max(1, rows)
            for batch_ms, rows in zip(batch_latencies, rows_per_batch)
            for _ in range(rows)
        ]
        stats = {
            "batches": len(batch_latencies),
            "rows": len(texts),
            "batch_ms_total": round(sum(batch_latencies), 3),
            "per_row_ms_avg": round(sum(per_row) / len(per_row), 3) if per_row else 0.0,
            "per_row_ms_p50": percentile(per_row, 50),
            "per_row_ms_p95": percentile(per_row, 95),
            "token_lengths": token_lengths,
            "device": self.device,
            "fp16": self.fp16,
        }
        return outputs, stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/runtime.yaml")
    p.add_argument("--data", action="append", required=True, help="JSONL/CSV/JSON source. Pass multiple times.")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--fp32", action="store_true", help="Disable FP16 on CUDA.")
    p.add_argument("--prompt-model-dir", default=None)
    p.add_argument("--moderation-model-dir", default=None)
    p.add_argument("--output", default="reports/final_gpu_pytorch_fp16_1k.json")
    p.add_argument("--threshold-report", default=None)
    p.add_argument("--exclude-label", action="append", default=[])
    p.add_argument("--include-label", action="append", default=[])
    p.add_argument("--full-scan", action="store_true",
                   help="Run both BERTs on every row, bypassing FastText routing.")
    p.add_argument("--enable-fasttext-direct-classification", action="store_true")
    p.add_argument("--enable-direct-safe", action="store_true")
    p.add_argument("--fast-allow", type=float, default=None)
    p.add_argument("--attack-route", type=float, default=None)
    p.add_argument("--moderation-route", type=float, default=None)
    p.add_argument("--direct-safe-score", type=float, default=None)
    p.add_argument("--direct-safe-max-route", type=float, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    thresholds = cfg["thresholds"]
    runtime_cfg = cfg.get("runtime", {})
    if runtime_cfg.get("full_scan_default"):
        args.full_scan = True
    if runtime_cfg.get("fasttext_direct_classification_enabled") and not args.full_scan:
        args.enable_fasttext_direct_classification = True
    if runtime_cfg.get("fasttext_direct_safe_enabled") and not args.full_scan:
        args.enable_direct_safe = True
    if args.threshold_report:
        report = json.loads(Path(args.threshold_report).read_text(encoding="utf-8"))
        thresholds.update(report.get("recommended", {}).get("thresholds", {}))
        recommended_runtime = report.get("recommended", {}).get("runtime", {})
        if recommended_runtime.get("fasttext_direct_classification_enabled"):
            args.enable_fasttext_direct_classification = True
        if recommended_runtime.get("fasttext_direct_safe_enabled"):
            args.enable_direct_safe = True
    if args.fast_allow is not None:
        thresholds["fast_allow"] = args.fast_allow
    if args.attack_route is not None:
        thresholds["attack_route"] = args.attack_route
    if args.moderation_route is not None:
        thresholds["moderation_route"] = args.moderation_route
    if args.direct_safe_score is not None:
        thresholds["fasttext_direct_safe_score"] = args.direct_safe_score
    if args.direct_safe_max_route is not None:
        thresholds["fasttext_direct_safe_max_route"] = args.direct_safe_max_route

    max_length = int(args.max_length or runtime_cfg.get("max_length", 128))
    prompt_dir = Path(args.prompt_model_dir) if args.prompt_model_dir else resolve_path(
        cfg["models"]["prompt_injection"]["local_path"]
    )
    moderation_dir = Path(args.moderation_model_dir) if args.moderation_model_dir else resolve_path(
        cfg["models"]["moderation"]["local_path"]
    )
    fasttext = FastTextRouter(resolve_path(cfg["models"]["fasttext_router"]["local_path"]))
    fp16 = not args.fp32 and bool(runtime_cfg.get("fp16_on_cuda", True))
    common = {
        "max_length": max_length,
        "device": args.device,
        "fp16": fp16,
    }
    prompt_model = TorchTextClassifier(prompt_dir, sigmoid_outputs=False, **common)
    moderation_model = TorchTextClassifier(moderation_dir, sigmoid_outputs=True, **common)

    rows = balanced_sample(
        [Path(x) for x in args.data],
        args.limit,
        args.seed,
        {x.lower() for x in args.exclude_label},
        {x.lower() for x in args.include_label},
    )
    if not rows:
        raise SystemExit(
            "No evaluation rows matched. Check --data paths and include/exclude labels. "
            "Run a file count/label inspection before evaluating."
        )
    print("[suite] sampled labels:", dict(Counter(bucket_for(set(r["labels"])) for r in rows)))

    started = time.perf_counter()
    routing_counts = Counter()
    route_indices = {"prompt": [], "moderation": []}
    route_texts = {"prompt": [], "moderation": []}
    gold = {label: [] for label in PUBLIC_LABELS}
    pred = {label: [0] * len(rows) for label in PUBLIC_LABELS}
    ft_latencies: list[float] = []
    norm_latencies: list[float] = []
    prompt_token_lengths: list[int] = []
    moderation_token_lengths: list[int] = []
    direct_stats = Counter()

    for i, row in enumerate(rows):
        t0 = time.perf_counter()
        norm = normalize(row["text"])
        rules = deterministic_gate(norm)
        norm_latencies.append((time.perf_counter() - t0) * 1000)
        ft = fasttext.predict(norm.detection_text)
        ft_latencies.append(float(ft["latency_ms"]))
        ft_scores = ft["scores"]
        run_attack = args.full_scan or ATTACK in rules.force_route or ft_scores["attack"] >= thresholds["attack_route"]
        run_moderation = args.full_scan or MODERATION in rules.force_route or ft_scores["moderation"] >= thresholds["moderation_route"]
        direct_prompt = (
            not args.full_scan
            and args.enable_fasttext_direct_classification
            and rules.allow_fast_skip
            and ft_scores["attack"] >= thresholds.get("fasttext_direct_prompt_injection_score", 1.1)
            and ft_scores["moderation"] <= thresholds.get("fasttext_direct_prompt_injection_max_moderation", 0.0)
        )
        direct_harmful = (
            not args.full_scan
            and args.enable_fasttext_direct_classification
            and not direct_prompt
            and rules.allow_fast_skip
            and ft_scores["moderation"] >= thresholds.get("fasttext_direct_harmful_content_score", 1.1)
            and ft_scores["attack"] <= thresholds.get("fasttext_direct_harmful_content_max_attack", 0.0)
        )
        direct_safe = (
            not args.full_scan
            and args.enable_direct_safe
            and not direct_prompt
            and not direct_harmful
            and rules.allow_fast_skip
            and ft_scores.get("safe", 0.0) >= thresholds["fasttext_direct_safe_score"]
            and ft_scores["attack"] < thresholds["fasttext_direct_safe_max_route"]
            and ft_scores["moderation"] < thresholds["fasttext_direct_safe_max_route"]
        )
        fast_allow_safe = (
            not args.full_scan
            and not direct_safe
            and rules.allow_fast_skip
            and ft_scores["attack"] < thresholds["fast_allow"]
            and ft_scores["moderation"] < thresholds["fast_allow"]
        )
        if direct_prompt:
            pred[PROMPT_INJECTION][i] = 1
            routing_counts["fasttext_direct_prompt_injection"] += 1
            direct_stats["prompt_injection_tp" if PROMPT_INJECTION in row["labels"] else "prompt_injection_fp"] += 1
        elif direct_harmful:
            pred[HARMFUL_CONTENT][i] = 1
            routing_counts["fasttext_direct_harmful_content"] += 1
            direct_stats["harmful_content_tp" if HARMFUL_CONTENT in row["labels"] else "harmful_content_fp"] += 1
        elif direct_safe:
            routing_counts["fasttext_direct_safe"] += 1
            direct_stats["safe_fp" if row["labels"] else "safe_tn"] += 1
        elif fast_allow_safe:
            routing_counts["fasttext_fast_allow_safe"] += 1
        else:
            if run_attack:
                route_indices["prompt"].append(i)
                route_texts["prompt"].append(norm.model_text)
            if run_moderation:
                route_indices["moderation"].append(i)
                route_texts["moderation"].append(norm.model_text)
            if run_attack and run_moderation:
                routing_counts["routed_both_berts"] += 1
            elif run_attack:
                routing_counts["routed_prompt_bert"] += 1
            elif run_moderation:
                routing_counts["routed_moderation_bert"] += 1
            else:
                routing_counts["no_route_no_fast_safe"] += 1
        if rules.force_route:
            routing_counts["deterministic_rule_forced"] += 1
        labels = set(row["labels"])
        for label in PUBLIC_LABELS:
            gold[label].append(1 if label in labels else 0)
        prompt_token_lengths.append(prompt_model.token_count(norm.model_text))
        moderation_token_lengths.append(moderation_model.token_count(norm.model_text))

    # Run both BERTs, then label. Moderation is scored first so the injection
    # label can be vetoed when a row is strongly harmful (toxic content the
    # injection model mistakes for an attack).
    prompt_raw, prompt_stats = prompt_model.predict_batches(route_texts["prompt"], args.batch_size)
    mod_raw, mod_stats = moderation_model.predict_batches(route_texts["moderation"], args.batch_size)

    mod_harmful_by_idx: dict[int, float] = {}
    for idx, raw in zip(route_indices["moderation"], mod_raw):
        scores = moderation_scores(raw)
        mod_harmful_by_idx[idx] = scores[HARMFUL_CONTENT]
        if scores[HARMFUL_CONTENT] >= thresholds["harmful_content_review"]:
            pred[HARMFUL_CONTENT][idx] = 1
        if scores[SEXUAL] >= thresholds["sexual_review"]:
            pred[SEXUAL][idx] = 1

    max_harmful = thresholds.get("prompt_injection_max_harmful", 1.01)
    for idx, raw in zip(route_indices["prompt"], prompt_raw):
        if (
            prompt_score(raw) >= thresholds["prompt_injection_review"]
            and mod_harmful_by_idx.get(idx, 0.0) < max_harmful
        ):
            pred[PROMPT_INJECTION][idx] = 1

    elapsed_ms = (time.perf_counter() - started) * 1000
    per_label = {label: binary_metrics(gold[label], pred[label]) for label in PUBLIC_LABELS}
    macro_f1 = sum(m["f1"] for m in per_label.values()) / len(per_label)
    unsafe = [1 if any(gold[label][i] for label in PUBLIC_LABELS) else 0 for i in range(len(rows))]
    predicted_any = [1 if any(pred[label][i] for label in PUBLIC_LABELS) else 0 for i in range(len(rows))]
    routed_any = [0] * len(rows)
    for i in route_indices["prompt"] + route_indices["moderation"]:
        routed_any[i] = 1
    unsafe_false_pass = sum(1 for u, p, r in zip(unsafe, predicted_any, routed_any) if u and not p and not r)

    token_buckets: dict[str, dict[str, Any]] = {}
    for name, lengths in (("prompt_tokenizer", prompt_token_lengths), ("moderation_tokenizer", moderation_token_lengths)):
        counts = Counter(token_bucket(x) for x in lengths)
        token_buckets[name] = {
            "avg_tokens": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
            "p50_tokens": percentile([float(x) for x in lengths], 50),
            "p95_tokens": percentile([float(x) for x in lengths], 95),
            "buckets": dict(counts),
        }

    report = {
        "rows": len(rows),
        "sample_distribution": dict(Counter(bucket_for(set(r["labels"])) for r in rows)),
        "excluded_labels": sorted({x.lower() for x in args.exclude_label}),
        "included_labels": sorted({x.lower() for x in args.include_label}),
        "device": args.device,
        "fp16": fp16,
        "batch_size": args.batch_size,
        "max_length": max_length,
        "models": {
            "fasttext": str(resolve_path(cfg["models"]["fasttext_router"]["local_path"])),
            "prompt_pytorch": str(prompt_dir),
            "moderation_pytorch": str(moderation_dir),
        },
        "thresholds": {
            "attack_route": thresholds["attack_route"],
            "moderation_route": thresholds["moderation_route"],
            "fast_allow": thresholds["fast_allow"],
            "fasttext_direct_safe_enabled": args.enable_direct_safe,
            "fasttext_direct_classification_enabled": args.enable_fasttext_direct_classification,
            "fasttext_direct_safe_score": thresholds["fasttext_direct_safe_score"],
            "fasttext_direct_safe_max_route": thresholds["fasttext_direct_safe_max_route"],
            "fasttext_direct_prompt_injection_score": thresholds.get("fasttext_direct_prompt_injection_score"),
            "fasttext_direct_prompt_injection_max_moderation": thresholds.get("fasttext_direct_prompt_injection_max_moderation"),
            "prompt_injection_review": thresholds["prompt_injection_review"],
            "prompt_injection_max_harmful": thresholds.get("prompt_injection_max_harmful", 1.01),
            "harmful_content_review": thresholds["harmful_content_review"],
            "sexual_review": thresholds["sexual_review"],
        },
        "performance": {
            "label_metrics": per_label,
            "macro_f1": round(macro_f1, 4),
            "unsafe_any_detection": binary_metrics(unsafe, predicted_any),
            "unsafe_false_pass_rows": unsafe_false_pass,
            "unsafe_false_pass_pct_of_unsafe": pct(unsafe_false_pass, sum(unsafe)),
        },
        "routing": {
            **dict(routing_counts),
            "prompt_bert_rows": len(route_indices["prompt"]),
            "moderation_bert_rows": len(route_indices["moderation"]),
            "any_bert_rows": sum(routed_any),
            "any_bert_pct": pct(sum(routed_any), len(rows)),
            "fasttext_direct_classified_rows": (
                routing_counts["fasttext_direct_prompt_injection"]
                + routing_counts["fasttext_direct_harmful_content"]
                + routing_counts["fasttext_direct_safe"]
                + routing_counts["fasttext_fast_allow_safe"]
            ),
            "fasttext_direct_classified_pct": pct(
                routing_counts["fasttext_direct_prompt_injection"]
                + routing_counts["fasttext_direct_harmful_content"]
                + routing_counts["fasttext_direct_safe"]
                + routing_counts["fasttext_fast_allow_safe"],
                len(rows),
            ),
            "route_any_bert_vs_unsafe": binary_metrics(unsafe, routed_any),
            "fasttext_direct_precision": {
                "prompt_injection": round(
                    direct_stats["prompt_injection_tp"]
                    / max(1, direct_stats["prompt_injection_tp"] + direct_stats["prompt_injection_fp"]),
                    4,
                ),
                "harmful_content": round(
                    direct_stats["harmful_content_tp"]
                    / max(1, direct_stats["harmful_content_tp"] + direct_stats["harmful_content_fp"]),
                    4,
                ),
                "safe_false_pass_rows": direct_stats["safe_fp"],
            },
        },
        "latency_ms": {
            "wall_total": round(elapsed_ms, 3),
            "throughput_rows_per_sec": round(1000 * len(rows) / elapsed_ms, 2) if elapsed_ms else 0.0,
            "estimated_per_row_avg": round(elapsed_ms / len(rows), 3) if rows else 0.0,
            "normalization_avg": round(sum(norm_latencies) / len(norm_latencies), 4) if norm_latencies else 0.0,
            "fasttext_avg": round(sum(ft_latencies) / len(ft_latencies), 4) if ft_latencies else 0.0,
            "prompt_pytorch": {k: v for k, v in prompt_stats.items() if k != "token_lengths"},
            "moderation_pytorch": {k: v for k, v in mod_stats.items() if k != "token_lengths"},
        },
        "token_impact": token_buckets,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[suite] wrote {out}")


if __name__ == "__main__":
    main()
