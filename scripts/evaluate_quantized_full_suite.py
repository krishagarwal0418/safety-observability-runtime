#!/usr/bin/env python3
"""CPU-only batched evaluation of the quantized full runtime stack."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from safety_observability.config import load_config, resolve_path
from safety_observability.constants import HARMFUL_CONTENT, PROMPT_INJECTION, PUBLIC_LABELS, SEXUAL
from safety_observability.deterministic import ATTACK, MODERATION, evaluate as deterministic_gate
from safety_observability.fasttext_router import FastTextRouter
from safety_observability.normalize import normalize


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
PROMPT_ALIASES = {"prompt_injection", "injection", "inject", "jailbreak", "attack", "malicious"}


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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
        if str(row.get(label)).lower() in ("1", "true", "yes", "positive"):
            out.add(label)
    if str(row.get("is_prompt_injection")).lower() in ("1", "true", "yes"):
        out.add(PROMPT_INJECTION)
    return out


def bucket_for(labels: set[str]) -> str:
    if PROMPT_INJECTION in labels:
        return PROMPT_INJECTION
    if SEXUAL in labels:
        return SEXUAL
    if HARMFUL_CONTENT in labels:
        return HARMFUL_CONTENT
    return "safe"


def unique_key(text: str) -> str:
    return " ".join(text.lower().split())[:1000]


def balanced_sample(paths: list[Path], limit: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for path in paths:
        for row in read_rows(path):
            text = pick_text(row)
            if not text:
                continue
            key = unique_key(text)
            if key in seen:
                continue
            seen.add(key)
            labels = labels_for(row)
            rec = {"text": text, "labels": sorted(labels), "source_path": str(path)}
            buckets[bucket_for(labels)].append(rec)
    for rows in buckets.values():
        rng.shuffle(rows)

    names = ["safe", PROMPT_INJECTION, HARMFUL_CONTENT, SEXUAL]
    target = max(1, limit // len(names))
    selected: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    for name in names:
        take = buckets.get(name, [])[:target]
        selected.extend(take)
        used_ids.update(id(x) for x in take)

    remaining = []
    for name in names:
        remaining.extend(x for x in buckets.get(name, []) if id(x) not in used_ids)
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, limit - len(selected))])
    rng.shuffle(selected)
    return selected[:limit]


def binary_metrics(gold: list[int], pred: list[int]) -> dict[str, float]:
    tp = sum(1 for g, p in zip(gold, pred) if g and p)
    fp = sum(1 for g, p in zip(gold, pred) if not g and p)
    fn = sum(1 for g, p in zip(gold, pred) if g and not p)
    tn = sum(1 for g, p in zip(gold, pred) if not g and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def percentile(values: list[float], q: int) -> float:
    if not values:
        return 0.0
    return round(float(statistics.quantiles(values, n=100, method="inclusive")[q - 1]), 3)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class OnnxTextClassifier:
    def __init__(self, model_dir: Path, *, max_length: int, sigmoid_outputs: bool) -> None:
        self.model_dir = model_dir
        self.max_length = max_length
        self.sigmoid_outputs = sigmoid_outputs
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        cfg_path = model_dir / "config.json"
        self.id2label: dict[int, str] = {}
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            self.id2label = {int(k): str(v) for k, v in cfg.get("id2label", {}).items()}
        onnx_files = sorted(model_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX file found in {model_dir}")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(onnx_files[0]),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {i.name for i in self.session.get_inputs()}

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
                return_tensors="np",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
            token_lengths.extend(int(x) for x in np.sum(enc["attention_mask"], axis=1).tolist())
            ort_inputs = {k: v for k, v in enc.items() if k in self.input_names}
            t0 = time.perf_counter()
            logits = self.session.run(None, ort_inputs)[0]
            elapsed = (time.perf_counter() - t0) * 1000
            batch_latencies.append(elapsed)
            rows_per_batch.append(len(batch))
            probs = sigmoid(logits) if self.sigmoid_outputs else np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
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
        }
        return outputs, stats


def prompt_score(raw: dict[str, float]) -> float:
    score = 0.0
    for label, prob in raw.items():
        low = label.lower()
        if any(tok in low for tok in ("injection", "malicious", "attack", "unsafe")) or label == "LABEL_1":
            score = max(score, prob)
    return score


def moderation_scores(raw: dict[str, float]) -> dict[str, float]:
    harmful = 0.0
    sexual = 0.0
    for label, prob in raw.items():
        low = label.lower()
        if "sexual" in low or low in ("s", "s3") or label == "LABEL_1":
            sexual = max(sexual, prob)
        if (
            "harmful" in low
            or "hate" in low
            or "harassment" in low
            or "toxic" in low
            or "violence" in low
            or "self" in low
            or low in ("h", "h2", "hr", "sh", "v", "v2")
            or label == "LABEL_0"
        ):
            harmful = max(harmful, prob)
    return {HARMFUL_CONTENT: harmful, SEXUAL: sexual}


def token_bucket(n: int) -> str:
    if n <= 32:
        return "001-032"
    if n <= 64:
        return "033-064"
    if n <= 128:
        return "065-128"
    return "129+"


def pct(num: int, den: int) -> float:
    return round(100 * num / den, 2) if den else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/runtime.yaml")
    p.add_argument("--data", action="append", required=True, help="JSONL/CSV/JSON source. Pass multiple times.")
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--prompt-onnx-dir", default=None)
    p.add_argument("--moderation-onnx-dir", default="models/onnx_int8/moderation")
    p.add_argument("--output", default="reports/quantized_full_suite_5k.json")
    p.add_argument("--enable-direct-safe", action="store_true")
    p.add_argument("--fast-allow", type=float, default=None)
    p.add_argument("--attack-route", type=float, default=None)
    p.add_argument("--moderation-route", type=float, default=None)
    p.add_argument("--direct-safe-score", type=float, default=0.995)
    p.add_argument("--direct-safe-max-route", type=float, default=0.01)
    args = p.parse_args()

    cfg = load_config(args.config)
    thresholds = cfg["thresholds"]
    if args.fast_allow is not None:
        thresholds["fast_allow"] = args.fast_allow
    if args.attack_route is not None:
        thresholds["attack_route"] = args.attack_route
    if args.moderation_route is not None:
        thresholds["moderation_route"] = args.moderation_route
    thresholds["fasttext_direct_safe_score"] = args.direct_safe_score
    thresholds["fasttext_direct_safe_max_route"] = args.direct_safe_max_route

    max_length = int(args.max_length or cfg.get("runtime", {}).get("max_length", 128))
    prompt_dir = Path(args.prompt_onnx_dir) if args.prompt_onnx_dir else resolve_path(
        cfg["models"]["prompt_injection_onnx_int8"]["local_path"]
    )
    moderation_dir = Path(args.moderation_onnx_dir)
    fasttext = FastTextRouter(resolve_path(cfg["models"]["fasttext_router"]["local_path"]))
    prompt_model = OnnxTextClassifier(prompt_dir, max_length=max_length, sigmoid_outputs=False)
    moderation_model = OnnxTextClassifier(moderation_dir, max_length=max_length, sigmoid_outputs=True)

    rows = balanced_sample([Path(x) for x in args.data], args.limit, args.seed)
    print("[suite] sampled labels:", dict(Counter(bucket_for(set(r["labels"])) for r in rows)))

    started = time.perf_counter()
    prepped = []
    routing_counts = Counter()
    route_indices = {"prompt": [], "moderation": []}
    route_texts = {"prompt": [], "moderation": []}
    gold = {label: [] for label in PUBLIC_LABELS}
    pred = {label: [0] * len(rows) for label in PUBLIC_LABELS}
    ft_latencies: list[float] = []
    norm_latencies: list[float] = []
    prompt_token_lengths: list[int] = []
    moderation_token_lengths: list[int] = []

    for i, row in enumerate(rows):
        t0 = time.perf_counter()
        norm = normalize(row["text"])
        rules = deterministic_gate(norm)
        norm_latencies.append((time.perf_counter() - t0) * 1000)
        ft = fasttext.predict(norm.detection_text)
        ft_latencies.append(float(ft["latency_ms"]))
        ft_scores = ft["scores"]
        run_attack = ATTACK in rules.force_route or ft_scores["attack"] >= thresholds["attack_route"]
        run_moderation = MODERATION in rules.force_route or ft_scores["moderation"] >= thresholds["moderation_route"]
        direct_safe = (
            args.enable_direct_safe
            and rules.allow_fast_skip
            and ft_scores.get("safe", 0.0) >= thresholds["fasttext_direct_safe_score"]
            and ft_scores["attack"] < thresholds["fasttext_direct_safe_max_route"]
            and ft_scores["moderation"] < thresholds["fasttext_direct_safe_max_route"]
        )
        fast_allow_safe = (
            not direct_safe
            and rules.allow_fast_skip
            and ft_scores["attack"] < thresholds["fast_allow"]
            and ft_scores["moderation"] < thresholds["fast_allow"]
        )
        if direct_safe:
            routing_counts["fasttext_direct_safe"] += 1
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
        prepped.append({"labels": labels, "text": norm.model_text})

    prompt_raw, prompt_stats = prompt_model.predict_batches(route_texts["prompt"], args.batch_size)
    for idx, raw in zip(route_indices["prompt"], prompt_raw):
        if prompt_score(raw) >= thresholds["prompt_injection_review"]:
            pred[PROMPT_INJECTION][idx] = 1

    mod_raw, mod_stats = moderation_model.predict_batches(route_texts["moderation"], args.batch_size)
    for idx, raw in zip(route_indices["moderation"], mod_raw):
        scores = moderation_scores(raw)
        if scores[HARMFUL_CONTENT] >= thresholds["harmful_content_review"]:
            pred[HARMFUL_CONTENT][idx] = 1
        if scores[SEXUAL] >= thresholds["sexual_review"]:
            pred[SEXUAL][idx] = 1

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
        "cpu_only": True,
        "batch_size": args.batch_size,
        "max_length": max_length,
        "models": {
            "fasttext": str(resolve_path(cfg["models"]["fasttext_router"]["local_path"])),
            "prompt_onnx_int8": str(prompt_dir),
            "moderation_onnx_int8": str(moderation_dir),
        },
        "thresholds": {
            "attack_route": thresholds["attack_route"],
            "moderation_route": thresholds["moderation_route"],
            "fast_allow": thresholds["fast_allow"],
            "fasttext_direct_safe_enabled": args.enable_direct_safe,
            "fasttext_direct_safe_score": thresholds["fasttext_direct_safe_score"],
            "fasttext_direct_safe_max_route": thresholds["fasttext_direct_safe_max_route"],
            "prompt_injection_review": thresholds["prompt_injection_review"],
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
            "fasttext_direct_classified_rows": routing_counts["fasttext_direct_safe"] + routing_counts["fasttext_fast_allow_safe"],
            "fasttext_direct_classified_pct": pct(
                routing_counts["fasttext_direct_safe"] + routing_counts["fasttext_fast_allow_safe"], len(rows)
            ),
            "route_any_bert_vs_unsafe": binary_metrics(unsafe, routed_any),
        },
        "latency_ms": {
            "wall_total": round(elapsed_ms, 3),
            "throughput_rows_per_sec": round(1000 * len(rows) / elapsed_ms, 2) if elapsed_ms else 0.0,
            "estimated_per_row_avg": round(elapsed_ms / len(rows), 3) if rows else 0.0,
            "normalization_avg": round(sum(norm_latencies) / len(norm_latencies), 4) if norm_latencies else 0.0,
            "fasttext_avg": round(sum(ft_latencies) / len(ft_latencies), 4) if ft_latencies else 0.0,
            "prompt_onnx": {k: v for k, v in prompt_stats.items() if k != "token_lengths"},
            "moderation_onnx": {k: v for k, v in mod_stats.items() if k != "token_lengths"},
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
