#!/usr/bin/env python3
"""Evaluate the current runtime on public HF datasets for the supported scope.

Supported labels in the current runtime:
  * prompt_injection
  * harmful_content, covering toxic/hate/harassment-like text and spam
  * safe, represented by no emitted labels

Sexual/self-harm/violence/dangerous-information/illegal-activity are out of
scope for the current MiniLM moderation substitution and are not evaluated here.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import tempfile
import time
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from safety_observability import constants as C
from safety_observability.config import load_config
from safety_observability.pipeline import SafetyObservabilityClassifier

LABELS = ("prompt_injection", "harmful_content", "safe")


def _text_column(df, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"none of these text columns exist: {candidates}; got {list(df.columns)}")


def _sample(items: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    rng.shuffle(items)
    return items[:n]


def load_prompt_injection(n: int, rng: random.Random) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows: list[dict[str, Any]] = []

    sg = load_dataset("xTRam1/safe-guard-prompt-injection", split="train").to_pandas()
    sg = sg[(sg["label"] == 1) & (sg["text"].str.strip() != "")]
    for text in sg["text"].drop_duplicates().tolist():
        rows.append({
            "text": text,
            "true_label": C.PROMPT_INJECTION,
            "source": "xTRam1/safe-guard-prompt-injection",
        })

    spml = load_dataset("reshabhs/SPML_Chatbot_Prompt_Injection", split="train").to_pandas()
    spml = spml[(spml["Prompt injection"] == 1) & (spml["User Prompt"].str.strip() != "")]
    for text in spml["User Prompt"].drop_duplicates().tolist():
        rows.append({
            "text": text,
            "true_label": C.PROMPT_INJECTION,
            "source": "reshabhs/SPML_Chatbot_Prompt_Injection",
        })

    return _sample(rows, n, rng)


def load_harmful(n: int, rng: random.Random) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows: list[dict[str, Any]] = []

    hate = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train").to_pandas()
    hate = hate[(hate["hate_speech_score"] >= 1.0) & (hate["text"].str.strip() != "")]
    for text in hate["text"].drop_duplicates().tolist():
        rows.append({
            "text": text,
            "true_label": C.HARMFUL_CONTENT,
            "source": "ucberkeley-dlab/measuring-hate-speech",
            "subtype": "hate",
        })

    toxic = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test").to_pandas()
    toxic = toxic[(toxic["toxicity"] == 1) & (toxic["user_input"].str.strip() != "")]
    for text in toxic["user_input"].drop_duplicates().tolist():
        rows.append({
            "text": text,
            "true_label": C.HARMFUL_CONTENT,
            "source": "lmsys/toxic-chat",
            "subtype": "toxic",
        })

    try:
        spam = load_dataset("ucirvine/sms_spam", split="train").to_pandas()
        msg_col = _text_column(spam, ("sms", "text", "message"))
        label_col = _text_column(spam, ("label",))
        spam = spam[(spam[label_col].astype(str).str.lower().isin(("spam", "1"))) & (spam[msg_col].str.strip() != "")]
        for text in spam[msg_col].drop_duplicates().tolist():
            rows.append({
                "text": text,
                "true_label": C.HARMFUL_CONTENT,
                "source": "ucirvine/sms_spam",
                "subtype": "spam",
            })
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] could not load ucirvine/sms_spam: {exc}")

    return _sample(rows, n, rng)


def load_safe(n: int, rng: random.Random) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows: list[dict[str, Any]] = []

    squad = load_dataset("rajpurkar/squad", split="validation").to_pandas()
    for text in squad["question"].drop_duplicates().tolist():
        if text and text.strip():
            rows.append({"text": text, "true_label": C.SAFE, "source": "rajpurkar/squad"})

    toxic = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test").to_pandas()
    toxic = toxic[(toxic["toxicity"] == 0) & (toxic["jailbreaking"] == 0) & (toxic["user_input"].str.strip() != "")]
    for text in toxic["user_input"].drop_duplicates().tolist():
        rows.append({"text": text, "true_label": C.SAFE, "source": "lmsys/toxic-chat"})

    return _sample(rows, n, rng)


def assemble(rows_per_label: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pools = {
        C.PROMPT_INJECTION: load_prompt_injection(rows_per_label * 2, rng),
        C.HARMFUL_CONTENT: load_harmful(rows_per_label * 2, rng),
        C.SAFE: load_safe(rows_per_label * 2, rng),
    }
    available = {label: len(pool) for label, pool in pools.items()}
    n = min(rows_per_label, *available.values())
    if n < rows_per_label:
        print(f"[WARN] balancing to {n}/label; available={available}")

    rows: list[dict[str, Any]] = []
    for label in LABELS:
        rows.extend(pools[label][:n])
    rng.shuffle(rows)
    print(f"[data] assembled {len(rows)} rows: {dict(Counter(row['true_label'] for row in rows))}")
    return rows


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
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
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return round(ordered[idx], 3)


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "avg": round(statistics.mean(values), 3),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": round(max(values), 3),
    }


def run_batched_models(clf, texts: list[str], *, prompt_batch_size: int, moderation_batch_size: int, parallel: bool):
    if parallel:
        print(
            "[eval] parallel batched inference "
            f"rows={len(texts)} prompt_batch_size={prompt_batch_size} "
            f"moderation_batch_size={moderation_batch_size}"
        )
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            prompt_future = pool.submit(
                clf.prompt_injection.classify_batch,
                texts,
                batch_size=prompt_batch_size,
            )
            moderation_future = pool.submit(
                clf.moderation.classify_batch,
                texts,
                batch_size=moderation_batch_size,
            )
            prompt_out = prompt_future.result()
            moderation_out = moderation_future.result()
        return prompt_out, moderation_out, time.perf_counter() - started

    print(
        "[eval] serial batched inference "
        f"rows={len(texts)} prompt_batch_size={prompt_batch_size} "
        f"moderation_batch_size={moderation_batch_size}"
    )
    started = time.perf_counter()
    prompt_out = clf.prompt_injection.classify_batch(texts, batch_size=prompt_batch_size)
    moderation_out = clf.moderation.classify_batch(texts, batch_size=moderation_batch_size)
    return prompt_out, moderation_out, time.perf_counter() - started


def write_eval_config(args: argparse.Namespace) -> str:
    cfg = load_config(args.config)
    if args.local_hf_upload:
        root = Path(args.local_hf_upload).resolve()
        cfg["models"]["fasttext_router"]["local_path"] = str(root / "safety-fasttext-router" / "router_head.ftz")
        cfg["models"]["prompt_injection"]["local_path"] = str(root / "safety-prompt-injection")
    moderation_spec = cfg["models"].get("moderation", {})
    if moderation_spec.get("backend") == "minilm_toxic_spam_onnx":
        local = Path(moderation_spec["local_path"])
        if not local.is_absolute():
            local = REPO / local
        if not (local / "onnx" / "model.onnx").exists():
            from huggingface_hub import snapshot_download

            print(f"[download] moderation: {moderation_spec['repo_id']} -> {local}")
            snapshot_download(
                moderation_spec["repo_id"],
                local_dir=str(local),
                allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "onnx/model.onnx"],
            )
    cfg.setdefault("runtime", {}).update({
        "device": args.device,
        "onnx_provider": args.onnx_provider,
        "max_length": args.max_length,
        "full_scan_default": True,
        "fasttext_direct_classification_enabled": False,
        "fasttext_direct_safe_enabled": False,
    })
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp, sort_keys=False)
    tmp.close()
    return tmp.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO / "configs/runtime.yaml"))
    parser.add_argument("--rows-per-label", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--onnx-provider", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prompt-batch-size", type=int, default=None)
    parser.add_argument("--moderation-batch-size", type=int, default=None)
    parser.add_argument("--parallel", action="store_true", help="run prompt model and moderation model concurrently")
    parser.add_argument("--warmup-rows", type=int, default=8)
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--local-hf-upload", default=None, help="use local hf_upload artifacts for private models")
    parser.add_argument("--output", default=str(REPO / "reports/scope_hf_eval.json"))
    args = parser.parse_args()

    rows = assemble(args.rows_per_label, args.seed)
    eval_config = write_eval_config(args)
    clf = SafetyObservabilityClassifier(eval_config)
    prompt_batch_size = args.prompt_batch_size or args.batch_size
    moderation_batch_size = args.moderation_batch_size or args.batch_size

    prompt_tokenizer = clf.prompt_injection.tokenizer
    moderation_tokenizer = getattr(clf.moderation, "tokenizer", None)

    confusion: dict[str, dict[str, int]] = {label: defaultdict(int) for label in LABELS}
    per_source: dict[str, Counter] = defaultdict(Counter)
    prompt_tokens: list[float] = []
    moderation_tokens: list[float] = []
    errors: list[dict[str, Any]] = []

    texts = [row["text"] for row in rows]
    for text in texts:
        prompt_tokens.append(len(prompt_tokenizer(text, truncation=False, verbose=False)["input_ids"]))
        if moderation_tokenizer is not None:
            moderation_tokens.append(len(moderation_tokenizer(text, truncation=False, verbose=False)["input_ids"]))

    warmup_texts = texts[: max(0, min(args.warmup_rows, len(texts)))]
    if warmup_texts:
        print(f"[warmup] {len(warmup_texts)} rows")
        clf.prompt_injection.classify_batch(warmup_texts, batch_size=min(prompt_batch_size, len(warmup_texts)))
        clf.moderation.classify_batch(warmup_texts, batch_size=min(moderation_batch_size, len(warmup_texts)))

    prompt_out, moderation_out, elapsed = run_batched_models(
        clf,
        texts,
        prompt_batch_size=prompt_batch_size,
        moderation_batch_size=moderation_batch_size,
        parallel=args.parallel,
    )

    prompt_scores = prompt_out["scores"]
    harmful_scores = moderation_out["scores"]
    thresholds = clf.config["thresholds"]

    for i, row in enumerate(rows):
        labels = []
        if prompt_scores[i] >= thresholds["prompt_injection_review"]:
            labels.append(C.PROMPT_INJECTION)
        if harmful_scores[i] >= thresholds["harmful_content_review"]:
            labels.append(C.HARMFUL_CONTENT)

        pred = C.SAFE
        if C.PROMPT_INJECTION in labels:
            pred = C.PROMPT_INJECTION
        elif C.HARMFUL_CONTENT in labels:
            pred = C.HARMFUL_CONTENT
        confusion[row["true_label"]][pred] += 1
        per_source[row["source"]][f"true:{row['true_label']}"] += 1
        per_source[row["source"]][f"pred:{pred}"] += 1

        if pred != row["true_label"] and len(errors) < 50:
            errors.append({
                "true_label": row["true_label"],
                "pred_label": pred,
                "source": row["source"],
                "subtype": row.get("subtype"),
                "scores": {
                    C.PROMPT_INJECTION: round(float(prompt_scores[i]), 4),
                    C.HARMFUL_CONTENT: round(float(harmful_scores[i]), 4),
                },
                "text_preview": row["text"][:180],
            })

    per_label = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(confusion[label][other] for other in LABELS if other != label)
        per_label[label] = prf(tp, fp, fn)
    macro_f1 = round(sum(float(per_label[label]["f1"]) for label in LABELS) / len(LABELS), 4)

    providers = None
    if hasattr(clf.moderation, "session"):
        providers = clf.moderation.session.get_providers()

    result = {
        "rows_per_label": args.rows_per_label,
        "total_rows": len(rows),
        "device": clf.device,
        "onnx_provider_requested": args.onnx_provider,
        "onnx_providers_active": providers,
        "max_length": args.max_length,
        "parallel": args.parallel,
        "batch_size": args.batch_size,
        "prompt_batch_size": prompt_batch_size,
        "moderation_batch_size": moderation_batch_size,
        "warmup_rows": len(warmup_texts),
        "thresholds": {
            "prompt_injection_review": clf.config["thresholds"]["prompt_injection_review"],
            "harmful_content_review": clf.config["thresholds"]["harmful_content_review"],
            "sexual_review": clf.config["thresholds"]["sexual_review"],
        },
        "macro_f1": macro_f1,
        "per_label": per_label,
        "confusion": {label: dict(confusion[label]) for label in LABELS},
        "per_source": {source: dict(counts) for source, counts in per_source.items()},
        "latency": {
            "batched_wall_seconds": round(elapsed, 3),
            "effective_ms_per_row": round((elapsed * 1000) / max(1, len(rows)), 3),
            "prompt_injection_total_ms": round(float(prompt_out["latency_ms"]), 3),
            "moderation_total_ms": round(float(moderation_out["latency_ms"]), 3),
            "prompt_injection_batch_ms": summarize_values(prompt_out["batch_latencies_ms"]),
            "moderation_batch_ms": summarize_values(moderation_out["batch_latencies_ms"]),
        },
        "throughput_rows_per_sec": round(len(rows) / elapsed if elapsed else 0.0, 3),
        "token_lengths": {
            "prompt_injection_tokenizer": summarize_values(prompt_tokens),
            "moderation_tokenizer": summarize_values(moderation_tokens),
            "prompt_rows_over_max_length": sum(1 for value in prompt_tokens if value > args.max_length),
            "moderation_rows_over_max_length": sum(1 for value in moderation_tokens if value > args.max_length),
        },
        "triggered_models": {
            "prompt_injection": len(rows),
            "moderation": len(rows),
        },
        "error_samples": errors,
        "datasets": [
            "xTRam1/safe-guard-prompt-injection",
            "reshabhs/SPML_Chatbot_Prompt_Injection",
            "ucberkeley-dlab/measuring-hate-speech",
            "lmsys/toxic-chat",
            "ucirvine/sms_spam",
            "rajpurkar/squad",
        ],
    }

    print("\nRESULTS")
    for label in LABELS:
        metric = per_label[label]
        print(
            f"{label:<18} P={metric['precision']:.4f} R={metric['recall']:.4f} "
            f"F1={metric['f1']:.4f} TP/FP/FN={metric['tp']}/{metric['fp']}/{metric['fn']}"
        )
    print(f"macro_f1={macro_f1:.4f}")
    print(f"latency={result['latency']}")
    print(f"throughput_rows_per_sec={result['throughput_rows_per_sec']}")
    print(f"prompt_tokens={result['token_lengths']['prompt_injection_tokenizer']}")
    print(f"moderation_tokens={result['token_lengths']['moderation_tokenizer']}")
    print(f"onnx_providers_active={providers}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
