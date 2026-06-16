#!/usr/bin/env python3
"""Compare the current 2-head moderation model with a candidate classifier.

The candidate model currently targeted by this script is:

    navodPeiris/minilm-toxic-spam-classifier

It is a 3-class ONNX softmax classifier (safe/toxic/spam), so it is not a
drop-in replacement for the current harmful_content + sexual sigmoid model.
By default this script compares the overlapping behavior:

  * harmful_content vs safe: current harmful head vs candidate toxic class

Pass --include-sexual to also run unsupported sexual sanity checks.

It also prints threshold sweeps so fixed-threshold numbers are not mistaken for
calibration proof.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CURRENT = REPO.parent / "hf_upload" / "safety-moderation-2head"
DEFAULT_CANDIDATE = "navodPeiris/minilm-toxic-spam-classifier"


def _load_harmful(n: int) -> list[dict]:
    from datasets import load_dataset

    print("[data] harmful positives: ucberkeley-dlab/measuring-hate-speech")
    df = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train").to_pandas()
    df = df[(df["hate_speech_score"] >= 1.0) & (df["text"].str.strip() != "")]
    texts = df["text"].drop_duplicates().tolist()
    random.shuffle(texts)
    return [{"text": t, "true_label": "harmful_content"} for t in texts[:n]]


def _load_sexual(n: int) -> list[dict]:
    from datasets import load_dataset

    print("[data] sexual positives: google/civil_comments sexual_explicit")
    texts: list[str] = []
    for split in ("test", "validation"):
        df = load_dataset("google/civil_comments", split=split).to_pandas()
        text_col = "comment_text" if "comment_text" in df.columns else "text"
        df = df[(df["sexual_explicit"] >= 0.4) & (df[text_col].str.strip() != "")]
        texts.extend(df[text_col].tolist())
    texts = list(dict.fromkeys(texts))
    random.shuffle(texts)
    return [{"text": t, "true_label": "sexual"} for t in texts[:n]]


def _load_safe(n: int) -> list[dict]:
    from datasets import load_dataset

    print("[data] safe negatives: SQuAD + benign civil_comments")
    sq = load_dataset("rajpurkar/squad", split="validation").to_pandas()
    texts = sq["question"].tolist()
    cc = load_dataset("google/civil_comments", split="test").to_pandas()
    text_col = "comment_text" if "comment_text" in cc.columns else "text"
    cc = cc[(cc["toxicity"] < 0.1) & (cc["sexual_explicit"] < 0.1) & (cc[text_col].str.strip() != "")]
    texts.extend(cc[text_col].tolist())
    texts = [t for t in texts if t and t.strip()]
    random.shuffle(texts)
    return [{"text": t, "true_label": "safe"} for t in texts[:n]]


def assemble(rows_per_label: int, seed: int, include_sexual: bool) -> list[dict]:
    random.seed(seed)
    pools = {
        "harmful_content": _load_harmful(rows_per_label * 2),
        "safe": _load_safe(rows_per_label * 2),
    }
    if include_sexual:
        pools["sexual"] = _load_sexual(rows_per_label * 2)
    n = min(rows_per_label, *(len(v) for v in pools.values()))
    rows: list[dict] = []
    for label, pool in pools.items():
        if not pool:
            raise RuntimeError(f"no rows loaded for {label}")
        rows.extend(pool[:n])
    random.shuffle(rows)
    print(f"[data] assembled {len(rows)} rows ({n}/label)")
    return rows


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def run_current(model_path: str, texts: list[str], device: str, batch_size: int, max_length: int) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).eval().to(device)
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    harmful_idx = next((i for i, label in id2label.items() if "harmful" in label), 0)
    sexual_idx = next((i for i, label in id2label.items() if "sexual" in label), 1)

    harmful: list[float] = []
    sexual: list[float] = []
    started = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            probs = torch.sigmoid(model(**enc).logits.float()).detach().cpu()
        harmful.extend(probs[:, harmful_idx].tolist())
        sexual.extend(probs[:, sexual_idx].tolist())
    elapsed = time.perf_counter() - started
    return {
        "harmful": harmful,
        "sexual": sexual,
        "latency_ms_per_row": (elapsed * 1000) / max(1, len(texts)),
        "rows_per_sec": len(texts) / elapsed if elapsed else 0.0,
    }


def _candidate_snapshot(repo_or_path: str) -> Path:
    p = Path(repo_or_path)
    if p.exists():
        return p
    return Path(
        snapshot_download(
            repo_or_path,
            allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "onnx/model.onnx"],
        )
    )


def run_candidate(repo_or_path: str, texts: list[str], batch_size: int, max_length: int) -> dict[str, object]:
    root = _candidate_snapshot(repo_or_path)
    tokenizer = AutoTokenizer.from_pretrained(root)
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    id2label = {int(k): v.lower() for k, v in cfg["id2label"].items()}
    toxic_idx = next((i for i, label in id2label.items() if label == "toxic"), 1)
    spam_idx = next((i for i, label in id2label.items() if label == "spam"), 2)

    session = ort.InferenceSession(str(root / "onnx" / "model.onnx"), providers=["CPUExecutionProvider"])
    input_names = {inp.name for inp in session.get_inputs()}

    toxic: list[float] = []
    spam: list[float] = []
    started = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[start : start + batch_size],
            return_tensors="np",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        feed = {k: v for k, v in enc.items() if k in input_names}
        logits = session.run(None, feed)[0]
        probs = _softmax(logits)
        toxic.extend(probs[:, toxic_idx].tolist())
        spam.extend(probs[:, spam_idx].tolist())
    elapsed = time.perf_counter() - started
    return {
        "toxic": toxic,
        "spam": spam,
        "latency_ms_per_row": (elapsed * 1000) / max(1, len(texts)),
        "rows_per_sec": len(texts) / elapsed if elapsed else 0.0,
    }


def prf(y_true: Iterable[int], scores: Iterable[float], threshold: float) -> dict[str, float | int]:
    truth = list(y_true)
    pred = [1 if s >= threshold else 0 for s in scores]
    tp = sum(1 for t, p in zip(truth, pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(truth, pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(truth, pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(truth, pred) if t == 0 and p == 0)
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
        "threshold": threshold,
    }


def best_threshold(y_true: list[int], scores: list[float]) -> dict[str, float | int]:
    best = prf(y_true, scores, 0.5)
    for threshold in [i / 100 for i in range(1, 100)]:
        metric = prf(y_true, scores, threshold)
        if (metric["f1"], metric["precision"], metric["recall"]) > (
            best["f1"],
            best["precision"],
            best["recall"],
        ):
            best = metric
    return best


def print_metric(name: str, metrics: dict[str, float | int]) -> None:
    print(
        f"{name:<34} "
        f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
        f"TP/FP/FN/TN={metrics['tp']}/{metrics['fp']}/{metrics['fn']}/{metrics['tn']} "
        f"thr={metrics['threshold']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-model", default=str(DEFAULT_CURRENT))
    parser.add_argument("--candidate-model", default=DEFAULT_CANDIDATE)
    parser.add_argument("--rows-per-label", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--current-threshold", type=float, default=0.93)
    parser.add_argument("--candidate-threshold", type=float, default=0.90)
    parser.add_argument("--include-sexual", action="store_true")
    parser.add_argument("--output", default="reports/moderation_candidate_compare.json")
    args = parser.parse_args()

    rows = assemble(args.rows_per_label, args.seed, args.include_sexual)
    texts = [row["text"] for row in rows]
    labels = [row["true_label"] for row in rows]

    print(f"[model] current   <- {args.current_model}")
    current = run_current(args.current_model, texts, args.device, args.batch_size, args.max_length)
    print(f"[model] candidate <- {args.candidate_model}")
    candidate = run_candidate(args.candidate_model, texts, args.batch_size, args.max_length)

    y_harm = [1 if label == "harmful_content" else 0 for label in labels if label in ("harmful_content", "safe")]
    idx_harm = [i for i, label in enumerate(labels) if label in ("harmful_content", "safe")]
    current_harm = [current["harmful"][i] for i in idx_harm]
    candidate_harm = [candidate["toxic"][i] for i in idx_harm]

    results = {
        "rows_per_label": args.rows_per_label,
        "total_rows": len(rows),
        "include_sexual": args.include_sexual,
        "thresholds": {
            "current": args.current_threshold,
            "candidate": args.candidate_threshold,
        },
        "latency": {
            "current": {
                "ms_per_row": round(float(current["latency_ms_per_row"]), 3),
                "rows_per_sec": round(float(current["rows_per_sec"]), 3),
            },
            "candidate": {
                "ms_per_row": round(float(candidate["latency_ms_per_row"]), 3),
                "rows_per_sec": round(float(candidate["rows_per_sec"]), 3),
            },
        },
        "fixed_threshold": {
            "harmful_vs_safe": {
                "current_harmful_head": prf(y_harm, current_harm, args.current_threshold),
                "candidate_toxic_class": prf(y_harm, candidate_harm, args.candidate_threshold),
            }
        },
        "best_threshold_on_this_eval": {
            "harmful_vs_safe": {
                "current_harmful_head": best_threshold(y_harm, current_harm),
                "candidate_toxic_class": best_threshold(y_harm, candidate_harm),
            }
        },
    }

    if args.include_sexual:
        y_sex = [1 if label == "sexual" else 0 for label in labels if label in ("sexual", "safe")]
        idx_sex = [i for i, label in enumerate(labels) if label in ("sexual", "safe")]
        y_any = [1 if label in ("harmful_content", "sexual") else 0 for label in labels]
        current_sex = [current["sexual"][i] for i in idx_sex]
        candidate_sex = [candidate["toxic"][i] for i in idx_sex]
        current_any = [max(h, s) for h, s in zip(current["harmful"], current["sexual"])]
        candidate_any = [max(t, s) for t, s in zip(candidate["toxic"], candidate["spam"])]
        results["fixed_threshold"]["sexual_vs_safe"] = {
            "current_sexual_head": prf(y_sex, current_sex, args.current_threshold),
            "candidate_toxic_class_unsupported": prf(y_sex, candidate_sex, args.candidate_threshold),
        }
        results["fixed_threshold"]["unsafe_any_vs_safe"] = {
            "current_max_harmful_sexual": prf(y_any, current_any, args.current_threshold),
            "candidate_max_toxic_spam": prf(y_any, candidate_any, args.candidate_threshold),
        }
        results["best_threshold_on_this_eval"]["sexual_vs_safe"] = {
            "current_sexual_head": best_threshold(y_sex, current_sex),
            "candidate_toxic_class_unsupported": best_threshold(y_sex, candidate_sex),
        }
        results["best_threshold_on_this_eval"]["unsafe_any_vs_safe"] = {
            "current_max_harmful_sexual": best_threshold(y_any, current_any),
            "candidate_max_toxic_spam": best_threshold(y_any, candidate_any),
        }

    print("\nFIXED THRESHOLDS")
    print_metric("current harmful vs safe", results["fixed_threshold"]["harmful_vs_safe"]["current_harmful_head"])
    print_metric("MiniLM toxic vs safe", results["fixed_threshold"]["harmful_vs_safe"]["candidate_toxic_class"])
    if args.include_sexual:
        print_metric("current sexual vs safe", results["fixed_threshold"]["sexual_vs_safe"]["current_sexual_head"])
        print_metric("MiniLM toxic on sexual vs safe", results["fixed_threshold"]["sexual_vs_safe"]["candidate_toxic_class_unsupported"])
        print_metric("current unsafe_any vs safe", results["fixed_threshold"]["unsafe_any_vs_safe"]["current_max_harmful_sexual"])
        print_metric("MiniLM toxic/spam unsafe_any", results["fixed_threshold"]["unsafe_any_vs_safe"]["candidate_max_toxic_spam"])

    print("\nBEST THRESHOLD ON THIS EVAL ONLY")
    print_metric("current harmful vs safe", results["best_threshold_on_this_eval"]["harmful_vs_safe"]["current_harmful_head"])
    print_metric("MiniLM toxic vs safe", results["best_threshold_on_this_eval"]["harmful_vs_safe"]["candidate_toxic_class"])
    if args.include_sexual:
        print_metric("current sexual vs safe", results["best_threshold_on_this_eval"]["sexual_vs_safe"]["current_sexual_head"])
        print_metric("MiniLM toxic on sexual vs safe", results["best_threshold_on_this_eval"]["sexual_vs_safe"]["candidate_toxic_class_unsupported"])
        print_metric("current unsafe_any vs safe", results["best_threshold_on_this_eval"]["unsafe_any_vs_safe"]["current_max_harmful_sexual"])
        print_metric("MiniLM toxic/spam unsafe_any", results["best_threshold_on_this_eval"]["unsafe_any_vs_safe"]["candidate_max_toxic_spam"])

    print("\nLATENCY")
    print(f"current:   {results['latency']['current']['ms_per_row']} ms/row, {results['latency']['current']['rows_per_sec']} rows/sec")
    print(f"candidate: {results['latency']['candidate']['ms_per_row']} ms/row, {results['latency']['candidate']['rows_per_sec']} rows/sec")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
