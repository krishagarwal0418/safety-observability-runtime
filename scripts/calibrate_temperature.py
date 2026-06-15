"""
Temperature-scale a safety classifier so the 0.5 threshold becomes a sane
operating point, and recommend a deployment threshold.

Why: our models have strong ranking ability (good bestF1) but the raw sigmoid/
softmax scores are mis-calibrated — the F1-optimal threshold drifts to 0.96 on
one distribution and 0.01 on another. Temperature scaling fits ONE scalar T and
maps every score through p = sigmoid(z / T), centering the distribution. It is a
monotonic transform: it does NOT change bestF1 (the ranking is untouched); it
moves WHERE the threshold sits so that 0.5 works in production.

We fit T on a held-out calibration split and report on a disjoint test split, so
the reported "after" numbers are honest (no fitting on the eval rows).

Both model families reduce to a single per-example scalar z:
  * injection (softmax, 2 classes): z = logit[INJECTION] - logit[SAFE]
  * moderation (sigmoid, 2 heads):  z = max(logit[harmful], logit[sexual])
  positive prob = sigmoid(z / T)  in both cases.

Usage:
  python scripts/calibrate_temperature.py --task injection \
     --model models/transformers/prompt_injection \
     --datasets JasperLS/prompt-injections,JailbreakBench/JBB-Behaviors:behaviors \
     --device cuda

  python scripts/calibrate_temperature.py --task moderation \
     --model models/transformers/moderation \
     --datasets ucberkeley-dlab/measuring-hate-speech,google/civil_comments \
     --device cuda
"""
from __future__ import annotations

import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Reuse the held-out dataset loaders / pos-sets from the bake-off.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_bakeoff import (  # noqa: E402
    HELD_OUT_LOADERS,
    INJECTION_POS,
    MODERATION_POS,
)


def _resolve(p: str) -> str:
    return str(Path(p).resolve()) if Path(p).exists() else p


def load_one_dataset(spec: str, task: str) -> list[dict]:
    """spec is 'hf_id' or 'hf_id:config'. Returns [{text, gold}]."""
    from datasets import load_dataset

    hf_id, _, cfg = spec.partition(":")
    if hf_id in HELD_OUT_LOADERS:
        ds = load_dataset(hf_id, cfg) if cfg else load_dataset(hf_id)
        rows = HELD_OUT_LOADERS[hf_id](ds)
    else:
        ds = load_dataset(hf_id, cfg) if cfg else load_dataset(hf_id)
        split = "test" if "test" in ds else list(ds.keys())[0]
        feats = list(ds[split].features)
        tf = next((k for k in feats if k.lower() in ("text", "prompt", "user_input", "input", "sentence", "comment_text", "question")), None)
        lf = next((k for k in feats if k.lower() in ("label", "labels", "toxicity", "jailbreaking", "is_injection", "class")), None)
        pos_set = INJECTION_POS if task == "injection" else MODERATION_POS
        rows = []
        for r in ds[split]:
            raw = r[lf]
            if isinstance(raw, (int, float, bool)):
                gold = 1 if int(raw) == 1 else 0
            else:
                gold = 1 if str(raw).lower().strip() in pos_set else 0
            txt = (r[tf] or "").strip()
            if txt:
                rows.append({"text": txt, "gold": gold})
    print(f"[data] {spec}: {len(rows)} rows | positives={sum(r['gold'] for r in rows)}")
    return rows


def extract_logits(task: str, model_path: str, texts: list[str], device: str,
                   batch_size: int, max_length: int) -> list[float]:
    """Return one scalar z per text (pre-temperature). p_pos = sigmoid(z)."""
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).eval().to(device)
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}

    if task == "injection":
        pos_idx = next((i for i, l in id2label.items() if "inject" in l or l == "label_1"), 1)
        neg_idx = next((i for i in id2label if i != pos_idx), 0)
    else:
        harmful_idx = next((i for i, l in id2label.items() if "harmful" in l), 0)
        sexual_idx = next((i for i, l in id2label.items() if "sexual" in l), 1)

    zs: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = model(**enc).logits.float().cpu()
        for row in logits:
            if task == "injection":
                zs.append(float(row[pos_idx] - row[neg_idx]))     # softmax margin
            else:
                zs.append(float(max(row[harmful_idx], row[sexual_idx])))  # max sigmoid logit
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return zs


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def bce(zs: list[float], ys: list[int], T: float) -> float:
    eps = 1e-7
    total = 0.0
    for z, y in zip(zs, ys):
        p = min(max(sigmoid(z / T), eps), 1 - eps)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(zs)


def fit_temperature(zs: list[float], ys: list[int]) -> float:
    """Grid + local refine on T minimizing BCE. Robust, no autograd needed."""
    grid = [round(0.05 * i, 2) for i in range(1, 200)]  # 0.05 .. 9.95
    best_T, best_loss = 1.0, float("inf")
    for T in grid:
        l = bce(zs, ys, T)
        if l < best_loss:
            best_T, best_loss = T, l
    # local refine around best_T
    lo, hi = max(0.01, best_T - 0.05), best_T + 0.05
    for i in range(50):
        T = lo + (hi - lo) * i / 49
        l = bce(zs, ys, T)
        if l < best_loss:
            best_T, best_loss = T, l
    return best_T


def prf_at(zs: list[float], ys: list[int], T: float, thr: float) -> dict:
    tp = fp = fn = tn = 0
    for z, y in zip(zs, ys):
        pred = 1 if sigmoid(z / T) >= thr else 0
        if y == 1 and pred == 1: tp += 1
        elif y == 0 and pred == 1: fp += 1
        elif y == 1 and pred == 0: fn += 1
        else: tn += 1
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def best_threshold(zs: list[float], ys: list[int], T: float) -> tuple[float, dict]:
    best_t, best = 0.5, prf_at(zs, ys, T, 0.5)
    for i in range(1, 100):
        t = i / 100
        m = prf_at(zs, ys, T, t)
        if m["f1"] > best["f1"]:
            best_t, best = t, m
    return best_t, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["injection", "moderation"])
    ap.add_argument("--model", required=True, help="local path to OUR model for this task")
    ap.add_argument("--datasets", required=True, help="comma-sep hf ids, each optionally :config")
    ap.add_argument("--limit-per-dataset", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", default=None, help="where to write calibration.json (default: <model>/calibration.json)")
    args = ap.parse_args()

    random.seed(args.seed)

    # Gather + balance held-out rows across all datasets
    all_rows: list[dict] = []
    for spec in args.datasets.split(","):
        rows = load_one_dataset(spec.strip(), args.task)
        random.shuffle(rows)
        pos = [r for r in rows if r["gold"] == 1][: args.limit_per_dataset // 2]
        neg = [r for r in rows if r["gold"] == 0][: args.limit_per_dataset // 2]
        all_rows.extend(pos + neg)
    random.shuffle(all_rows)
    texts = [r["text"] for r in all_rows]
    ys = [r["gold"] for r in all_rows]
    print(f"\n[data] total calibration pool: {len(all_rows)} rows | positives={sum(ys)}\n")

    model_path = _resolve(args.model)
    print(f"[model] extracting logits from {model_path} on {args.device} ...")
    zs = extract_logits(args.task, model_path, texts, args.device, args.batch_size, args.max_length)

    # 50/50 stratified split: fit T on one half, report on the other
    idx_pos = [i for i, y in enumerate(ys) if y == 1]
    idx_neg = [i for i, y in enumerate(ys) if y == 0]
    random.shuffle(idx_pos)
    random.shuffle(idx_neg)
    half_p, half_n = len(idx_pos) // 2, len(idx_neg) // 2
    fit_idx = set(idx_pos[:half_p] + idx_neg[:half_n])
    fit_z = [zs[i] for i in range(len(zs)) if i in fit_idx]
    fit_y = [ys[i] for i in range(len(ys)) if i in fit_idx]
    test_z = [zs[i] for i in range(len(zs)) if i not in fit_idx]
    test_y = [ys[i] for i in range(len(ys)) if i not in fit_idx]

    # Fit temperature on the fit split
    T = fit_temperature(fit_z, fit_y)
    # Recommend a deployment threshold = F1-optimal on the fit split (post-T)
    rec_thr, _ = best_threshold(fit_z, fit_y, T)

    # Report on the disjoint test split
    before_half = prf_at(test_z, test_y, 1.0, 0.5)          # raw, thr 0.5
    after_half = prf_at(test_z, test_y, T, 0.5)             # calibrated, thr 0.5
    after_rec = prf_at(test_z, test_y, T, rec_thr)          # calibrated, recommended thr
    raw_bt, raw_best = best_threshold(test_z, test_y, 1.0)  # raw ceiling (unchanged by T)

    print("\n" + "=" * 78)
    print(f"  TEMPERATURE CALIBRATION  ({args.task})   fitted T = {T:.3f}")
    print("=" * 78)
    print(f"  test split: {len(test_y)} rows | positives={sum(test_y)}\n")
    print(f"{'config':<34} {'F1':>7} {'P':>7} {'R':>7}")
    print("-" * 60)
    print(f"{'RAW @0.5 (before)':<34} {before_half['f1']:>7.3f} {before_half['precision']:>7.3f} {before_half['recall']:>7.3f}")
    print(f"{'CALIBRATED @0.5 (after)':<34} {after_half['f1']:>7.3f} {after_half['precision']:>7.3f} {after_half['recall']:>7.3f}")
    print(f"{'CALIBRATED @' + format(rec_thr, '.2f') + ' (recommended)':<34} {after_rec['f1']:>7.3f} {after_rec['precision']:>7.3f} {after_rec['recall']:>7.3f}")
    print(f"{'raw bestF1 ceiling @' + format(raw_bt, '.2f'):<34} {raw_best['f1']:>7.3f} {raw_best['precision']:>7.3f} {raw_best['recall']:>7.3f}")
    print("=" * 60)
    print("\nNote: temperature does NOT raise the ceiling; it makes 0.5 usable.")
    print(f"Deploy: apply p = sigmoid(z / {T:.3f}); use threshold {rec_thr:.2f} (or 0.5 after T).\n")

    out_path = Path(args.output) if args.output else Path(model_path) / "calibration.json"
    payload = {
        "task": args.task,
        "temperature": round(T, 4),
        "recommended_threshold": round(rec_thr, 3),
        "score_formula": "p = sigmoid(z / T); injection z = logit[INJ]-logit[SAFE]; moderation z = max(logit[harmful], logit[sexual])",
        "datasets": args.datasets,
        "test_eval": {
            "raw_at_0.5": before_half,
            "calibrated_at_0.5": after_half,
            "calibrated_at_recommended": after_rec,
            "raw_best_threshold": raw_bt,
            "raw_best": raw_best,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
