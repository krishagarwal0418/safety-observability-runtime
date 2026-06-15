"""
Full end-to-end pipeline evaluation with both BERT models.

Assembles a mixed dataset covering all four pipeline outputs:
  prompt_injection  — from xTRam1/safe-guard-prompt-injection (clean, not in training)
  harmful_content   — from ucberkeley-dlab/measuring-hate-speech (clean, not in training)
  sexual            — from google/civil_comments sexual_explicit field (clean, not in training)
  safe              — benign rows from the same sources

Runs both BERTs with calibrated thresholds (0.50 injection, 0.93 moderation) and
reports per-label precision/recall/F1 plus a confusion matrix showing how labels
cross-fire and what falls through.

Usage:
  python scripts/eval_full_pipeline.py \
      --injection-model models/transformers/prompt_injection \
      --moderation-model models/transformers/moderation \
      --device cuda \
      --rows-per-label 500
"""
from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ("prompt_injection", "harmful_content", "sexual", "safe")


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def _load_injection(n: int) -> list[dict]:
    from datasets import load_dataset
    print("[data] loading injection positives: xTRam1/safe-guard-prompt-injection ...")
    ds = load_dataset("xTRam1/safe-guard-prompt-injection")["train"]
    pos = [{"text": r["text"], "true_label": "prompt_injection"}
           for r in ds if int(r.get("label", 0)) == 1 and r.get("text", "").strip()]
    random.shuffle(pos)
    return pos[:n]


def _load_harmful(n: int) -> list[dict]:
    from datasets import load_dataset
    print("[data] loading harmful positives: ucberkeley-dlab/measuring-hate-speech ...")
    ds = load_dataset("ucberkeley-dlab/measuring-hate-speech")["train"]
    # Use high-confidence hate speech (score >= 1.0 = clearly positive)
    pos = [{"text": r["text"], "true_label": "harmful_content"}
           for r in ds if float(r.get("hate_speech_score", 0)) >= 1.0 and r.get("text", "").strip()]
    random.shuffle(pos)
    return pos[:n]


def _load_sexual(n: int) -> list[dict]:
    from datasets import load_dataset
    print("[data] loading sexual positives: google/civil_comments sexual_explicit ...")
    ds = load_dataset("google/civil_comments")["train"]
    pos = [{"text": r["comment_text"], "true_label": "sexual"}
           for r in ds if float(r.get("sexual_explicit", 0)) >= 0.5 and r.get("comment_text", "").strip()]
    random.shuffle(pos)
    return pos[:n]


def _load_safe(n: int) -> list[dict]:
    from datasets import load_dataset
    print("[data] loading safe negatives: SQuAD questions ...")
    ds = load_dataset("rajpurkar/squad")["validation"]
    rows = [{"text": r["question"], "true_label": "safe"}
            for r in ds if r.get("question", "").strip()]
    # Also grab benign rows from civil_comments (low toxicity, low sexual)
    print("[data] also loading safe from civil_comments low-toxicity rows ...")
    ds2 = load_dataset("google/civil_comments")["test"]
    civil_safe = [{"text": r["comment_text"], "true_label": "safe"}
                  for r in ds2
                  if float(r.get("toxicity", 1)) < 0.1
                  and float(r.get("sexual_explicit", 1)) < 0.1
                  and r.get("comment_text", "").strip()]
    combined = rows + civil_safe
    random.shuffle(combined)
    return combined[:n]


def assemble(rows_per_label: int, seed: int) -> list[dict]:
    random.seed(seed)
    rows = []
    rows += _load_injection(rows_per_label)
    rows += _load_harmful(rows_per_label)
    rows += _load_sexual(rows_per_label)
    rows += _load_safe(rows_per_label)
    random.shuffle(rows)
    label_counts = {l: sum(1 for r in rows if r["true_label"] == l) for l in LABELS}
    print(f"\n[data] assembled {len(rows)} rows: {label_counts}\n")
    return rows


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _resolve(p: str) -> str:
    return str(Path(p).resolve()) if Path(p).exists() else p


def load_model(path: str, device: str):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval().to(device)
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    return tok, model, id2label


def run_injection(tok, model, id2label, texts, device, bs=64, max_len=128) -> list[float]:
    pos_idx = next((i for i, l in id2label.items() if "inject" in l or l == "label_1"), 1)
    out = []
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s+bs], return_tensors="pt", truncation=True, padding=True, max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            probs = torch.softmax(model(**enc).logits.float(), dim=-1).cpu()
        out.extend(probs[:, pos_idx].tolist())
    return out


def run_moderation(tok, model, id2label, texts, device, bs=64, max_len=128) -> tuple[list[float], list[float]]:
    harmful_idx = next((i for i, l in id2label.items() if "harmful" in l), 0)
    sexual_idx = next((i for i, l in id2label.items() if "sexual" in l), 1)
    harmful_out, sexual_out = [], []
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s+bs], return_tensors="pt", truncation=True, padding=True, max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            probs = torch.sigmoid(model(**enc).logits.float()).cpu()
        harmful_out.extend(probs[:, harmful_idx].tolist())
        sexual_out.extend(probs[:, sexual_idx].tolist())
    return harmful_out, sexual_out


def pipeline_label(inj: float, harmful: float, sexual: float,
                   inj_thr: float, mod_thr: float) -> str:
    """Mimic the production pipeline decision logic."""
    if inj >= inj_thr:
        return "prompt_injection"
    if harmful >= mod_thr:
        return "harmful_content"
    if sexual >= mod_thr:
        return "sexual"
    return "safe"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--injection-model", default="models/transformers/prompt_injection")
    ap.add_argument("--moderation-model", default="models/transformers/moderation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rows-per-label", type=int, default=500)
    ap.add_argument("--inj-threshold", type=float, default=0.50)
    ap.add_argument("--mod-threshold", type=float, default=0.93)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--output", default="reports/eval_full_pipeline.json")
    args = ap.parse_args()

    os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

    rows = assemble(args.rows_per_label, args.seed)
    texts = [r["text"] for r in rows]

    inj_path = _resolve(args.injection_model)
    mod_path = _resolve(args.moderation_model)

    print(f"[model] injection  <- {inj_path}")
    itok, imodel, iid = load_model(inj_path, args.device)
    print(f"[model] moderation <- {mod_path}\n")
    mtok, mmodel, mid = load_model(mod_path, args.device)

    print("[inference] running injection model ...")
    inj_scores = run_injection(itok, imodel, iid, texts, args.device, args.batch_size)
    del imodel
    if args.device == "cuda":
        torch.cuda.empty_cache()

    print("[inference] running moderation model ...")
    harmful_scores, sexual_scores = run_moderation(mtok, mmodel, mid, texts, args.device, args.batch_size)
    del mmodel
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # Pipeline decisions
    pred_labels = [
        pipeline_label(inj_scores[i], harmful_scores[i], sexual_scores[i],
                       args.inj_threshold, args.mod_threshold)
        for i in range(len(rows))
    ]

    # Confusion matrix: true_label -> pred_label -> count
    confusion: dict[str, dict[str, int]] = {l: defaultdict(int) for l in LABELS}
    for row, pred in zip(rows, pred_labels):
        confusion[row["true_label"]][pred] += 1

    # Per-label P/R/F1
    per_label = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[tl][label] for tl in LABELS if tl != label)
        fn = sum(confusion[label][pl] for pl in LABELS if pl != label)
        p, r, f1 = prf(tp, fp, fn)
        per_label[label] = {"precision": round(p, 3), "recall": round(r, 3),
                            "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}

    macro_f1 = sum(v["f1"] for v in per_label.values()) / len(LABELS)

    # Print results
    print("\n" + "=" * 82)
    print(f"  FULL PIPELINE EVAL  ({len(rows)} rows, {args.rows_per_label}/label)")
    print(f"  inj_thr={args.inj_threshold}  mod_thr={args.mod_threshold}")
    print("=" * 82)
    print(f"\n{'Label':<20} {'P':>7} {'R':>7} {'F1':>7}   TP / FP / FN")
    print("-" * 60)
    for label in LABELS:
        m = per_label[label]
        print(f"{label:<20} {m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f}   "
              f"{m['tp']} / {m['fp']} / {m['fn']}")
    print(f"\n{'macro F1':<20} {' ':>7} {' ':>7} {macro_f1:>7.3f}")
    print("=" * 82)

    print(f"\n{'CONFUSION MATRIX':}")
    print(f"  rows=true label, cols=predicted label\n")
    print(f"  {'true \\ pred':<22}", end="")
    for pl in LABELS:
        print(f"{pl[:10]:>12}", end="")
    print()
    print("  " + "-" * (22 + 12 * len(LABELS)))
    for tl in LABELS:
        print(f"  {tl:<22}", end="")
        for pl in LABELS:
            c = confusion[tl][pl]
            marker = " <--" if tl == pl else ""
            print(f"{c:>12}{marker[:0]}", end="")  # no marker in matrix itself
        print()

    # Coverage holes — both models missed
    holes = [(rows[i]["text"][:70], rows[i]["true_label"], inj_scores[i], harmful_scores[i], sexual_scores[i])
             for i in range(len(rows))
             if rows[i]["true_label"] != "safe" and pred_labels[i] == "safe"]
    if holes:
        print(f"\n[coverage holes] {len(holes)} true positives predicted SAFE (missed entirely):")
        for txt, tl, inj, harm, sex in holes[:10]:
            print(f"  true={tl:<17} inj={inj:.3f} harm={harm:.3f} sex={sex:.3f} | {txt}")

    # Write output
    import json
    result = {
        "rows_per_label": args.rows_per_label,
        "total_rows": len(rows),
        "thresholds": {"injection": args.inj_threshold, "moderation": args.mod_threshold},
        "macro_f1": round(macro_f1, 3),
        "per_label": per_label,
        "confusion": {tl: dict(confusion[tl]) for tl in LABELS},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
