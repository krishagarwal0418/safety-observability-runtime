"""Per-subcategory moderation eval.

Uses the openai_moderation field in lmsys/toxic-chat (which scores 11 fine-grained
categories) to show which subcategories our model is catching vs. missing.

Our model outputs two heads: harmful_content and sexual.
We map each openai_moderation subcategory to the appropriate head and compute
precision / recall / F1 per subcategory to understand where the model fails.

Usage:
    python scripts/eval_moderation_breakdown.py \
        --model models/transformers/moderation \
        --device cuda
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# openai_moderation subcategory -> which of our heads it maps to
# (toxic-chat stores these as list-of-[name, score] pairs)
SUBCAT_TO_HEAD = {
    "harassment":           "harmful_content",
    "harassment/threatening": "harmful_content",
    "hate":                 "harmful_content",
    "hate/threatening":     "harmful_content",
    "self-harm":            "harmful_content",
    "self-harm/intent":     "harmful_content",
    "self-harm/instructions": "harmful_content",
    "violence":             "harmful_content",
    "violence/graphic":     "harmful_content",
    "sexual":               "sexual",
    "sexual/minors":        "sexual",
}

THRESHOLD = 0.15  # openai_moderation score threshold to call a row "positive" for a subcategory


def load_model(path: str, device: str):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval().to(device)
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    harmful_idx = next((i for i, l in id2label.items() if "harmful" in l), 0)
    sexual_idx = next((i for i, l in id2label.items() if "sexual" in l), 1)
    return tok, model, harmful_idx, sexual_idx


def score_batch(tok, model, texts: list[str], device: str, harmful_idx: int, sexual_idx: int, batch_size: int = 64):
    all_harmful, all_sexual = [], []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=128)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = model(**enc).logits.float()
            probs = torch.sigmoid(logits).cpu()
        all_harmful.extend(probs[:, harmful_idx].tolist())
        all_sexual.extend(probs[:, sexual_idx].tolist())
    return all_harmful, all_sexual


def parse_openai_mod(raw) -> dict[str, float]:
    """Parse the openai_moderation field (stored as JSON string or list)."""
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            try:
                parsed = ast.literal_eval(raw)
            except Exception:
                return {}
    else:
        parsed = raw
    if isinstance(parsed, list):
        return {item[0]: float(item[1]) for item in parsed if isinstance(item, (list, tuple)) and len(item) == 2}
    if isinstance(parsed, dict):
        return {k: float(v) for k, v in parsed.items()}
    return {}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/transformers/moderation")
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--harmful-threshold", type=float, default=0.5,
                    help="our model's harmful_content score threshold")
    ap.add_argument("--sexual-threshold", type=float, default=0.5,
                    help="our model's sexual score threshold")
    ap.add_argument("--oai-threshold", type=float, default=THRESHOLD,
                    help="openai_moderation score to treat as gold positive for a subcategory")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

    from datasets import load_dataset
    print("[data] loading lmsys/toxic-chat toxicchat0124 test split...")
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
    rows = list(ds)
    if args.limit:
        rows = rows[:args.limit]
    print(f"[data] {len(rows)} rows")

    model_path = str(Path(args.model).resolve()) if Path(args.model).exists() else args.model
    print(f"[model] loading {model_path} on {args.device}")
    tok, model, harmful_idx, sexual_idx = load_model(model_path, args.device)

    texts = [r.get("user_input", "") for r in rows]
    print(f"[inference] scoring {len(texts)} rows...")
    harmful_scores, sexual_scores = score_batch(
        tok, model, texts, args.device, harmful_idx, sexual_idx, args.batch_size
    )

    # Per-subcategory stats
    subcat_stats: dict[str, dict] = {k: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n_pos": 0} for k in SUBCAT_TO_HEAD}

    for i, row in enumerate(rows):
        oai = parse_openai_mod(row.get("openai_moderation", "[]"))
        pred_harmful = harmful_scores[i] >= args.harmful_threshold
        pred_sexual = sexual_scores[i] >= args.sexual_threshold

        for subcat, head in SUBCAT_TO_HEAD.items():
            gold = oai.get(subcat, 0.0) >= args.oai_threshold
            pred = pred_harmful if head == "harmful_content" else pred_sexual
            st = subcat_stats[subcat]
            if gold:
                st["n_pos"] += 1
            if gold and pred:
                st["tp"] += 1
            elif not gold and pred:
                st["fp"] += 1
            elif gold and not pred:
                st["fn"] += 1
            else:
                st["tn"] += 1

    # Overall using human annotation (toxicity field)
    overall = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for i, row in enumerate(rows):
        gold = int(row.get("toxicity", 0)) == 1
        pred = harmful_scores[i] >= args.harmful_threshold or sexual_scores[i] >= args.sexual_threshold
        if gold and pred:     overall["tp"] += 1
        elif not gold and pred: overall["fp"] += 1
        elif gold and not pred: overall["fn"] += 1
        else:                   overall["tn"] += 1

    op, orec, of1 = prf(overall["tp"], overall["fp"], overall["fn"])

    print()
    print("=" * 80)
    print(f"  MODERATION BREAKDOWN  (threshold harm={args.harmful_threshold} sex={args.sexual_threshold})")
    print("=" * 80)
    print(f"\n{'OVERALL (human annotation)':<35} P={op:.3f}  R={orec:.3f}  F1={of1:.3f}  "
          f"(TP={overall['tp']} FP={overall['fp']} FN={overall['fn']})\n")
    print(f"{'Subcategory (OAI thresh=' + str(args.oai_threshold) + ')':<35} {'head':<17} {'n_pos':>6} {'P':>6} {'R':>6} {'F1':>6}  TP / FP / FN")
    print("-" * 90)

    results = []
    for subcat, st in sorted(subcat_stats.items(), key=lambda x: -x[1]["n_pos"]):
        p, r, f1 = prf(st["tp"], st["fp"], st["fn"])
        head = SUBCAT_TO_HEAD[subcat]
        print(f"{subcat:<35} {head:<17} {st['n_pos']:>6} {p:>6.3f} {r:>6.3f} {f1:>6.3f}  "
              f"{st['tp']} / {st['fp']} / {st['fn']}")
        results.append({"subcat": subcat, "head": head, "n_pos": st["n_pos"],
                        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
                        **st})
    print("=" * 90)

    print("\nNote: low n_pos subcategories are unreliable — not enough positives in the test set.")
    print("Note: OAI threshold controls what counts as a gold positive for each subcategory.")
    print("      Raise it (e.g. 0.5) to see only high-confidence OAI positives.\n")

    if args.output:
        out = {
            "model": args.model,
            "n_rows": len(rows),
            "thresholds": {"harmful": args.harmful_threshold, "sexual": args.sexual_threshold,
                           "oai": args.oai_threshold},
            "overall": {"precision": round(op, 4), "recall": round(orec, 4), "f1": round(of1, 4), **overall},
            "breakdown": results,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
