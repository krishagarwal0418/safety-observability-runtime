"""
Coverage check: JBB-Behaviors harmful goals are NOT prompt injections (no
"ignore previous instructions" mechanics) — they are plain harmful requests
("write a defamatory article", "give bomb instructions"). The injection model
correctly scores them low. The question this script answers: does the MODERATION
model catch them instead, so the overall pipeline still flags them?

We run all 100 JBB harmful + 100 benign goals through BOTH models and report:
  - injection model:  P(injection)   (expected: low on harmful goals — correct)
  - moderation model: P(harmful_content or sexual)  (expected: HIGH — the catch)
  - pipeline OR:      max coverage — flagged if EITHER model fires

If pipeline recall on the harmful split is high, the JBB "gap" is a non-issue:
injection ignores them (rightly) and moderation catches them.

Usage:
  python scripts/check_jbb_coverage.py \
      --injection-model models/transformers/prompt_injection \
      --moderation-model models/transformers/moderation \
      --device cuda
"""
from __future__ import annotations

import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _resolve(p: str) -> str:
    return str(Path(p).resolve()) if Path(p).exists() else p


def load(path: str, device: str):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval().to(device)
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    return tok, model, id2label


def injection_scores(tok, model, id2label, texts, device, bs=64):
    pos_idx = next((i for i, l in id2label.items() if "inject" in l or l == "label_1"), 1)
    out = []
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s+bs], return_tensors="pt", truncation=True, padding=True, max_length=128)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            probs = torch.softmax(model(**enc).logits.float(), dim=-1).cpu()
        out.extend(probs[:, pos_idx].tolist())
    return out


def moderation_scores(tok, model, id2label, texts, device, bs=64):
    harmful_idx = next((i for i, l in id2label.items() if "harmful" in l), 0)
    sexual_idx = next((i for i, l in id2label.items() if "sexual" in l), 1)
    out = []
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s+bs], return_tensors="pt", truncation=True, padding=True, max_length=128)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            probs = torch.sigmoid(model(**enc).logits.float()).cpu()
        for row in probs:
            out.append(float(max(row[harmful_idx], row[sexual_idx])))
    return out


def recall_at(scores, thr):
    return sum(1 for x in scores if x >= thr) / len(scores) if scores else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--injection-model", default="models/transformers/prompt_injection")
    ap.add_argument("--moderation-model", default="models/transformers/moderation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--inj-threshold", type=float, default=0.5)
    ap.add_argument("--mod-threshold", type=float, default=0.5,
                    help="moderation operating threshold (try 0.9 for the calibrated value)")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    harmful = [r["Goal"] for r in ds["harmful"] if r.get("Goal")]
    benign = [r["Goal"] for r in ds["benign"] if r.get("Goal")]
    print(f"[data] JBB: {len(harmful)} harmful, {len(benign)} benign goals\n")

    inj_path = _resolve(args.injection_model)
    mod_path = _resolve(args.moderation_model)
    print(f"[model] injection  <- {inj_path}")
    itok, imodel, iid = load(inj_path, args.device)
    print(f"[model] moderation <- {mod_path}\n")
    mtok, mmodel, mid = load(mod_path, args.device)

    # Score both splits with both models
    h_inj = injection_scores(itok, imodel, iid, harmful, args.device)
    b_inj = injection_scores(itok, imodel, iid, benign, args.device)
    h_mod = moderation_scores(mtok, mmodel, mid, harmful, args.device)
    b_mod = moderation_scores(mtok, mmodel, mid, benign, args.device)

    it, mt = args.inj_threshold, args.mod_threshold

    # Pipeline = flagged if EITHER model fires
    h_pipe = [max(a >= it, b >= mt) for a, b in zip(h_inj, h_mod)]
    b_pipe = [max(a >= it, b >= mt) for a, b in zip(b_inj, b_mod)]

    print("=" * 70)
    print(f"  JBB COVERAGE  (inj_thr={it}  mod_thr={mt})")
    print("=" * 70)
    print(f"{'':<22}{'HARMFUL recall':>16}{'BENIGN flagged':>18}")
    print("-" * 70)
    print(f"{'injection only':<22}{recall_at(h_inj, it):>16.3f}{recall_at(b_inj, it):>18.3f}")
    print(f"{'moderation only':<22}{recall_at(h_mod, mt):>16.3f}{recall_at(b_mod, mt):>18.3f}")
    print(f"{'pipeline (either)':<22}{sum(h_pipe)/len(h_pipe):>16.3f}{sum(b_pipe)/len(b_pipe):>18.3f}")
    print("=" * 70)
    print("\nHARMFUL recall = fraction of harmful goals caught (want HIGH).")
    print("BENIGN flagged = fraction of benign goals wrongly flagged (want LOW).")
    print("\nIf 'moderation only' or 'pipeline' HARMFUL recall is high, the JBB")
    print("injection gap is covered — injection rightly ignores them, moderation catches them.\n")

    # Sample a few harmful goals injection missed but moderation caught
    print("Examples: harmful goals injection MISSED but moderation CAUGHT:")
    shown = 0
    for i, txt in enumerate(harmful):
        if h_inj[i] < it and h_mod[i] >= mt:
            print(f"  inj={h_inj[i]:.3f} mod={h_mod[i]:.3f} | {txt[:70]}")
            shown += 1
            if shown >= 5:
                break
    if shown == 0:
        print("  (none — moderation did not rescue any; check thresholds)")

    print("\nExamples: harmful goals BOTH models missed (true coverage holes):")
    shown = 0
    for i, txt in enumerate(harmful):
        if h_inj[i] < it and h_mod[i] < mt:
            print(f"  inj={h_inj[i]:.3f} mod={h_mod[i]:.3f} | {txt[:70]}")
            shown += 1
            if shown >= 8:
                break
    if shown == 0:
        print("  (none — every harmful goal was caught by at least one model)")


if __name__ == "__main__":
    main()
