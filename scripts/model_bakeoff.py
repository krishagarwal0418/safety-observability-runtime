"""
Head-to-head bake-off: compare safety classifiers on a shared benchmark with a
common binary taxonomy, apples-to-apples.

Two independent tasks (run separately):
  --task injection   positive class = prompt injection
  --task moderation  positive class = harmful/toxic content

Every model's raw output is mapped, via a per-model adapter, to a single number:
P(positive). All models are scored on the SAME rows. We report precision/recall/
F1 at the default 0.5 threshold AND at each model's best-F1 threshold, so a model
isn't penalised merely for being calibrated to a different operating point.

Models are tagged "deployable" (similar size to ours) or "ceiling" (much bigger,
useful only as a distillation teacher / upper bound — not a deployment candidate).

Usage:
  python scripts/model_bakeoff.py --task injection \
      --dataset qualifire/prompt-injections-benchmark --limit 5000 --device cuda
  python scripts/model_bakeoff.py --task moderation \
      --dataset lmsys/toxic-chat --dataset-config toxichat0124 --limit 5000 --device cuda

You can also point --data at a local JSONL ({"text":..., "labels":[...]}) instead
of a HF --dataset.
"""
from __future__ import annotations

import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")  # competitors download from hub

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ---------------------------------------------------------------------------
# Model adapters: map a model's raw {label: prob} -> P(positive class)
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    name: str
    hf_id: str                       # local path or hub id
    to_positive: Callable[[dict], float]  # raw {label: prob} -> P(positive)
    sigmoid: bool = False            # multi-label sigmoid vs single-label softmax
    deployable: bool = True          # False = "ceiling/teacher", much bigger than ours
    note: str = ""
    generative: bool = False         # True = CausalLM judge (ShieldGemma/Llama-Guard style)
    guideline: str = ""              # safety policy text for generative judges
    parent: bool = False             # True = the model we fine-tuned FROM (not a fair rival)
    size_m: float = 0.0              # approx params in millions, for the leaderboard


def _get(raw: dict, *names: str, contains: tuple[str, ...] = ()) -> float:
    """Pick a prob by exact label name (case-insensitive) or substring match."""
    low = {k.lower(): v for k, v in raw.items()}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    best = 0.0
    for k, v in low.items():
        if any(c in k for c in contains):
            best = max(best, v)
    return best


# ---- Injection task: positive = injection -------------------------------------

def inj_generic(raw):
    # Works for SAFE/INJECTION, LEGIT/INJECTION, LABEL_0/LABEL_1, and
    # Prompt-Guard's BENIGN/INJECTION/JAILBREAK (jailbreak counts as positive).
    return _get(raw, "injection", "label_1", contains=("inject", "malicious", "attack", "jailbreak"))


# ---- Moderation task: positive = harmful/toxic --------------------------------

def mod_ours(raw):       # our model: {harmful_content, sexual} (sigmoid, multi-label)
    return max(
        _get(raw, "harmful_content", contains=("harmful", "hate", "toxic", "harass", "violence", "self")),
        _get(raw, "sexual", contains=("sexual",)),
    )

def mod_koala(raw):      # KoalaAI/Text-Moderation: {OK, H, HR, S, ...} softmax — positive = 1-P(OK)
    ok = _get(raw, "ok", "label_0", contains=("safe", "neutral", "ok"))
    return 1.0 - ok if ok > 0 else _get(raw, contains=("h", "s", "v", "hr", "sh"))

def mod_toxic_multi(raw):  # toxic-bert / MiniLM-jigsaw: {toxic, insult, ...} sigmoid multi-label
    return _get(raw, "toxic", contains=("toxic", "threat", "insult", "hate", "obscene", "identity"))

def mod_toxic_binary(raw):  # martin-ha/toxic-comment-model: {non-toxic, toxic} softmax
    return _get(raw, "toxic", "label_1", contains=("toxic",))


REGISTRIES: dict[str, list[ModelSpec]] = {
    "injection": [
        # ours — local path resolved at runtime from --our-model
        ModelSpec("ours",        "__OURS__",                                       inj_generic, note="your fine-tune", size_m=184),
        # Same-size independent competitors / off-the-shelf adoption candidates.
        ModelSpec("protectai",   "protectai/deberta-v3-base-prompt-injection-v2",  inj_generic, note="verify this isn't your base!", size_m=184),
        ModelSpec("deepset",     "deepset/deberta-v3-base-injection",              inj_generic, size_m=184),
        ModelSpec("prompt-guard","meta-llama/Prompt-Guard-86M",                    inj_generic, note="gated: needs HF auth", size_m=86),
        ModelSpec("distilbert-pi","fmops/distilbert-prompt-injection",             inj_generic, note="off-the-shelf distilled", size_m=66),
    ],
    "moderation": [
        ModelSpec("ours",        "__OURS__",                          mod_ours,         sigmoid=True,  note="your fine-tune (from Koala)", size_m=86),
        # Independent same-size competitors / off-the-shelf adoption candidates.
        ModelSpec("toxic-bert",  "unitary/toxic-bert",               mod_toxic_multi,  sigmoid=True,  size_m=110),
        ModelSpec("minilm-jigsaw","minuva/MiniLMv2-toxic-jigsaw",    mod_toxic_multi,  sigmoid=True,  note="off-the-shelf distilled MiniLM", size_m=23),
        ModelSpec("distil-toxic","martin-ha/toxic-comment-model",    mod_toxic_binary, sigmoid=False, note="off-the-shelf distilled", size_m=67),
        # Parent — the model we fine-tuned FROM; not a fair rival. Shown with --include-parents.
        ModelSpec("koala(parent)","KoalaAI/Text-Moderation",         mod_koala,        sigmoid=False, parent=True, note="your base model — reference only", size_m=86),
        # Ceiling / distillation-teacher candidate (much bigger, generative judge).
        ModelSpec("shieldgemma-2b", "google/shieldgemma-2b", to_positive=lambda raw: 0.0,
                  deployable=False, generative=True, note="gated ~2B teacher/ceiling; --include-ceiling + small --limit", size_m=2000,
                  guideline=('"No Harmful Content": The prompt shall not contain or seek generation of content '
                             "that is hateful, harassing, violent, dangerous, or sexually explicit.")),
    ],
}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _resolve_id(hf_id: str) -> str:
    """Convert relative local paths to absolute so HF hub validation doesn't reject them."""
    p = Path(hf_id)
    if p.exists():
        return str(p.resolve())
    return hf_id


def run_model(spec: ModelSpec, texts: list[str], device: str, batch_size: int, max_length: int) -> list[float] | None:
    hf_id = _resolve_id(spec.hf_id)
    try:
        tok = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForSequenceClassification.from_pretrained(hf_id).eval().to(device)
    except Exception as e:
        print(f"  [skip] {spec.name} ({spec.hf_id}): {e}")
        return None
    id2label = {int(k): v for k, v in getattr(model.config, "id2label", {}).items()}
    out: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = model(**enc).logits.float()
            probs = torch.sigmoid(logits) if spec.sigmoid else torch.softmax(logits, dim=-1)
            probs = probs.detach().cpu()
        for row in probs:
            raw = {id2label.get(i, f"LABEL_{i}"): float(row[i]) for i in range(len(row))}
            out.append(spec.to_positive(raw))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def run_generative_model(spec: ModelSpec, texts: list[str], device: str, max_length: int) -> list[float] | None:
    """ShieldGemma/Llama-Guard-style judge: prompt the model, read P(violation)
    from the first generated token (Yes/No or safe/unsafe)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id = _resolve_id(spec.hf_id)
    try:
        tok = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).eval().to(device)
    except Exception as e:
        print(f"  [skip] {spec.name} ({spec.hf_id}): {e}")
        return None

    def tok_id(word: str) -> int | None:
        ids = tok.encode(word, add_special_tokens=False)
        return ids[0] if ids else None

    yes_ids = [i for i in (tok_id("Yes"), tok_id(" Yes"), tok_id("unsafe"), tok_id(" unsafe")) if i is not None]
    no_ids = [i for i in (tok_id("No"), tok_id(" No"), tok_id("safe"), tok_id(" safe")) if i is not None]
    if not yes_ids or not no_ids:
        print(f"  [skip] {spec.name}: could not locate Yes/No tokens")
        return None

    out: list[float] = []
    for t in texts:
        try:
            inputs = tok.apply_chat_template(
                [{"role": "user", "content": t}],
                guideline=spec.guideline, return_tensors="pt", return_dict=True,
            ).to(device)
        except Exception:
            prompt = f"{spec.guideline}\n\nUser: {t}\n\nDoes this violate the policy? Answer Yes or No.\n"
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=max_length * 4).to(device)
        with torch.inference_mode():
            last = model(**inputs).logits[0, -1, :].float()
        probs = torch.softmax(last, dim=-1)
        p_yes = float(probs[yes_ids].sum())
        p_no = float(probs[no_ids].sum())
        out.append(p_yes / (p_yes + p_no) if (p_yes + p_no) > 0 else 0.0)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def prf(gold: list[int], scores: list[float], threshold: float) -> dict:
    tp = fp = fn = tn = 0
    for g, s in zip(gold, scores):
        pred = 1 if s >= threshold else 0
        if g == 1 and pred == 1: tp += 1
        elif g == 0 and pred == 1: fp += 1
        elif g == 1 and pred == 0: fn += 1
        else: tn += 1
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    acc = (tp + tn) / len(gold) if gold else 0.0
    return {"precision": p, "recall": r, "f1": f1, "accuracy": acc, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def best_threshold(gold: list[int], scores: list[float]) -> tuple[float, dict]:
    best_t, best = 0.5, prf(gold, scores, 0.5)
    for i in range(1, 100):
        t = i / 100
        m = prf(gold, scores, t)
        if m["f1"] > best["f1"]:
            best_t, best = t, m
    return best_t, best


# ---------------------------------------------------------------------------
# Dataset loading + taxonomy mapping to binary gold
# ---------------------------------------------------------------------------

INJECTION_POS = {"injection", "inject", "prompt_injection", "jailbreak", "1", "true", "attack", "malicious", "unsafe"}
MODERATION_POS = {"harmful_content", "harmful", "toxic", "hate", "sexual", "1", "true", "unsafe", "harassment", "violence"}

# Per-dataset loaders for held-out benchmarks with non-standard schemas
HELD_OUT_DATASETS = {
    # ---- injection (not in training) ----
    "JasperLS/prompt-injections": "injection",
    "JailbreakBench/JBB-Behaviors": "injection",
    # ---- moderation (not in training) ----
    "ucberkeley-dlab/measuring-hate-speech": "moderation",
    "civil_comments": "moderation",
}


def _load_jasper(ds) -> list[dict]:
    # JasperLS/prompt-injections: text (str), label (0/1 int)
    split = "train" if "train" in ds else list(ds.keys())[0]
    rows = []
    for r in ds[split]:
        txt = (r.get("text") or r.get("prompt") or "").strip()
        label = r.get("label", 0)
        if txt:
            rows.append({"text": txt, "gold": 1 if str(label) in ("1", "true", "injection") else 0})
    return rows


def _load_jbb(ds) -> list[dict]:
    # JailbreakBench/JBB-Behaviors: splits = 'harmful' and 'benign', text in 'Goal' column
    rows = []
    for split, gold in [("harmful", 1), ("benign", 0)]:
        if split in ds:
            for r in ds[split]:
                txt = (r.get("Goal") or r.get("Behavior") or "").strip()
                if txt:
                    rows.append({"text": txt, "gold": gold})
    return rows


def _load_measuring_hate(ds) -> list[dict]:
    # ucberkeley-dlab/measuring-hate-speech: text (str), hate_speech_score (float, >0.5 = hate)
    split = "train" if "train" in ds else list(ds.keys())[0]
    rows = []
    for r in ds[split]:
        txt = (r.get("text") or "").strip()
        score = float(r.get("hate_speech_score", 0))
        gold = 1 if score >= 0.5 else 0
        if txt:
            rows.append({"text": txt, "gold": gold})
    return rows


def _load_civil_comments(ds) -> list[dict]:
    # civil_comments: comment_text (str), toxicity (float 0-1, >=0.5 = toxic)
    split = "train" if "train" in ds else list(ds.keys())[0]
    rows = []
    for r in ds[split]:
        txt = (r.get("comment_text") or "").strip()
        gold = 1 if float(r.get("toxicity", 0)) >= 0.5 else 0
        if txt:
            rows.append({"text": txt, "gold": gold})
    return rows


HELD_OUT_LOADERS = {
    "JasperLS/prompt-injections": _load_jasper,
    "JailbreakBench/JBB-Behaviors": _load_jbb,
    "ucberkeley-dlab/measuring-hate-speech": _load_measuring_hate,
    "civil_comments": _load_civil_comments,
}


def load_rows(args) -> list[dict]:
    if args.data:
        rows = [json.loads(l) for l in open(args.data)]
        print(f"[data] local file {args.data}: {len(rows)} rows")
        return rows

    from datasets import load_dataset
    # Use dedicated loader for held-out datasets with non-standard schemas
    if args.dataset in HELD_OUT_LOADERS:
        print(f"[data] loading held-out dataset: {args.dataset}")
        ds = load_dataset(args.dataset, args.dataset_config) if args.dataset_config else load_dataset(args.dataset)
        rows = HELD_OUT_LOADERS[args.dataset](ds)
        print(f"[data] {len(rows)} rows loaded | positives={sum(r['gold'] for r in rows)}")
        return rows

    ds = load_dataset(args.dataset, args.dataset_config) if args.dataset_config else load_dataset(args.dataset)
    split = args.split or ("test" if "test" in ds else list(ds.keys())[0])
    feats = list(ds[split].features)
    print(f"[data] {args.dataset} split={split} | features={feats}")
    print(f"[data] sample row: {ds[split][0]}")

    tf = args.text_field or next((k for k in feats if k.lower() in ("text", "prompt", "user_input", "input", "sentence", "comment_text", "question")), None)
    lf = args.label_field or next((k for k in feats if k.lower() in ("label", "labels", "toxicity", "jailbreaking", "is_injection", "class")), None)
    if not tf or not lf:
        raise SystemExit(f"Could not auto-detect text/label fields; pass --text-field/--label-field. features={feats}")
    print(f"[data] text_field={tf} label_field={lf}")

    pos_set = INJECTION_POS if args.task == "injection" else MODERATION_POS
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
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["injection", "moderation"])
    ap.add_argument("--dataset", default=None, help="HF dataset id")
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--data", default=None, help="local JSONL instead of HF dataset")
    ap.add_argument("--text-field", default=None)
    ap.add_argument("--label-field", default=None)
    ap.add_argument("--our-model", default="models/transformers/prompt_injection",
                    help="path to our model for this task")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--include-ceiling", action="store_true", help="also run much-bigger teacher/ceiling models")
    ap.add_argument("--include-parents", action="store_true", help="also run the model we fine-tuned FROM (reference only)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    rows = load_rows(args)
    # Balanced subsample to --limit (keep class balance)
    import random
    random.seed(args.seed)
    random.shuffle(rows)
    pos = [r for r in rows if r["gold"] == 1]
    neg = [r for r in rows if r["gold"] == 0]
    half = args.limit // 2
    rows = pos[:half] + neg[:half]
    random.shuffle(rows)
    texts = [r["text"] for r in rows]
    gold = [r["gold"] for r in rows]
    print(f"[data] evaluating on {len(rows)} rows | positives={sum(gold)} negatives={len(gold)-sum(gold)}\n")

    specs = list(REGISTRIES[args.task])
    if not args.include_ceiling:
        specs = [s for s in specs if s.deployable]
    if not args.include_parents:
        specs = [s for s in specs if not s.parent]

    import dataclasses
    results = []
    for spec in specs:
        hf_id = args.our_model if spec.hf_id == "__OURS__" else spec.hf_id
        spec = dataclasses.replace(spec, hf_id=hf_id)
        size = f"{spec.size_m:.0f}M" if spec.size_m else "?"
        print(f"[run] {spec.name} ({size}) <- {spec.hf_id}{('  ('+spec.note+')') if spec.note else ''}")
        if spec.generative:
            scores = run_generative_model(spec, texts, args.device, args.max_length)
        else:
            scores = run_model(spec, texts, args.device, args.batch_size, args.max_length)
        if scores is None:
            continue
        at_half = prf(gold, scores, 0.5)
        bt, at_best = best_threshold(gold, scores)
        tag = "parent" if spec.parent else ("deployable" if spec.deployable else "ceiling")
        results.append({"name": spec.name, "tag": tag, "size_m": spec.size_m,
                        "at_0.5": at_half, "best_threshold": bt, "at_best": at_best})

    # Leaderboard
    print("\n" + "=" * 92)
    print(f"  {args.task.upper()} BAKE-OFF  ({len(rows)} rows)")
    print("=" * 92)
    print(f"{'model':<15} {'tag':<11} {'size':>5} | {'F1@0.5':>7} {'P@0.5':>7} {'R@0.5':>7} | {'bestF1':>7} {'@thr':>5} {'P':>6} {'R':>6}")
    print("-" * 100)
    for x in sorted(results, key=lambda z: z["at_best"]["f1"], reverse=True):
        a, b = x["at_0.5"], x["at_best"]
        size = f"{x['size_m']:.0f}M" if x["size_m"] else "?"
        print(f"{x['name']:<15} {x['tag']:<11} {size:>5} | {a['f1']:>7.3f} {a['precision']:>7.3f} {a['recall']:>7.3f} | "
              f"{b['f1']:>7.3f} {x['best_threshold']:>5.2f} {b['precision']:>6.3f} {b['recall']:>6.3f}")
    print("=" * 100)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({"task": args.task, "n": len(rows), "results": results}, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
