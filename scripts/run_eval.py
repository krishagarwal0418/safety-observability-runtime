#!/usr/bin/env python3
"""
One-command clone-and-run evaluation entrypoint.

After `git clone`, a user only needs:

    export HF_TOKEN=hf_xxx        # required: models are private
    pip install -r requirements.txt
    python scripts/run_eval.py

This downloads both BERT models from HuggingFace, then runs the full pipeline
evaluation (both BERTs, mixed injection+harmful+sexual+safe corpus) with the
calibrated thresholds from configs/runtime.yaml.

Flags:
    --rows-per-label N   corpus size per category (default 200)
    --skip-download      models already present locally
    --device cuda|cpu
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INJECTION_REPO = "Krishagarwal314/safety-prompt-injection"
MODERATION_REPO = "Krishagarwal314/safety-moderation-2head"
INJECTION_DIR = REPO / "models/transformers/prompt_injection"
MODERATION_DIR = REPO / "models/transformers/moderation"


def _read_thresholds() -> tuple[float, float]:
    """Pull calibrated thresholds from configs/runtime.yaml (fall back to defaults)."""
    inj, mod = 0.50, 0.93
    cfg = REPO / "configs/runtime.yaml"
    if cfg.exists():
        try:
            import yaml
            data = yaml.safe_load(cfg.read_text())
            th = data.get("thresholds", {})
            inj = float(th.get("prompt_injection_review", inj))
            # harmful + sexual share the calibrated moderation threshold
            mod = float(th.get("harmful_content_review", mod))
        except Exception as e:
            print(f"[warn] could not parse runtime.yaml ({e}); using defaults")
    return inj, mod


def _download(token: str):
    from huggingface_hub import snapshot_download
    for repo, dst in ((INJECTION_REPO, INJECTION_DIR), (MODERATION_REPO, MODERATION_DIR)):
        if (dst / "config.json").exists():
            print(f"[download] {dst.name} already present — skipping")
            continue
        print(f"[download] {repo} -> {dst}")
        snapshot_download(repo_id=repo, local_dir=str(dst), token=token)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-per-label", type=int, default=200)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and not args.skip_download:
        sys.exit("ERROR: set HF_TOKEN (models are private). "
                 "export HF_TOKEN=hf_xxx  then re-run, or pass --skip-download if models are local.")

    if not args.skip_download:
        _download(token)

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"

    inj_thr, mod_thr = _read_thresholds()
    print(f"\n[run] device={args.device}  inj_thr={inj_thr}  mod_thr={mod_thr}\n")

    cmd = [
        sys.executable, str(REPO / "scripts/eval_full_pipeline.py"),
        "--injection-model", str(INJECTION_DIR),
        "--moderation-model", str(MODERATION_DIR),
        "--device", args.device,
        "--rows-per-label", str(args.rows_per_label),
        "--inj-threshold", str(inj_thr),
        "--mod-threshold", str(mod_thr),
        "--output", str(REPO / "reports/eval_full_pipeline.json"),
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
