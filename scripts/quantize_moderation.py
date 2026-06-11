#!/usr/bin/env python3
"""Export and quantize the 2-head moderation model to ONNX.

Example:
    python scripts/quantize_moderation.py \
      --model models/transformers/moderation \
      --output models/onnx_int8/moderation

Requires:
    pip install optimum[onnxruntime] onnxruntime
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


def _load_tokenizer(path: str | Path):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(str(path), fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(str(path))


def _copy_tokenizer_files(src: Path, dst: Path) -> None:
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spm.model",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
    ):
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)


def export_fp32(model_dir: Path, output_dir: Path) -> Path:
    from optimum.onnxruntime import ORTModelForSequenceClassification

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = _load_tokenizer(model_dir)
    model = ORTModelForSequenceClassification.from_pretrained(str(model_dir), export=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    _copy_tokenizer_files(model_dir, output_dir)
    onnx_files = sorted(output_dir.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX file exported in {output_dir}")
    return onnx_files[0]


def quantize_dynamic(fp32_onnx: Path, output_dir: Path, *, per_channel: bool, weight_type: str) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / fp32_onnx.name
    qtype = {
        "qint8": QuantType.QInt8,
        "quint8": QuantType.QUInt8,
    }[weight_type]
    quantize_dynamic(
        model_input=str(fp32_onnx),
        model_output=str(out),
        weight_type=qtype,
        per_channel=per_channel,
    )
    _copy_tokenizer_files(fp32_onnx.parent, output_dir)
    return out


def _iter_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("rows", [])
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    raise ValueError(f"Unsupported eval data format: {path}")


def _labels(row: dict[str, Any]) -> set[str]:
    raw = row.get("labels", row.get("label", []))
    if raw is None:
        raw = []
    elif isinstance(raw, (str, int, float, bool)):
        raw = [raw]
    labels = {str(x).lower().strip() for x in raw or []}
    out: set[str] = set()
    if "sexual" in labels:
        out.add("sexual")
    if labels & {
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
    }:
        out.add("harmful_content")
    return out


def _pick_text(row: dict[str, Any]) -> str:
    for key in ("text", "prompt", "instruction", "content", "input"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _score_model(model_dir: Path, data: Path, limit: int) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort
    from sklearn.metrics import precision_recall_fscore_support

    rows = [r for r in _iter_rows(data) if _pick_text(r)][:limit]
    tokenizer = _load_tokenizer(model_dir)
    cfg_path = model_dir / "config.json"
    id2label = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        id2label = {int(k): str(v).lower() for k, v in cfg.get("id2label", {}).items()}
    harmful_idx = next((i for i, label in id2label.items() if "harmful" in label), 0)
    sexual_idx = next((i for i, label in id2label.items() if "sexual" in label), 1)
    onnx_file = sorted(model_dir.glob("*.onnx"))[0]
    sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in sess.get_inputs()}
    gold = {"harmful_content": [], "sexual": []}
    pred = {"harmful_content": [], "sexual": []}
    for row in rows:
        enc = tokenizer(_pick_text(row), return_tensors="np", truncation=True, padding=True, max_length=128)
        ort_inputs = {k: v for k, v in enc.items() if k in input_names}
        logits = sess.run(None, ort_inputs)[0][0]
        probs = 1.0 / (1.0 + np.exp(-logits))
        labels = _labels(row)
        gold["harmful_content"].append(1 if "harmful_content" in labels else 0)
        gold["sexual"].append(1 if "sexual" in labels else 0)
        pred["harmful_content"].append(1 if harmful_idx < len(probs) and float(probs[harmful_idx]) >= 0.5 else 0)
        pred["sexual"].append(1 if sexual_idx < len(probs) and float(probs[sexual_idx]) >= 0.5 else 0)

    per_label = {}
    for label in ("harmful_content", "sexual"):
        p, r, f1, _ = precision_recall_fscore_support(
            gold[label], pred[label], average="binary", zero_division=0
        )
        per_label[label] = {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f1), 4)}
    macro_f1 = sum(per_label[label]["f1"] for label in ("harmful_content", "sexual")) / 2
    return {"rows": len(rows), "macro_f1": round(float(macro_f1), 4), "per_label": per_label}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/transformers/moderation")
    parser.add_argument("--output", default="models/onnx_int8/moderation")
    parser.add_argument("--fp32-output", default="models/onnx/moderation")
    parser.add_argument("--weight-type", choices=["quint8", "qint8"], default="quint8")
    parser.add_argument("--per-channel", action="store_true", default=True)
    parser.add_argument("--no-per-channel", dest="per_channel", action="store_false")
    parser.add_argument("--skip-fp32-export", action="store_true")
    parser.add_argument("--eval-data", default=None, help="Optional JSONL/CSV moderation eval file.")
    parser.add_argument("--eval-limit", type=int, default=500)
    args = parser.parse_args()

    model_dir = Path(args.model)
    fp32_dir = Path(args.fp32_output)
    out_dir = Path(args.output)
    fp32_onnx = sorted(fp32_dir.glob("*.onnx"))[0] if args.skip_fp32_export else export_fp32(model_dir, fp32_dir)
    quant_onnx = quantize_dynamic(
        fp32_onnx,
        out_dir,
        per_channel=args.per_channel,
        weight_type=args.weight_type,
    )
    report = {
        "source_model": str(model_dir),
        "fp32_onnx": str(fp32_onnx),
        "quantized_onnx": str(quant_onnx),
        "weight_type": args.weight_type,
        "per_channel": args.per_channel,
        "fp32_size_bytes": fp32_onnx.stat().st_size,
        "quantized_size_bytes": quant_onnx.stat().st_size,
    }
    if args.eval_data:
        eval_path = Path(args.eval_data)
        report["eval_limit"] = args.eval_limit
        report["fp32_eval"] = _score_model(fp32_dir, eval_path, args.eval_limit)
        report["quantized_eval"] = _score_model(out_dir, eval_path, args.eval_limit)
    (out_dir / "quantization_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
