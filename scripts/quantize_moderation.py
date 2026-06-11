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
import json
import shutil
from pathlib import Path


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
    from transformers import AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/transformers/moderation")
    parser.add_argument("--output", default="models/onnx_int8/moderation")
    parser.add_argument("--fp32-output", default="models/onnx/moderation")
    parser.add_argument("--weight-type", choices=["quint8", "qint8"], default="quint8")
    parser.add_argument("--per-channel", action="store_true", default=True)
    parser.add_argument("--no-per-channel", dest="per_channel", action="store_false")
    parser.add_argument("--skip-fp32-export", action="store_true")
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
    (out_dir / "quantization_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
