#!/usr/bin/env python3
"""Export GPU-oriented FP16 ONNX artifacts for prompt and moderation models.

This is intended for CUDAExecutionProvider / TensorRT-style deployment
experiments. CPU deployment should keep using the INT8 ONNX artifacts.

Requires:
    pip install "optimum[onnxruntime]" onnx onnxconverter-common onnxruntime-gpu
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spm.model",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
)


def load_tokenizer(path: str | Path):
    from transformers import AutoTokenizer

    # Do NOT pass fix_mistral_regex=True — it changes tokenization away from
    # what these DeBERTa models were fine-tuned with and collapses outputs.
    # See evaluate_pytorch_gpu_suite.load_tokenizer for the full explanation.
    return AutoTokenizer.from_pretrained(str(path))


def copy_aux_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in TOKENIZER_FILES:
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)


def export_fp32(model_dir: Path, output_dir: Path, *, overwrite: bool) -> Path:
    from optimum.onnxruntime import ORTModelForSequenceClassification

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("*.onnx"))
    if existing and not overwrite:
        return existing[0]

    tokenizer = load_tokenizer(model_dir)
    model = ORTModelForSequenceClassification.from_pretrained(str(model_dir), export=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    copy_aux_files(model_dir, output_dir)

    onnx_files = sorted(output_dir.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX file exported in {output_dir}")
    return onnx_files[0]


def convert_to_fp16(fp32_onnx: Path, output_dir: Path, *, overwrite: bool) -> Path:
    import onnx
    from onnx import TensorProto, numpy_helper
    from onnxconverter_common import float16
    import numpy as np

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / fp32_onnx.name
    if output.exists() and not overwrite:
        return output

    model = onnx.load(str(fp32_onnx))
    fp16_model = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=False,
    )
    normalize_float_constants_to_fp16(fp16_model, numpy_helper, TensorProto, np)
    clear_intermediate_type_annotations(fp16_model)
    onnx.checker.check_model(fp16_model)
    onnx.save(fp16_model, str(output))
    copy_aux_files(fp32_onnx.parent, output_dir)
    return output


def normalize_float_constants_to_fp16(model, numpy_helper, tensor_proto, np) -> None:
    """Convert leftover float32 constants after FP16 graph conversion.

    DeBERTa exports can leave scalar Constant nodes in float32 while neighboring
    tensors are FP16, which ONNX Runtime rejects for ops such as Mul. Keeping
    graph inputs/outputs in their original types is fine; this pass only touches
    embedded initializers and Constant node tensor attributes.
    """

    def convert_tensor(tensor):
        if tensor.data_type != tensor_proto.FLOAT:
            return tensor
        arr = numpy_helper.to_array(tensor).astype(np.float16)
        return numpy_helper.from_array(arr, name=tensor.name)

    graph = model.graph
    for idx, initializer in enumerate(graph.initializer):
        if initializer.data_type == tensor_proto.FLOAT:
            graph.initializer[idx].CopyFrom(convert_tensor(initializer))

    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.type == attr.TENSOR and attr.t.data_type == tensor_proto.FLOAT:
                attr.t.CopyFrom(convert_tensor(attr.t))
            elif attr.type == attr.TENSORS:
                for idx, tensor in enumerate(attr.tensors):
                    if tensor.data_type == tensor_proto.FLOAT:
                        attr.tensors[idx].CopyFrom(convert_tensor(tensor))


def clear_intermediate_type_annotations(model) -> None:
    """Remove stale intermediate type metadata after FP16 conversion.

    Some Optimum/DeBERTa exports carry value_info entries inferred before
    conversion. ONNX Runtime can then see a Cast output annotated as float32 even
    though the converted node now produces float16. Inputs and outputs are kept;
    intermediate types are cheap for ORT to infer at load time.
    """

    del model.graph.value_info[:]


def process_one(name: str, model_dir: Path, root: Path, *, overwrite: bool) -> dict[str, object]:
    fp32_dir = root / "fp32" / name
    fp16_dir = root / "fp16" / name
    fp32 = export_fp32(model_dir, fp32_dir, overwrite=overwrite)
    fp16 = convert_to_fp16(fp32, fp16_dir, overwrite=overwrite)
    return {
        "name": name,
        "source_model": str(model_dir),
        "fp32_dir": str(fp32_dir),
        "fp16_dir": str(fp16_dir),
        "fp32_onnx": str(fp32),
        "fp16_onnx": str(fp16),
        "fp32_size_bytes": fp32.stat().st_size,
        "fp16_size_bytes": fp16.stat().st_size,
        "size_reduction_pct": round(100 * (1 - (fp16.stat().st_size / fp32.stat().st_size)), 2)
        if fp32.stat().st_size
        else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-model", default="models/transformers/prompt_injection")
    parser.add_argument("--moderation-model", default="models/transformers/moderation")
    parser.add_argument("--output-root", default="models/onnx_gpu")
    parser.add_argument("--report", default="reports/gpu_onnx_fp16_export.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root)
    results = {
        "prompt_injection": process_one(
            "prompt_injection",
            Path(args.prompt_model),
            root,
            overwrite=args.overwrite,
        ),
        "moderation": process_one(
            "moderation",
            Path(args.moderation_model),
            root,
            overwrite=args.overwrite,
        ),
    }
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"[gpu-export] wrote {report}")


if __name__ == "__main__":
    main()
