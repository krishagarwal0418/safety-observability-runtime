---
library_name: onnx
license: apache-2.0
language:
- en
tags:
- prompt-injection
- safety
- observability
- onnx
- int8
- quantized
pipeline_tag: text-classification
---

# Safety Prompt-Injection ONNX INT8

This repository contains the quantized ONNX INT8 prompt-injection classifier
used by the CPU batch evaluation/deployment path in
[`safety-observability-runtime`](https://github.com/krishagarwal0418/safety-observability-runtime).

It is intended for **observability-only** detection, not as a standalone
guardrail.

## Labels

| Model label | Runtime label |
|---|---|
| `SAFE` | no prompt-injection counter |
| `INJECTION` | `prompt_injection` |

## Artifact

- `model.onnx`
- tokenizer files
- `config.json`

## Runtime Role

The runtime uses this artifact in CPU ONNX evaluation scripts. The default
Python runtime classifier uses the Transformers artifact unless configured
otherwise.

Current scoped full-stack CPU result on 1,000 balanced rows:

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| prompt_injection | 0.9268 | 0.9120 | 0.9194 |

FastText directly handled 15.7% of rows as prompt-injection with 0.9809
precision in that run, reducing prompt ONNX calls.

## Usage

Use through ONNX Runtime:

```python
import onnxruntime as ort
from transformers import AutoTokenizer

repo_dir = "path/to/downloaded/repo"
tokenizer = AutoTokenizer.from_pretrained(repo_dir)
session = ort.InferenceSession(f"{repo_dir}/model.onnx", providers=["CPUExecutionProvider"])
```

The easiest path is the runtime repository:

```bash
pip install git+https://github.com/krishagarwal0418/safety-observability-runtime.git
safety-observe download --config configs/runtime.yaml
```

## Limitations

- Quantization can shift score calibration. Use runtime thresholds or
  recalibrate on your traffic.
- The model only detects prompt-injection/attack style content.
- This is an observability artifact, not an enforcement policy.
