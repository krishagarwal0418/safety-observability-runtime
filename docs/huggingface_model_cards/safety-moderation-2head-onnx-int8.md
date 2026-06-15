---
library_name: onnx
license: apache-2.0
language:
- en
tags:
- safety
- moderation
- observability
- onnx
- int8
- quantized
pipeline_tag: text-classification
---

# Safety Moderation 2-Head ONNX INT8

This repository contains the quantized ONNX INT8 moderation classifier used by
the CPU batch evaluation/deployment path in
[`safety-observability-runtime`](https://github.com/krishagarwal0418/safety-observability-runtime).

It is intended for **observability-only** safety counters, not as a standalone
moderation guardrail.

## Labels

| Model output | Runtime label |
|---|---|
| `harmful_content` | broad toxicity/hate/harassment counter |
| `sexual` | sexual-content counter |

`safe` is implicit when both scores are below configured thresholds.

## Quantization

The model was exported to ONNX and dynamically quantized to INT8 for CPU
deployment experiments.

Known artifact size from project runs:

| Artifact | Size |
|---|---:|
| FP32 ONNX | ~557 MB |
| INT8 ONNX | ~143 MB |

Small 500-row validation snapshot:

| Model | Macro F1 |
|---|---:|
| FP32 ONNX | 0.8556 |
| INT8 ONNX | 0.8807 |

The INT8 score being slightly higher is a sample-level effect, not a claim that
quantization improves the model.

## Current Runtime Metrics

CPU-only full-stack evaluation on 1,000 balanced scoped rows:

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| harmful_content | 0.7117 | 0.8464 | 0.7732 |
| sexual | 0.9731 | 0.8680 | 0.9175 |

CPU throughput in that run was approximately 3.36 rows/sec with average latency
around 298 ms/row. Moderation ONNX is the main CPU bottleneck.

## Usage

```python
import onnxruntime as ort
from transformers import AutoTokenizer

repo_dir = "path/to/downloaded/repo"
tokenizer = AutoTokenizer.from_pretrained(repo_dir)
session = ort.InferenceSession(f"{repo_dir}/model.onnx", providers=["CPUExecutionProvider"])
```

The runtime repository downloads this artifact via `configs/runtime.yaml`:

```bash
pip install git+https://github.com/krishagarwal0418/safety-observability-runtime.git
safety-observe download --config configs/runtime.yaml
```

## Limitations

- Current scoped thresholds are intended for toxicity, hate, harassment, and
  sexual counters.
- Self-harm, dangerous information, illegal activity, and violence are not
  claimed as reliable categories in the public runtime thresholds.
- Use for analytics and monitoring, not autonomous enforcement.
