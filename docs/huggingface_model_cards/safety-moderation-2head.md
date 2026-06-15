---
library_name: transformers
license: apache-2.0
language:
- en
tags:
- safety
- moderation
- observability
- text-classification
- deberta
pipeline_tag: text-classification
---

# Safety Moderation 2-Head Classifier

This repository contains the non-quantized moderation classifier used by the
[`safety-observability-runtime`](https://github.com/krishagarwal0418/safety-observability-runtime)
project.

The model is intended for **observability-only** safety counters. It is not a
standalone moderation guardrail.

## Labels

The model is a two-head/multi-label classifier:

| Model output | Runtime label |
|---|---|
| `harmful_content` | broad harmful-content counter |
| `sexual` | sexual-content counter |

`safe` is implicit when both scores are below configured thresholds.

## Current Supported Scope

The public runtime thresholds are calibrated for:

- sexual content,
- harmful content covering toxicity, hate, and harassment.

The current runtime does not claim reliable coverage for:

- self-harm,
- dangerous information,
- illegal activity,
- violence.

## Training Context

The model was fine-tuned from a Koala/Text-Moderation style base into a compact
two-label task:

```text
harmful_content = toxicity + hate + harassment + violence + self_harm
sexual = sexual
safe = implicit negative class
```

For the public scoped runtime, evaluation excludes labels that were weak or out
of scope for the current operating target.

## Current Runtime Metrics

CPU-only ONNX full-stack evaluation on 1,000 balanced scoped rows:

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| harmful_content | 0.7117 | 0.8464 | 0.7732 |
| sexual | 0.9731 | 0.8680 | 0.9175 |

Overall unsafe-any detection in the same run:

| Metric | Value |
|---|---:|
| precision | 0.9314 |
| recall | 0.9227 |
| F1 | 0.9270 |
| unsafe false-pass | 0.0% |

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "Krishagarwal314/safety-moderation-2head"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo)
```

For calibrated routing and thresholds, use the runtime repository:

```bash
pip install git+https://github.com/krishagarwal0418/safety-observability-runtime.git
safety-observe download --config configs/runtime.yaml
```

## Limitations

- The `harmful_content` label is broad and should not be interpreted as a
  precise policy taxonomy.
- Current scoped thresholds intentionally exclude several risk categories.
- This model should be used for monitoring/counters, not autonomous enforcement.
