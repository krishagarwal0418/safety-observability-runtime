---
library_name: transformers
license: apache-2.0
language:
- en
tags:
- prompt-injection
- safety
- observability
- text-classification
- deberta
pipeline_tag: text-classification
---

# Safety Prompt-Injection Classifier

This repository contains the non-quantized prompt-injection classifier used by
the [`safety-observability-runtime`](https://github.com/krishagarwal0418/safety-observability-runtime)
project.

The model is intended for **observability-only** detection of prompt-injection
and prompt-attack style inputs. It is not a standalone guardrail.

## Labels

| Model label | Runtime label |
|---|---|
| `SAFE` | no prompt-injection counter |
| `INJECTION` | `prompt_injection` |

## Training Context

The prompt-injection model was fine-tuned from a DeBERTa prompt-injection base
model using a mixture of public prompt-injection datasets and deterministic
positive augmentations such as masking, spacing, and zero-width character
variants.

Project fine-tuning snapshot:

| Metric | Value |
|---|---:|
| eval F1 | ~0.985 |
| eval PR-AUC | ~0.999 |

Those metrics are from the model-specific prompt-injection validation setup.
End-to-end runtime metrics are lower because the full stack evaluates mixed
traffic and routing behavior.

## Runtime Role

In the public runtime, this model is called only for uncertain attack-routed
rows. Obvious prompt-injection rows can be labeled directly by the FastText
router when calibrated high-confidence thresholds are met.

Current scoped runtime result on 1,000 balanced rows:

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| prompt_injection | 0.9268 | 0.9120 | 0.9194 |

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "Krishagarwal314/safety-prompt-injection"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo)
```

For the full observability pipeline, use the runtime repository rather than
calling this model directly:

```bash
pip install git+https://github.com/krishagarwal0418/safety-observability-runtime.git
safety-observe download --config configs/runtime.yaml
```

## Limitations

- Designed for prompt-injection/attack observability, not full moderation.
- Does not replace a policy engine or human review.
- Thresholds should be calibrated for your traffic distribution.

## Safety

This model emits monitoring signals. It should not be used as the sole basis for
blocking, rejecting, or taking action on user content.
