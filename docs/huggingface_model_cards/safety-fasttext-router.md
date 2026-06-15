---
library_name: fasttext
license: apache-2.0
language:
- en
tags:
- safety
- observability
- text-classification
- fasttext
- routing
pipeline_tag: text-classification
---

# Safety FastText Router

This repository contains the FastText routing head used by the
[`safety-observability-runtime`](https://github.com/krishagarwal0418/safety-observability-runtime)
project.

The model is intended for **observability-only** safety classification. It is
not a guardrail and should not be used as the sole enforcement layer for user
traffic.

## Outputs

The router predicts one of three coarse classes:

| FastText label | Runtime meaning |
|---|---|
| `attack` | Route to the prompt-injection confirmer. |
| `moderation` | Route to the moderation confirmer. |
| `safe` | Candidate low-risk traffic. Direct safe-pass is disabled in the public runtime thresholds. |

## Intended Use

Use this model as a cheap first-stage router in a larger observability stack:

```text
validation -> normalization -> deterministic rules -> FastText router -> BERT confirmers -> counters
```

The public runtime currently uses FastText for:

- routing prompt-injection candidates,
- routing moderation candidates,
- direct high-confidence prompt-injection classification for obvious cases.

Direct safe-pass is disabled because calibration showed too many unsafe false
passes for the current router.

## Current Runtime Scope

The checked-in runtime thresholds are calibrated for:

- `prompt_injection`
- `sexual`
- `harmful_content` for toxicity, hate, and harassment

The current runtime does not claim reliable coverage for:

- self-harm
- dangerous information
- illegal activity
- violence

## Usage

The runtime downloads this artifact automatically:

```bash
pip install git+https://github.com/krishagarwal0418/safety-observability-runtime.git
safety-observe download --config configs/runtime.yaml
```

Artifact:

- `router_head.ftz`

## Limitations

- This is a coarse router, not a final classifier.
- False routing decisions can affect whether a confirmer model is called.
- Do not enable direct safe-pass unless you recalibrate on your own traffic and
  verify a sufficiently low unsafe false-pass rate.

## Safety

This model is for analytics and monitoring. It should not be presented as a
policy-complete moderation system or used as an autonomous blocking mechanism.
