#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e .
safety-slim download --config configs/runtime.yaml
