from __future__ import annotations

import argparse
import json
import sys

from .classifier import SafetyClassifier
from .download import download_models


def main() -> None:
    parser = argparse.ArgumentParser(prog="safety-slim")
    sub = parser.add_subparsers(dest="cmd", required=True)

    download = sub.add_parser("download")
    download.add_argument("--config", default=None)

    classify = sub.add_parser("classify")
    classify.add_argument("text", nargs="*")
    classify.add_argument("--config", default=None)
    classify.add_argument("--device", choices=("cpu", "cuda"), default=None)
    classify.add_argument("--onnx-provider", choices=("auto", "cpu", "cuda"), default=None)
    classify.add_argument("--max-length", type=int, default=None)
    classify.add_argument("--include-raw", action="store_true")

    args = parser.parse_args()
    if args.cmd == "download":
        print(json.dumps(download_models(args.config), indent=2))
        return

    text = " ".join(args.text).strip() or sys.stdin.read()
    clf = SafetyClassifier(
        args.config,
        device=args.device,
        onnx_provider=args.onnx_provider,
        max_length=args.max_length,
    )
    print(json.dumps(clf.classify(text, include_raw=args.include_raw), indent=2))


if __name__ == "__main__":
    main()
