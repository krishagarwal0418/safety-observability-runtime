from __future__ import annotations

import argparse
import json
import sys

from .download import download_models
from .pipeline import SafetyObservabilityClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--config", default=None)
    c = sub.add_parser("classify")
    c.add_argument("text", nargs="*")
    c.add_argument("--config", default=None)
    c.add_argument("--full-scan", action="store_true")
    c.add_argument("--include-raw", action="store_true")
    args = parser.parse_args()
    if args.cmd == "download":
        print(json.dumps(download_models(args.config), indent=2))
        return
    text = " ".join(args.text).strip() or sys.stdin.read()
    clf = SafetyObservabilityClassifier(args.config)
    print(json.dumps(clf.classify(text, full_scan=args.full_scan, include_raw=args.include_raw), indent=2))


if __name__ == "__main__":
    main()
