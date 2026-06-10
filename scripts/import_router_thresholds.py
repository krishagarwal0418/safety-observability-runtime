#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--config", default="configs/runtime.yaml")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    route = report.get("route", {})
    if "attack" in route:
        cfg["thresholds"]["attack_route"] = float(route["attack"])
    if "moderation" in route:
        cfg["thresholds"]["moderation_route"] = float(route["moderation"])
    fast_allow = report.get("fast_allow", {})
    if "threshold" in fast_allow:
        cfg["thresholds"]["fast_allow"] = float(fast_allow["threshold"])
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"updated {cfg_path}")


if __name__ == "__main__":
    main()
