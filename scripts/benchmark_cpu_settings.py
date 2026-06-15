#!/usr/bin/env python3
"""Run quick CPU sweeps for batch size, max length, and ONNX thread settings."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/runtime.yaml")
    p.add_argument("--threshold-report", default=None)
    p.add_argument("--data", action="append", required=True)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--batch-sizes", default="16,32,64")
    p.add_argument("--max-lengths", default="96,128")
    p.add_argument("--intra-threads", default="0,2,4")
    p.add_argument("--inter-threads", default="1")
    p.add_argument("--exclude-label", action="append", default=[])
    p.add_argument("--output", default="reports/cpu_settings_sweep.json")
    args = p.parse_args()

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for batch, max_len, intra, inter in itertools.product(
        parse_csv_ints(args.batch_sizes),
        parse_csv_ints(args.max_lengths),
        parse_csv_ints(args.intra_threads),
        parse_csv_ints(args.inter_threads),
    ):
        run_out = out_dir / f"cpu_sweep_b{batch}_l{max_len}_intra{intra}_inter{inter}.json"
        cmd = [
            sys.executable,
            "scripts/evaluate_quantized_full_suite.py",
            "--config",
            args.config,
            "--limit",
            str(args.limit),
            "--batch-size",
            str(batch),
            "--max-length",
            str(max_len),
            "--onnx-intra-threads",
            str(intra),
            "--onnx-inter-threads",
            str(inter),
            "--fast-allow",
            "0.0",
            "--output",
            str(run_out),
        ]
        if args.threshold_report:
            cmd += ["--threshold-report", args.threshold_report]
        for path in args.data:
            cmd += ["--data", path]
        for label in args.exclude_label:
            cmd += ["--exclude-label", label]
        print("[sweep]", " ".join(cmd))
        subprocess.run(cmd, check=True)
        report = json.loads(run_out.read_text(encoding="utf-8"))
        results.append(
            {
                "batch_size": batch,
                "max_length": max_len,
                "onnx_intra_threads": intra,
                "onnx_inter_threads": inter,
                "throughput_rows_per_sec": report["latency_ms"]["throughput_rows_per_sec"],
                "estimated_per_row_avg": report["latency_ms"]["estimated_per_row_avg"],
                "macro_f1": report["performance"]["macro_f1"],
                "unsafe_any_f1": report["performance"]["unsafe_any_detection"]["f1"],
                "unsafe_false_pass_pct_of_unsafe": report["performance"]["unsafe_false_pass_pct_of_unsafe"],
                "path": str(run_out),
            }
        )
    results.sort(key=lambda r: (-r["throughput_rows_per_sec"], -r["macro_f1"]))
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"[sweep] wrote {args.output}")


if __name__ == "__main__":
    main()
