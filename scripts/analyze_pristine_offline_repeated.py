#!/usr/bin/env python3
"""Analyze the counterbalanced pristine fixed-FrameSpec comparison."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


BACKEND = "Cuda_NvidiaGpu_GpuCuda"
MODES = (
    "baseline_0",
    "offline_0",
    "offline_1",
    "baseline_1",
    "offline_2",
    "baseline_2",
    "baseline_3",
    "offline_3",
)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def elapsed_seconds(path):
    text = path.read_text(encoding="utf-8")
    match = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", text)
    if match is None:
        raise ValueError(f"missing GNU time elapsed value in {path}")
    fields = [float(value) for value in match.group(1).split(":")]
    if len(fields) == 2:
        return fields[0] * 60.0 + fields[1]
    if len(fields) == 3:
        return fields[0] * 3600.0 + fields[1] * 60.0 + fields[2]
    raise ValueError(f"unexpected GNU time elapsed value in {path}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.run_root / "raw" / BACKEND
    failures = []
    runs = []
    reference = None
    for mode in MODES:
        directory = raw / mode
        summary = load_json(directory / "summary.json")
        elapsed = elapsed_seconds(directory / "time.txt")
        metadata = (directory / "metadata.txt").read_text(encoding="utf-8")
        expected_flag = "ON" if mode.startswith("offline_") else "OFF"
        if f"HASE_FORWARD_OFFLINE_FRAMESPEC:BOOL={expected_flag}" not in metadata:
            failures.append(f"{mode}: wrong compiled offline FrameSpec flag")
        if "alpakaTune" in metadata:
            failures.append(f"{mode}: binary links alpakaTune")
        if summary["completed_steps"] != 1000 or summary["timed_out"] or not summary["finite"]:
            failures.append(f"{mode}: incomplete, timed out, or non-finite")
        if reference is None:
            reference = summary
        for key in ("phi_ase_sum", "beta_cells_sum", "beta_volume_sum"):
            if not math.isclose(summary[key], reference[key], rel_tol=1.0e-12, abs_tol=0.0):
                failures.append(f"{mode}: {key} differs from baseline")
        runs.append((mode, elapsed))

    by_mode = dict(runs)
    baseline = [elapsed for mode, elapsed in runs if mode.startswith("baseline_")]
    offline = [elapsed for mode, elapsed in runs if mode.startswith("offline_")]
    baseline_mean = statistics.mean(baseline)
    offline_mean = statistics.mean(offline)
    delta = offline_mean - baseline_mean
    delta_fraction = offline_mean / baseline_mean - 1.0
    block_deltas = [
        statistics.mean([by_mode["offline_0"], by_mode["offline_1"]])
        - statistics.mean([by_mode["baseline_0"], by_mode["baseline_1"]]),
        statistics.mean([by_mode["offline_2"], by_mode["offline_3"]])
        - statistics.mean([by_mode["baseline_2"], by_mode["baseline_3"]]),
    ]

    output = args.run_root / "evaluations"
    output.mkdir(parents=True, exist_ok=True)
    report = [
        "# Repeated pristine baseline versus fixed offline FrameSpec",
        "",
        "After one unmeasured 100-step warm-up per binary, all eight measured runs shared one A100 allocation and executed 1,000 application steps with normal per-step openPMD snapshots. The order `B-F-F-B / F-B-B-F` counterbalances linear drift. Performance is external GNU `/usr/bin/time` wall clock; no internal or tuner instrumentation was enabled.",
        "",
        "| Run | External wall time (s) |",
        "|---|---:|",
    ]
    report.extend(f"| {mode} | {elapsed:.2f} |" for mode, elapsed in runs)
    report.extend(
        [
            "",
            "| Configuration | Mean (s) | Sample SD (s) |",
            "|---|---:|---:|",
            f"| original FrameSpec | {baseline_mean:.3f} | {statistics.stdev(baseline):.3f} |",
            f"| fixed offline winners | {offline_mean:.3f} | {statistics.stdev(offline):.3f} |",
            "",
            f"The fixed offline configuration changed mean wall time by {delta:+.3f} s ({delta_fraction:+.2%}).",
            f"Counterbalanced block deltas (offline minus baseline): {block_deltas[0]:+.3f} s and {block_deltas[1]:+.3f} s.",
            "",
            "## Validation",
            "",
        ]
    )
    report.extend([f"- FAIL: {failure}" for failure in failures] or ["All runs completed with baseline-identical numerical integrals."])
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("validation failed; see evaluations/report.md")


if __name__ == "__main__":
    main()
