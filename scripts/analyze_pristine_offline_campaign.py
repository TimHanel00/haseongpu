#!/usr/bin/env python3
"""Validate and summarize the uninstrumented offline FrameSpec comparison."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


BACKEND = "Cuda_NvidiaGpu_GpuCuda"
MODES = ("baseline_0", "offline_fixed", "baseline_1")


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
        expected_flag = "ON" if mode == "offline_fixed" else "OFF"
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
        runs.append((mode, elapsed, summary))

    baseline_seconds = (runs[0][1] + runs[2][1]) / 2.0
    offline_seconds = runs[1][1]
    delta_seconds = offline_seconds - baseline_seconds
    delta_fraction = offline_seconds / baseline_seconds - 1.0
    output = args.run_root / "evaluations"
    output.mkdir(parents=True, exist_ok=True)
    report = [
        "# Pristine baseline versus fixed offline FrameSpec",
        "",
        "All three runs shared one A100 allocation and executed 1,000 application steps with the normal per-step openPMD snapshot cadence. Performance is the external GNU `/usr/bin/time` wall clock. No run enabled internal step timing, frontend timing output, tuner device timing, model inference, trace, metrics, or history.",
        "",
        "| Mode | External wall time (s) | Delta from bracketed baseline |",
        "|---|---:|---:|",
    ]
    for mode, elapsed, _summary in runs:
        report.append(
            f"| {mode} | {elapsed:.2f} | {elapsed - baseline_seconds:+.2f} ({elapsed / baseline_seconds - 1.0:+.2%}) |"
        )
    report.extend(
        [
            f"| baseline bracket average | {baseline_seconds:.2f} | +0.00 (+0.00%) |",
            "",
            f"The fixed offline configuration changed total wall time by {delta_seconds:+.2f} s ({delta_fraction:+.2%}).",
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
