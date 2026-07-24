#!/usr/bin/env python3
"""Analyze native, instrumented, online, and offline HASE GPU runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BACKEND = "Cuda_NvidiaGpu_GpuCuda"
MODES = (
    "baseline",
    "instrumented_baseline",
    "offline_history",
    "random",
    "learned_hybrid",
)
REPETITIONS = range(3)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_campaign(run_root: Path):
    runs = []
    for mode in MODES:
        for repetition in REPETITIONS:
            directory = run_root / "raw" / BACKEND / mode / f"rep-{repetition}"
            summary = load_json(directory / "summary.json")
            timings = load_jsonl(directory / "step-timings.jsonl")
            if len(timings) != summary["completed_steps"]:
                raise RuntimeError(
                    f"{directory}: {len(timings)} timings for "
                    f'{summary["completed_steps"]} completed steps'
                )
            metrics = []
            traces = []
            if mode != "baseline":
                metrics = load_jsonl(directory / "metrics.jsonl")
                traces = load_jsonl(directory / "trace.jsonl")
                if not metrics or not traces:
                    raise RuntimeError(f"{directory}: tuned run has no instrumentation output")
                fallbacks = [
                    item.get("learned_status")
                    for item in traces
                    if str(item.get("learned_status", "")).startswith("fallback_")
                ]
                if fallbacks:
                    raise RuntimeError(f"{directory}: learned fallback: {fallbacks}")
            runs.append(
                {
                    "mode": mode,
                    "repetition": repetition,
                    "directory": directory,
                    "summary": summary,
                    "timings": timings,
                    "metrics": metrics,
                    "traces": traces,
                }
            )
    return runs


def validate(runs):
    failures = []
    baseline = next(run for run in runs if run["mode"] == "baseline")
    reference = baseline["summary"]
    for run in runs:
        label = f'{run["mode"]}/rep-{run["repetition"]}'
        summary = run["summary"]
        if summary["timed_out"]:
            failures.append(f"{label}: timed out")
        if not summary["finite"]:
            failures.append(f"{label}: non-finite result")
        if summary["completed_steps"] != reference["completed_steps"]:
            failures.append(f"{label}: incomplete step count")
        for metric in ("phi_ase_sum", "beta_cells_sum", "beta_volume_sum"):
            if not np.isclose(summary[metric], reference[metric], rtol=1.0e-12, atol=0.0):
                failures.append(f"{label}: {metric} differs from baseline")
    return failures


def timing_matrix(runs, mode):
    selected = [run for run in runs if run["mode"] == mode]
    step_sets = [{item["step"] for item in run["timings"]} for run in selected]
    steps = sorted(set.intersection(*step_sets))
    matrices = []
    for run in selected:
        by_step = {item["step"]: item["elapsed_seconds"] for item in run["timings"]}
        matrices.append([by_step[step] for step in steps])
    return np.asarray(steps), np.asarray(matrices, dtype=float)


def stable_break_even(steps, delta):
    suffix_max = np.maximum.accumulate(delta[::-1])[::-1]
    candidates = np.flatnonzero(suffix_max <= 0.0)
    return None if candidates.size == 0 else int(steps[candidates[0]])


def final_slope(steps, curve, window=200):
    count = min(window, len(steps))
    return float(np.polyfit(steps[-count:], curve[-count:], 1)[0])


def metric_totals(run):
    totals = defaultdict(float)
    for item in run["metrics"]:
        for key in (
            "invocation_count",
            "measured_count",
            "replay_count",
            "host_call_seconds",
            "measured_host_call_seconds",
            "measured_kernel_seconds",
            "recommendation_seconds",
            "measured_recommendation_seconds",
            "estimated_measured_control_and_sync_seconds",
        ):
            totals[key] += item.get(key, 0.0)
    return totals


def write_run_csv(runs, output):
    fields = (
        "mode",
        "repetition",
        "completed_steps",
        "elapsed_seconds",
        "finite",
        "phi_ase_sum",
        "beta_cells_sum",
        "beta_volume_sum",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {key: run["summary"].get(key) for key in fields}
            row["mode"] = run["mode"]
            row["repetition"] = run["repetition"]
            writer.writerow(row)


def write_metrics_csv(runs, output):
    fields = (
        "mode",
        "repetition",
        "invocation_count",
        "measured_count",
        "replay_count",
        "host_call_seconds",
        "measured_host_call_seconds",
        "measured_kernel_seconds",
        "recommendation_seconds",
        "measured_recommendation_seconds",
        "estimated_measured_control_and_sync_seconds",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            if run["mode"] == "baseline":
                continue
            row = dict(metric_totals(run))
            row["mode"] = run["mode"]
            row["repetition"] = run["repetition"]
            writer.writerow(row)


def plot_timings(runs, output_dir):
    curves = {}
    for mode in MODES:
        steps, matrix = timing_matrix(runs, mode)
        curves[mode] = (steps, np.median(matrix, axis=0))

    figure, axis = plt.subplots(figsize=(10, 6))
    for mode, (steps, curve) in curves.items():
        axis.plot(steps, curve, label=mode)
    axis.set_xlabel("Completed outer steps")
    axis.set_ylabel("Cumulative application time (s)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "cumulative-application-time.png", dpi=180)
    plt.close(figure)

    baseline_steps, baseline_curve = curves["baseline"]
    figure, axis = plt.subplots(figsize=(10, 6))
    for mode in MODES[1:]:
        steps, curve = curves[mode]
        if not np.array_equal(steps, baseline_steps):
            raise RuntimeError(f"{mode}: step grid differs from baseline")
        axis.plot(steps, curve - baseline_curve, label=mode)
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xlabel("Completed outer steps")
    axis.set_ylabel("Cumulative time minus native baseline (s)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "cumulative-overhead-vs-baseline.png", dpi=180)
    plt.close(figure)
    return curves


def write_report(runs, curves, failures, output):
    _, baseline_curve = curves["baseline"]
    baseline_final = baseline_curve[-1]
    baseline_slope = final_slope(*curves["baseline"])
    lines = [
        "# HASEonGPU tuner overhead and break-even",
        "",
        "Three GPU repetitions per mode; cumulative curves use the pointwise median.",
        "",
        "| Mode | Median total (s) | Backend delta (s) | Backend delta (%) | Last-200-step slope (ms/step) | Stable break-even |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        steps, curve = curves[mode]
        total = statistics.median(
            run["summary"]["elapsed_seconds"] for run in runs if run["mode"] == mode
        )
        delta = curve[-1] - baseline_final
        break_even = None if mode == "baseline" else stable_break_even(steps, curve - baseline_curve)
        lines.append(
            f"| {mode} | {total:.6f} | {delta:+.6f} | "
            f"{delta / baseline_final:+.2%} | {final_slope(steps, curve) * 1e3:.4f} | "
            f"{break_even if break_even is not None else ('n/a' if mode == 'baseline' else 'none')} |"
        )
    lines.extend(
        [
            "",
            f"Native baseline terminal slope: {baseline_slope * 1e3:.4f} ms/step.",
            "",
            "## Tuner-side accounting",
            "",
            "| Mode | Launches | Measured launches | Host call (ms) | Recommendation (ms) | Estimated measured control/sync (ms) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES[1:]:
        totals = [metric_totals(run) for run in runs if run["mode"] == mode]
        med = lambda key: statistics.median(item[key] for item in totals)
        lines.append(
            f"| {mode} | {med('invocation_count'):.0f} | {med('measured_count'):.0f} | "
            f"{med('host_call_seconds') * 1e3:.3f} | "
            f"{med('recommendation_seconds') * 1e3:.3f} | "
            f"{med('estimated_measured_control_and_sync_seconds') * 1e3:.3f} |"
        )
    lines.extend(["", "## Validation", ""])
    if failures:
        lines.extend(f"- FAIL: {failure}" for failure in failures)
    else:
        lines.append("All runs completed with finite, baseline-matching numerical integrals.")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = args.run_root / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_campaign(args.run_root)
    failures = validate(runs)
    write_run_csv(runs, output_dir / "runs.csv")
    write_metrics_csv(runs, output_dir / "tuner-metrics.csv")
    curves = plot_timings(runs, output_dir)
    write_report(runs, curves, failures, output_dir / "report.md")
    if failures:
        raise SystemExit("validation failed; see evaluations/report.md")


if __name__ == "__main__":
    main()
