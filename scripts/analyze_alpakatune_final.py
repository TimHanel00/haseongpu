#!/usr/bin/env python3
"""Analyze the final paired online-adaptive/offline HASE A100 campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


BACKEND = "Cuda_NvidiaGpu_GpuCuda"
KERNEL = "AccumulateForwardPhiAse"
MODES = (
    "baseline_0",
    "random_online",
    "random_offline",
    "learned_online",
    "learned_offline",
    "baseline_1",
)
TUNED_MODES = MODES[1:5]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_runs(run_root: Path):
    runs = []
    for mode in MODES:
        directory = run_root / "raw" / BACKEND / mode
        summary = load_json(directory / "summary.json")
        timings = load_jsonl(directory / "step-timings.jsonl")
        if len(timings) != summary["completed_steps"]:
            raise RuntimeError(
                f"{directory}: {len(timings)} timings for "
                f'{summary["completed_steps"]} completed steps'
            )
        traces = []
        metrics = []
        if mode in TUNED_MODES:
            traces = load_jsonl(directory / "trace.jsonl")
            metrics = load_jsonl(directory / "metrics.jsonl")
            if not traces or not metrics:
                raise RuntimeError(f"{directory}: missing tuner instrumentation")
            for records, label in ((traces, "trace"), (metrics, "metrics")):
                kernels = {record["kernel"] for record in records}
                if kernels != {KERNEL}:
                    raise RuntimeError(
                        f"{directory}: unexpected {label} kernels {sorted(kernels)}"
                    )
        runs.append(
            {
                "mode": mode,
                "directory": directory,
                "summary": summary,
                "timings": timings,
                "traces": traces,
                "metrics": metrics,
            }
        )
    return runs


def baseline_curve(runs):
    baselines = [run for run in runs if run["mode"].startswith("baseline_")]
    step_grids = [[item["step"] for item in run["timings"]] for run in baselines]
    if step_grids[0] != step_grids[1]:
        raise RuntimeError("baseline step grids differ")
    elapsed = [
        sum(run["timings"][index]["elapsed_seconds"] for run in baselines)
        / len(baselines)
        for index in range(len(step_grids[0]))
    ]
    return step_grids[0], elapsed


def metric_totals(run):
    totals = defaultdict(float)
    for item in run["metrics"]:
        for key in (
            "invocation_count",
            "measured_count",
            "replay_count",
            "host_call_seconds",
            "measured_kernel_seconds",
            "recommendation_seconds",
            "estimated_measured_control_and_sync_seconds",
        ):
            totals[key] += item.get(key, 0.0)
    return totals


def first_sustained_break_even_step(run, reference):
    deltas = [
        item["elapsed_seconds"] - reference[index]
        for index, item in enumerate(run["timings"])
    ]
    for index, delta in enumerate(deltas):
        if delta <= 0.0 and all(value <= 0.0 for value in deltas[index:]):
            return run["timings"][index]["step"]
    return None


def validate(run_root: Path, runs):
    failures = []
    reference = runs[0]["summary"]
    for run in runs:
        summary = run["summary"]
        mode = run["mode"]
        if summary["timed_out"]:
            failures.append(f"{mode}: timed out")
        if not summary["finite"]:
            failures.append(f"{mode}: non-finite result")
        if summary["completed_steps"] != 1000:
            failures.append(f"{mode}: completed {summary['completed_steps']} steps")
        for metric in ("phi_ase_sum", "beta_cells_sum", "beta_volume_sum"):
            if not math.isclose(
                summary[metric], reference[metric], rel_tol=1.0e-12, abs_tol=0.0
            ):
                failures.append(f"{mode}: {metric} differs from baseline_0")

    for strategy in ("random", "learned"):
        before = (
            run_root / f"{strategy}-before-offline.sha256"
        ).read_text(encoding="utf-8").split()[0]
        after = (
            run_root / f"{strategy}-after-offline.sha256"
        ).read_text(encoding="utf-8").split()[0]
        if before != after:
            failures.append(f"{strategy}: offline run changed its input history")

        offline = next(
            run for run in runs if run["mode"] == f"{strategy}_offline"
        )
        if not any(trace.get("loaded_from_cache") for trace in offline["traces"]):
            failures.append(f"{strategy}_offline: no compatible history load")
        if metric_totals(offline)["measured_count"] != 0:
            failures.append(f"{strategy}_offline: unexpectedly measured kernels")

        online = next(run for run in runs if run["mode"] == f"{strategy}_online")
        if metric_totals(online)["measured_count"] == 0:
            failures.append(f"{strategy}_online: no measured kernels")
    return failures


def write_tables(run_root: Path, runs):
    output = run_root / "evaluations"
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "mode",
            "completed_steps",
            "elapsed_seconds",
            "finite",
            "phi_ase_sum",
            "beta_cells_sum",
            "beta_volume_sum",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {"mode": run["mode"]}
                | {key: run["summary"][key] for key in fields if key != "mode"}
            )

    with (output / "tuner-metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = (
            "mode",
            "invocation_count",
            "measured_count",
            "replay_count",
            "host_call_seconds",
            "measured_kernel_seconds",
            "recommendation_seconds",
            "estimated_measured_control_and_sync_seconds",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            if run["mode"] not in TUNED_MODES:
                continue
            totals = metric_totals(run)
            writer.writerow(
                {"mode": run["mode"]}
                | {key: totals[key] for key in fields if key != "mode"}
            )


def write_plots(run_root: Path, runs):
    output = run_root / "evaluations"
    steps, reference = baseline_curve(runs)

    figure, axis = plt.subplots(figsize=(10, 6))
    for run in runs:
        axis.plot(
            [item["step"] for item in run["timings"]],
            [item["elapsed_seconds"] for item in run["timings"]],
            label=run["mode"],
        )
    axis.set_xlabel("Completed outer steps")
    axis.set_ylabel("Cumulative application time (s)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "cumulative-application-time.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6))
    for run in runs:
        if run["mode"] not in TUNED_MODES:
            continue
        run_steps = [item["step"] for item in run["timings"]]
        if run_steps != steps:
            raise RuntimeError(f"{run['mode']}: step grid differs from baselines")
        axis.plot(
            steps,
            [
                item["elapsed_seconds"] - reference[index]
                for index, item in enumerate(run["timings"])
            ],
            label=run["mode"],
        )
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xlabel("Completed outer steps")
    axis.set_ylabel("Cumulative time minus mean native baseline (s)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "cumulative-delta-vs-baseline.png", dpi=180)
    plt.close(figure)


def write_report(run_root: Path, runs, failures):
    _, reference = baseline_curve(runs)
    baseline_total = sum(
        run["summary"]["elapsed_seconds"]
        for run in runs
        if run["mode"].startswith("baseline_")
    ) / 2.0
    baseline_steps = reference[-1]
    lines = [
        "# Final HASEonGPU A100 online-adaptive/offline comparison",
        "",
        "One continuous A100 allocation. Both native baselines and both",
        "online/offline strategy pairs ran 1000 time steps and 1000 pump steps.",
        "Only `AccumulateForwardPhiAse` was routed through alpakaTune.",
        "",
        "| Mode | Total (s) | Total delta baseline | Step loop (s) | Step delta baseline | Measured | Recommendation (ms) | Control/sync (ms) | Sustained break-even |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        total = run["summary"]["elapsed_seconds"]
        step_seconds = run["timings"][-1]["elapsed_seconds"]
        totals = metric_totals(run)
        break_even = (
            first_sustained_break_even_step(run, reference)
            if run["mode"] in TUNED_MODES
            else None
        )
        lines.append(
            f"| {run['mode']} | {total:.6f} | "
            f"{total - baseline_total:+.6f} s "
            f"({total / baseline_total - 1:+.2%}) | "
            f"{step_seconds:.6f} | {step_seconds - baseline_steps:+.6f} s "
            f"({step_seconds / baseline_steps - 1:+.2%}) | "
            f"{totals['measured_count']:.0f} | "
            f"{totals['recommendation_seconds'] * 1.0e3:.3f} | "
            f"{totals['estimated_measured_control_and_sync_seconds'] * 1.0e3:.3f} | "
            f"{break_even if break_even is not None else 'none'} |"
        )

    lines.extend(["", "## Online versus offline", ""])
    for strategy in ("random", "learned"):
        online = next(
            run for run in runs if run["mode"] == f"{strategy}_online"
        )
        offline = next(
            run for run in runs if run["mode"] == f"{strategy}_offline"
        )
        online_steps = online["timings"][-1]["elapsed_seconds"]
        offline_steps = offline["timings"][-1]["elapsed_seconds"]
        lines.append(
            f"- `{strategy}`: offline is {offline_steps - online_steps:+.6f} s "
            f"({offline_steps / online_steps - 1:+.2%}) relative to its online "
            "collection run; its history remained byte-for-byte unchanged."
        )

    lines.extend(["", "## Validation", ""])
    if failures:
        lines.extend(f"- FAIL: {failure}" for failure in failures)
    else:
        lines.append(
            "All six runs completed with finite, baseline-matching numerical "
            "integrals and the expected measurement/history contracts."
        )
    (run_root / "evaluations" / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.run_root / "evaluations"
    output.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.run_root)
    failures = validate(args.run_root, runs)
    write_tables(args.run_root, runs)
    write_plots(args.run_root, runs)
    write_report(args.run_root, runs, failures)
    if failures:
        raise SystemExit("validation failed; see evaluations/report.md")


if __name__ == "__main__":
    main()
