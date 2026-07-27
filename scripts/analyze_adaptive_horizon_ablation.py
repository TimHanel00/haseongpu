#!/usr/bin/env python3
"""Analyze the A100 online-adaptive horizon ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


BACKEND = "Cuda_NvidiaGpu_GpuCuda"
KERNEL = "AccumulateForwardPhiAse"
MODES = (
    "baseline_0",
    "random_horizon",
    "learned_continuous",
    "learned_horizon",
    "random_continuous",
    "baseline_1",
)
TUNED_MODES = tuple(mode for mode in MODES if not mode.startswith("baseline_"))
FRAME_EXTENTS = (32, 64, 128, 256, 512, 1024)


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
        traces = []
        metrics = []
        if mode in TUNED_MODES:
            traces = load_jsonl(directory / "trace.jsonl")
            metrics = load_jsonl(directory / "metrics.jsonl")
        runs.append(
            {
                "mode": mode,
                "summary": summary,
                "timings": timings,
                "traces": traces,
                "metrics": metrics,
            }
        )
    return runs


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


def default_num_frames(upper_limit):
    splits = []
    value = upper_limit
    while True:
        splits.append(value)
        if value == 1:
            break
        value //= 2
    values = set(splits)
    ascending = sorted(splits)
    for lower, upper in zip(ascending, ascending[1:]):
        values.add(lower + (upper - lower) // 2)
    return values


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


def validate(runs):
    failures = []
    baseline = runs[0]["summary"]
    for run in runs:
        mode = run["mode"]
        summary = run["summary"]
        if summary["timed_out"]:
            failures.append(f"{mode}: timed out")
        if not summary["finite"]:
            failures.append(f"{mode}: non-finite result")
        if summary["completed_steps"] != 1000:
            failures.append(f"{mode}: completed {summary['completed_steps']} steps")
        if len(run["timings"]) != summary["completed_steps"]:
            failures.append(f"{mode}: step timing count differs from summary")
        for metric in ("phi_ase_sum", "beta_cells_sum", "beta_volume_sum"):
            if not math.isclose(
                summary[metric], baseline[metric], rel_tol=1.0e-12, abs_tol=0.0
            ):
                failures.append(f"{mode}: {metric} differs from baseline")
        if mode not in TUNED_MODES:
            continue
        if not run["traces"] or not run["metrics"]:
            failures.append(f"{mode}: missing tuner instrumentation")
        if {item["kernel"] for item in run["traces"]} != {KERNEL}:
            failures.append(f"{mode}: unexpected trace kernel")
        statuses = {
            item.get("learned_status")
            for item in run["traces"]
            if item.get("learned_status") is not None
        }
        if mode.startswith("learned_") and statuses != {"active"}:
            failures.append(f"{mode}: learned statuses are {sorted(statuses)}")
        totals = metric_totals(run)
        if totals["measured_count"] != totals["invocation_count"]:
            failures.append(f"{mode}: not every invocation was measured")
        if totals["replay_count"] != 0:
            failures.append(f"{mode}: unexpected unmeasured replay")
        if all("candidate_count" in item for item in run["traces"]):
            for item in run["traces"]:
                original_num_frames = item["original_num_frames"][0]
                expected_num_frames = default_num_frames(4 * original_num_frames)
                expected_count = len(FRAME_EXTENTS) * len(expected_num_frames)
                if item["candidate_count"] != expected_count:
                    failures.append(
                        f"{mode}: candidate count {item['candidate_count']} differs "
                        f"from {expected_count} for numFrames={original_num_frames}"
                    )
                    break
                if item["rejected_candidate_count"] != 0:
                    failures.append(f"{mode}: default HASE candidate was rejected")
                    break
                if item["selected_frame_extent"][0] not in FRAME_EXTENTS:
                    failures.append(f"{mode}: unexpected selected frame extent")
                    break
                if item["selected_num_frames"][0] not in expected_num_frames:
                    failures.append(f"{mode}: unexpected selected numFrames")
                    break
    return failures


def write_outputs(run_root: Path, runs, failures):
    output = run_root / "evaluations"
    output.mkdir(parents=True, exist_ok=True)
    steps, baseline_elapsed = baseline_curve(runs)
    baseline_total = sum(
        run["summary"]["elapsed_seconds"]
        for run in runs
        if run["mode"].startswith("baseline_")
    ) / 2.0
    baseline_steps = baseline_elapsed[-1]

    rows = []
    for run in runs:
        totals = metric_totals(run)
        candidate_counts = Counter(
            item["candidate_index"]
            for item in run["traces"]
            if item.get("measured")
        )
        measured = int(totals["measured_count"])
        candidate_space_sizes = {
            item["candidate_count"]
            for item in run["traces"]
            if "candidate_count" in item
        }
        rows.append(
            {
                "mode": run["mode"],
                "total_seconds": run["summary"]["elapsed_seconds"],
                "step_loop_seconds": run["timings"][-1]["elapsed_seconds"],
                "invocation_count": int(totals["invocation_count"]),
                "measured_count": measured,
                "distinct_candidates": len(candidate_counts),
                "candidate_space_sizes": ",".join(
                    str(size) for size in sorted(candidate_space_sizes)
                ),
                "largest_candidate_share": (
                    max(candidate_counts.values()) / measured if measured else 0.0
                ),
                "recommendation_seconds": totals["recommendation_seconds"],
                "control_sync_seconds": totals[
                    "estimated_measured_control_and_sync_seconds"
                ],
                "measured_kernel_seconds": totals["measured_kernel_seconds"],
            }
        )

    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

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
            raise RuntimeError(f"{run['mode']}: step grid differs from baseline")
        axis.plot(
            steps,
            [
                item["elapsed_seconds"] - baseline_elapsed[index]
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

    lines = [
        "# A100 online-adaptive horizon ablation",
        "",
        "One continuous A100 allocation; every case used 1000 time steps and "
        "1000 pump steps. Only `AccumulateForwardPhiAse` was instrumented.",
        "Random used the explicit nondeterministic seed mode in both cases.",
        "",
        "| Mode | Total (s) | Total delta baseline | Step loop (s) | Step delta baseline | Measured | Candidate spaces | Distinct candidates | Largest share | Recommendation (ms) | Control/sync (ms) |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    by_mode = {row["mode"]: row for row in rows}
    for row in rows:
        total = row["total_seconds"]
        step_loop = row["step_loop_seconds"]
        lines.append(
            f"| {row['mode']} | {total:.6f} | "
            f"{total - baseline_total:+.6f} s ({total / baseline_total - 1:+.2%}) | "
            f"{step_loop:.6f} | "
            f"{step_loop - baseline_steps:+.6f} s ({step_loop / baseline_steps - 1:+.2%}) | "
            f"{row['measured_count']} | {row['candidate_space_sizes']} | "
            f"{row['distinct_candidates']} | "
            f"{row['largest_candidate_share']:.2%} | "
            f"{row['recommendation_seconds'] * 1.0e3:.3f} | "
            f"{row['control_sync_seconds'] * 1.0e3:.3f} |"
        )

    lines.extend(["", "## Horizon effect", ""])
    for strategy in ("random", "learned"):
        horizon = by_mode[f"{strategy}_horizon"]
        continuous = by_mode[f"{strategy}_continuous"]
        lines.append(
            f"- `{strategy}` continuous minus horizon: "
            f"{continuous['step_loop_seconds'] - horizon['step_loop_seconds']:+.6f} s "
            f"({continuous['step_loop_seconds'] / horizon['step_loop_seconds'] - 1:+.2%}) "
            "in the step loop."
        )

    lines.extend(["", "## Validation", ""])
    if failures:
        lines.extend(f"- FAIL: {failure}" for failure in failures)
    else:
        lines.append(
            "All six runs completed with finite, baseline-matching numerical "
            "integrals; every adaptive invocation was measured and no learned "
            "fallback occurred."
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = load_runs(args.run_root)
    failures = validate(runs)
    write_outputs(args.run_root, runs, failures)
    if failures:
        raise SystemExit("validation failed; see evaluations/report.md")


if __name__ == "__main__":
    main()
