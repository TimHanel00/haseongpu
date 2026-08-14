#!/usr/bin/env python3
"""Validate and summarize the targeted A100 FrameSpec campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


BACKEND = "Cuda_NvidiaGpu_GpuCuda"
MODES = (
    "baseline_0",
    "instrumented_baseline",
    "learned_fresh",
    "exhaustive_collect",
    "offline_best",
    "learned_resume",
    "baseline_1",
)
TUNED = tuple(mode for mode in MODES if not mode.startswith("baseline_"))
LEARNED = ("learned_fresh", "learned_resume")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def totals(records):
    result = defaultdict(float)
    for record in records:
        for key, value in record.items():
            if key != "kernel" and isinstance(value, (int, float)):
                result[key] += value
    return result


def sustained_break_even(records, reference):
    deltas = [item["elapsed_seconds"] - reference[index] for index, item in enumerate(records)]
    suffix_maximum = -math.inf
    first = None
    for index in range(len(deltas) - 1, -1, -1):
        suffix_maximum = max(suffix_maximum, deltas[index])
        if suffix_maximum <= 0.0:
            first = records[index]["step"]
    return first


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.run_root / "raw" / BACKEND
    runs = []
    failures = []
    reference = None

    for mode in MODES:
        directory = raw / mode
        summary = load_json(directory / "summary.json")
        steps = load_jsonl(directory / "step-timings.jsonl")
        traces = load_jsonl(directory / "trace.jsonl") if mode in TUNED else []
        metrics = load_jsonl(directory / "metrics.jsonl") if mode in TUNED else []
        run = {"mode": mode, "summary": summary, "steps": steps, "traces": traces, "metrics": metrics}
        runs.append(run)
        if reference is None:
            reference = summary
        if summary["timed_out"] or not summary["finite"] or summary["completed_steps"] != 10000:
            failures.append(f"{mode}: incomplete or non-finite run")
        for key in ("phi_ase_sum", "beta_cells_sum", "beta_volume_sum"):
            if not math.isclose(summary[key], reference[key], rel_tol=1.0e-12, abs_tol=0.0):
                failures.append(f"{mode}: {key} differs from baseline")
        if mode not in TUNED:
            continue
        if not traces or not metrics:
            failures.append(f"{mode}: missing tuner telemetry")
            continue
        expected_profile = "expanded" if mode == "instrumented_baseline" else "forward_traversal"
        expected_candidates = 1 if mode == "instrumented_baseline" else 30
        if {item.get("frame_tuning_profile") for item in traces} != {expected_profile}:
            failures.append(f"{mode}: unexpected tuning profile")
        if {item.get("candidate_count") for item in traces} != {expected_candidates}:
            failures.append(f"{mode}: raw candidate count is not {expected_candidates}")
        for item in traces:
            ratio = item["selected_coverage"] / item["coverage"]
            if mode == "instrumented_baseline" and (
                item["selected_num_frames"] != item["original_num_frames"]
                or item["selected_frame_extent"] != item["original_frame_extent"]
            ):
                failures.append("instrumented_baseline: did not retain the original FrameSpec")
                break
            if mode != "instrumented_baseline" and not 0.75 <= ratio <= 2.0:
                failures.append(f"{mode}: selected coverage ratio {ratio} is outside the profile")
                break
        statuses = {item.get("learned_status") for item in traces if item.get("learned_status") is not None}
        if (mode.startswith("learned_") or mode == "instrumented_baseline") and statuses != {"active"}:
            failures.append(f"{mode}: learned status is {sorted(statuses)}")
        if mode == "offline_best" and any(item["measured"] for item in traces):
            failures.append("offline_best: replay unexpectedly measured a kernel")

    baselines = [run for run in runs if run["mode"].startswith("baseline_")]
    baseline_curve = [
        sum(run["steps"][index]["elapsed_seconds"] for run in baselines) / len(baselines)
        for index in range(len(baselines[0]["steps"]))
    ]
    baseline_total = sum(run["summary"]["elapsed_seconds"] for run in baselines) / len(baselines)
    baseline_steps = baseline_curve[-1]
    instrumented = next(run for run in runs if run["mode"] == "instrumented_baseline")
    instrumented_curve = [item["elapsed_seconds"] for item in instrumented["steps"]]
    rows = []
    for run in runs:
        metric = totals(run["metrics"])
        selections = Counter(
            (item["original_num_frames"][0], item["selected_frame_extent"][0], item["selected_num_frames"][0])
            for item in run["traces"]
        )
        rows.append(
            {
                "mode": run["mode"],
                "total_seconds": run["summary"]["elapsed_seconds"],
                "step_loop_seconds": run["steps"][-1]["elapsed_seconds"],
                "measured_count": int(metric["measured_count"]),
                "replay_count": int(metric["replay_count"]),
                "distinct_selections": len(selections),
                "largest_selection_share": max(selections.values()) / sum(selections.values()) if selections else 0.0,
                "recommendation_seconds": metric["recommendation_seconds"],
                "control_sync_seconds": metric["estimated_measured_control_and_sync_seconds"],
                "break_even_native": sustained_break_even(run["steps"], baseline_curve)
                if run["mode"] not in ("baseline_0", "baseline_1")
                else None,
                "break_even_instrumented": sustained_break_even(run["steps"], instrumented_curve)
                if run["mode"] != "instrumented_baseline"
                else None,
            }
        )

    learned_winners = [
        row
        for row in rows
        if row["mode"] in LEARNED
        and row["step_loop_seconds"] < baseline_steps
        and row["step_loop_seconds"] < instrumented_curve[-1]
        and row["break_even_native"] is not None
        and row["break_even_instrumented"] is not None
    ]
    if not learned_winners:
        failures.append(
            "no learned strategy achieved sustained cumulative break-even against both native and instrumented baselines"
        )

    output = args.run_root / "evaluations"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# A100 targeted FrameSpec and ML strategy campaign",
        "",
        "All modes used one continuous A100 allocation and 10,000 application rounds. The targeted traversal profile contains 30 raw and 15--16 legal candidates per runtime context. `instrumented_baseline` uses the identical original FrameSpec through the learned tuner, including model inference, device timing, trace, metrics, and history overhead.",
        "",
        "| Mode | Total (s) | Delta baseline | Step loop (s) | Step delta | Measured | Replayed | Distinct | Largest share | Recommend (ms) | Control/sync (ms) | Break-even native | Break-even instrumented |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['total_seconds']:.6f} | {row['total_seconds'] - baseline_total:+.6f} ({row['total_seconds'] / baseline_total - 1:+.2%}) | "
            f"{row['step_loop_seconds']:.6f} | {row['step_loop_seconds'] - baseline_steps:+.6f} ({row['step_loop_seconds'] / baseline_steps - 1:+.2%}) | "
            f"{row['measured_count']} | {row['replay_count']} | {row['distinct_selections']} | {row['largest_selection_share']:.2%} | "
            f"{row['recommendation_seconds'] * 1e3:.3f} | {row['control_sync_seconds'] * 1e3:.3f} | "
            f"{row['break_even_native'] if row['break_even_native'] is not None else 'none'} | "
            f"{row['break_even_instrumented'] if row['break_even_instrumented'] is not None else 'none'} |"
        )

    offline = next(run for run in runs if run["mode"] == "offline_best")
    lines.extend(["", "## Offline winners", ""])
    for item in sorted(offline["traces"], key=lambda value: value["original_num_frames"][0]):
        lines.append(
            f"- `{item['original_num_frames'][0]} x {item['original_frame_extent'][0]}` -> "
            f"`{item['selected_num_frames'][0]} x {item['selected_frame_extent'][0]}` "
            f"({item['selected_coverage'] / item['coverage']:.3f}x coverage)"
        )
    lines.extend(["", "## Validation", ""])
    lines.extend([f"- FAIL: {failure}" for failure in failures] or ["All runs completed with baseline-identical numerical integrals and valid targeted candidates."])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("validation failed; see evaluations/report.md")


if __name__ == "__main__":
    main()
