#!/usr/bin/env python3
"""Analyze the paired history-aware HASEonGPU GPU campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


BACKEND = "Cuda_NvidiaGpu_GpuCuda"
MODES = (
    "baseline",
    "learned_collect",
    "learned_resume",
    "random_collect",
    "offline_replay",
)


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
        if mode != "baseline":
            traces = load_jsonl(directory / "trace.jsonl")
            metrics = load_jsonl(directory / "metrics.jsonl")
            if not traces or not metrics:
                raise RuntimeError(f"{directory}: missing tuner instrumentation")
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
                "directory": directory,
                "summary": summary,
                "timings": timings,
                "traces": traces,
                "metrics": metrics,
            }
        )
    return runs


def validate(run_root: Path, runs):
    failures = []
    baseline = runs[0]["summary"]
    for run in runs:
        summary = run["summary"]
        label = run["mode"]
        if summary["timed_out"]:
            failures.append(f"{label}: timed out")
        if not summary["finite"]:
            failures.append(f"{label}: non-finite result")
        if summary["completed_steps"] != baseline["completed_steps"]:
            failures.append(f"{label}: incomplete step count")
        for metric in ("phi_ase_sum", "beta_cells_sum", "beta_volume_sum"):
            if not math.isclose(
                summary[metric], baseline[metric], rel_tol=1.0e-12, abs_tol=0.0
            ):
                failures.append(f"{label}: {metric} differs from baseline")

    pairs = (
        ("learned-before-resume.sha256", "learned-after-resume.sha256"),
        ("random-before-offline.sha256", "random-after-offline.sha256"),
    )
    for before_name, after_name in pairs:
        before = (run_root / before_name).read_text(encoding="utf-8").split()[0]
        after = (run_root / after_name).read_text(encoding="utf-8").split()[0]
        if before != after:
            failures.append(f"read-only history changed: {before_name} != {after_name}")

    for mode in ("learned_resume", "offline_replay"):
        run = next(item for item in runs if item["mode"] == mode)
        if not any(trace.get("loaded_from_cache") for trace in run["traces"]):
            failures.append(f"{mode}: no trace confirms compatible cache loading")
    return failures


def metric_totals(run):
    totals = defaultdict(float)
    for item in run["metrics"]:
        for key in (
            "invocation_count",
            "measured_count",
            "replay_count",
            "host_call_seconds",
            "recommendation_seconds",
            "estimated_measured_control_and_sync_seconds",
        ):
            totals[key] += item.get(key, 0.0)
    return totals


def measured_candidate_counts(run):
    counts = Counter()
    for trace in run["traces"]:
        if trace.get("measured"):
            counts[(trace["kernel"], trace["candidate_index"])] += 1
    return counts


def residual_adapter_summary(history_path: Path):
    adapters = []

    def visit(value):
        if isinstance(value, dict):
            adapter = value.get("residual_adapter")
            if isinstance(adapter, dict):
                adapters.append(adapter)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(load_json(history_path))
    return {
        "contexts": len(adapters),
        "updated_contexts": sum(item.get("update_count", 0) > 0 for item in adapters),
        "update_count": sum(item.get("update_count", 0) for item in adapters),
        "observations": sum(len(item.get("observations", [])) for item in adapters),
    }


def completion_reason_summary(history_path: Path):
    contexts = load_json(history_path).get("contexts", {}).values()
    reasons = Counter(context.get("completion_reason", "none") for context in contexts)
    retry_limits = {
        context.get("policy", {}).get("maximum_consecutive_strategy_retries")
        for context in contexts
    }
    return {
        "contexts": sum(reasons.values()),
        "retry_limit": retry_limits.pop() if len(retry_limits) == 1 else None,
        "retry_completions": reasons["maximum_consecutive_strategy_retries"],
    }


def write_tables(run_root: Path, runs):
    output_dir = run_root / "evaluations"
    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
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

    with (output_dir / "tuner-metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = (
            "mode",
            "invocation_count",
            "measured_count",
            "replay_count",
            "host_call_seconds",
            "recommendation_seconds",
            "estimated_measured_control_and_sync_seconds",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs[1:]:
            totals = metric_totals(run)
            writer.writerow(
                {"mode": run["mode"]}
                | {key: totals[key] for key in fields if key != "mode"}
            )


def plot_timings(run_root: Path, runs):
    output_dir = run_root / "evaluations"
    curves = {}
    for run in runs:
        curves[run["mode"]] = (
            [item["step"] for item in run["timings"]],
            [item["elapsed_seconds"] for item in run["timings"]],
        )

    figure, axis = plt.subplots(figsize=(10, 6))
    for mode, (steps, elapsed) in curves.items():
        axis.plot(steps, elapsed, label=mode)
    axis.set_xlabel("Completed outer steps")
    axis.set_ylabel("Cumulative application time (s)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "cumulative-application-time.png", dpi=180)
    plt.close(figure)

    baseline_steps, baseline_elapsed = curves["baseline"]
    figure, axis = plt.subplots(figsize=(10, 6))
    for mode in MODES[1:]:
        steps, elapsed = curves[mode]
        if steps != baseline_steps:
            raise RuntimeError(f"{mode}: step grid differs from baseline")
        axis.plot(
            steps,
            [current - reference for current, reference in zip(elapsed, baseline_elapsed)],
            label=mode,
        )
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xlabel("Completed outer steps")
    axis.set_ylabel("Cumulative time minus native baseline (s)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "cumulative-delta-vs-baseline.png", dpi=180)
    plt.close(figure)


def write_report(run_root: Path, runs, failures):
    baseline_seconds = runs[0]["summary"]["elapsed_seconds"]
    baseline_step_seconds = runs[0]["timings"][-1]["elapsed_seconds"]
    learned_first = next(
        run for run in runs if run["mode"] == "learned_collect"
    )["summary"]["elapsed_seconds"]
    random_first = next(
        run for run in runs if run["mode"] == "random_collect"
    )["summary"]["elapsed_seconds"]
    adapter = residual_adapter_summary(run_root / "histories" / "learned.json")
    learned_completion = completion_reason_summary(
        run_root / "histories" / "learned.json"
    )
    random_completion = completion_reason_summary(
        run_root / "histories" / "random.json"
    )
    lines = [
        "# HASEonGPU history-aware horizon comparison",
        "",
        "One continuous A100 allocation; every mode ran 1000 time steps and 1000 pump steps.",
        "The two collection runs use three measured launches per complete candidate residency.",
        "Total time includes one-time setup. Step-loop time is the recurring application region",
        "and is therefore the relevant view of whether online overhead amortizes.",
        "",
        "| Mode | Total (s) | Total delta native | Step loop (s) | Step delta native | Delta first run | Measured launches | Recommendation (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        elapsed = run["summary"]["elapsed_seconds"]
        step_elapsed = run["timings"][-1]["elapsed_seconds"]
        totals = metric_totals(run)
        if run["mode"] in ("learned_collect", "learned_resume"):
            first = learned_first
        elif run["mode"] in ("random_collect", "offline_replay"):
            first = random_first
        else:
            first = elapsed
        lines.append(
            f"| {run['mode']} | {elapsed:.6f} | "
            f"{elapsed - baseline_seconds:+.6f} s ({elapsed / baseline_seconds - 1:+.2%}) | "
            f"{step_elapsed:.6f} | "
            f"{step_elapsed - baseline_step_seconds:+.6f} s "
            f"({step_elapsed / baseline_step_seconds - 1:+.2%}) | "
            f"{elapsed - first:+.6f} s | {totals['measured_count']:.0f} | "
            f"{totals['recommendation_seconds'] * 1.0e3:.3f} |"
        )

    lines.extend(["", "## Collection sampling", ""])
    for mode in ("learned_collect", "random_collect"):
        run = next(item for item in runs if item["mode"] == mode)
        counts = measured_candidate_counts(run)
        minimum = min(counts.values()) if counts else 0
        full_residencies = sum(count // 3 for count in counts.values())
        partial_candidates = sum(count % 3 != 0 for count in counts.values())
        lines.append(
            f"- `{mode}`: {len(counts)} measured kernel/candidate pairs, "
            f"minimum observed count {minimum}, {full_residencies} complete "
            f"three-measurement residencies, {partial_candidates} pairs with a "
            "final partial residency."
        )

    lines.extend(
        [
            "",
            "## Persistence and adapter",
            "",
            "- Learned resume left the learned input history byte-for-byte unchanged.",
            "- Offline replay left the random input history byte-for-byte unchanged.",
            f"- Learned collection terminated {learned_completion['retry_completions']} "
            f"of {learned_completion['contexts']} contexts at "
            f"`maximum_consecutive_strategy_retries="
            f"{learned_completion['retry_limit']}`.",
            f"- Random collection terminated {random_completion['retry_completions']} "
            f"of {random_completion['contexts']} contexts at "
            f"`maximum_consecutive_strategy_retries="
            f"{random_completion['retry_limit']}`.",
            f"- Learned history contains {adapter['contexts']} residual-adapter states; "
            f"{adapter['updated_contexts']} were fitted, with "
            f"{adapter['update_count']} total updates and "
            f"{adapter['observations']} retained observations.",
            "",
            "## Validation",
            "",
        ]
    )
    if failures:
        lines.extend(f"- FAIL: {failure}" for failure in failures)
    else:
        lines.append(
            "All runs completed with finite, baseline-matching numerical integrals."
        )
    (run_root / "evaluations" / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = args.run_root / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.run_root)
    failures = validate(args.run_root, runs)
    write_tables(args.run_root, runs)
    plot_timings(args.run_root, runs)
    write_report(args.run_root, runs, failures)
    if failures:
        raise SystemExit("validation failed; see evaluations/report.md")


if __name__ == "__main__":
    main()
