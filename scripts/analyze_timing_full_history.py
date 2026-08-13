#!/usr/bin/env python3
"""Validate and plot an all-kernel, timing-only HASE full history."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def short_kernel(type_name: object) -> str:
    return str(type_name).rsplit("::", maxsplit=1)[-1]


def collect_history(history: dict) -> tuple[list[dict], list[str]]:
    failures: list[str] = []
    if history.get("schema_version") != 11:
        failures.append(f"complete history schema is {history.get('schema_version')}, expected 11")

    contexts = history.get("contexts", {})
    if not contexts:
        return [], [*failures, "complete history contains no contexts"]

    pending: list[dict] = []
    for fingerprint, context in sorted(contexts.items()):
        metadata = context.get("metadata", {})
        kernel = short_kernel(metadata.get("kernel", ""))
        if not kernel:
            failures.append(f"{fingerprint[:12]}: missing kernel identity")
        if metadata.get("mode") != "online_adaptive":
            failures.append(f"{fingerprint[:12]}: mode is not online_adaptive")
        if context.get("candidate_count") != 1:
            failures.append(f"{fingerprint[:12]}: candidate_count is not one")
        if context.get("policy", {}).get("queue") is not None:
            failures.append(f"{fingerprint[:12]}: active queue metadata is present")

        configurations = context.get("candidate_configurations", [])
        dimensions = metadata.get("model_context", {}).get("dimensions", [])
        if configurations != [{}]:
            failures.append(f"{fingerprint[:12]}: candidate configuration is not empty")
        if dimensions:
            failures.append(f"{fingerprint[:12]}: tunable dimensions are present")

        candidate_samples = context.get("candidate_samples", [])
        if len(candidate_samples) != 1 or not candidate_samples[0]:
            failures.append(f"{fingerprint[:12]}: expected one non-empty timing history")
            continue

        launch = metadata.get("launch_specification", "unknown")
        started = context.get("started_at_unix_seconds", "")
        for context_sample, runtime_seconds in enumerate(candidate_samples[0], start=1):
            if not isinstance(runtime_seconds, (int, float)) or not math.isfinite(runtime_seconds) or runtime_seconds < 0:
                failures.append(f"{fingerprint[:12]}: invalid runtime at sample {context_sample}")
                continue
            pending.append(
                {
                    "Kernel": kernel,
                    "ContextSample": context_sample,
                    "Device": metadata.get("device", "unknown"),
                    "LaunchSpecification": launch,
                    "CandidateConfiguration": "{}",
                    "StartedUnixSeconds": started,
                    "DurationNs": runtime_seconds * 1.0e9,
                    "RuntimeSource": "device_event",
                    "Fingerprint": fingerprint,
                }
            )

    kernel_samples: Counter[str] = Counter()
    rows: list[dict] = []
    for row in pending:
        kernel_samples[row["Kernel"]] += 1
        rows.append({"Sample": kernel_samples[row["Kernel"]], **row})
    return rows, failures


def validate_trace(rows: list[dict], trace: list[dict]) -> list[str]:
    failures: list[str] = []
    if len(trace) != len(rows):
        failures.append(f"trace has {len(trace)} samples but complete history has {len(rows)}")
    sources = {entry.get("runtime_measurement_source") for entry in trace}
    if sources != {"device_event"}:
        failures.append(f"runtime measurement sources are {sorted(map(str, sources))}")
    if any(not entry.get("measured") for entry in trace):
        failures.append("trace contains an unmeasured launch")
    if any(entry.get("candidate_count") != 1 for entry in trace):
        failures.append("trace contains more than one candidate")
    if any(entry.get("tunable_parameter_count") != 0 for entry in trace):
        failures.append("trace contains a tunable parameter")
    if any(entry.get("active_queue") is not False for entry in trace):
        failures.append("trace contains an active candidate queue")

    history_by_kernel: dict[str, list[float]] = defaultdict(list)
    trace_by_kernel: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        history_by_kernel[str(row["Kernel"])].append(float(row["DurationNs"]) * 1.0e-9)
    for entry in trace:
        if isinstance(entry.get("runtime_seconds"), (int, float)):
            trace_by_kernel[str(entry.get("kernel", ""))].append(float(entry["runtime_seconds"]))

    if set(history_by_kernel) != set(trace_by_kernel):
        failures.append("trace and full history contain different kernel identities")
    for kernel in sorted(set(history_by_kernel) & set(trace_by_kernel)):
        history_values = np.sort(np.asarray(history_by_kernel[kernel]))
        trace_values = np.sort(np.asarray(trace_by_kernel[kernel]))
        if history_values.size != trace_values.size or not np.allclose(
            history_values, trace_values, rtol=1.0e-12, atol=1.0e-15
        ):
            failures.append(f"{kernel}: trace durations do not match complete history")
    return failures


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = [
        "Sample",
        "ContextSample",
        "Kernel",
        "Device",
        "LaunchSpecification",
        "CandidateConfiguration",
        "StartedUnixSeconds",
        "DurationNs",
        "RuntimeSource",
        "Fingerprint",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    grouped: dict[str, list[float]] = defaultdict(list)
    launches: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grouped[str(row["Kernel"])].append(float(row["DurationNs"]))
        launches[str(row["Kernel"])].add(str(row["LaunchSpecification"]))

    kernels = {}
    for kernel, values in sorted(grouped.items()):
        array = np.asarray(values, dtype=float)
        kernels[kernel] = {
            "samples": int(array.size),
            "median_ns": float(np.median(array)),
            "p05_ns": float(np.percentile(array, 5)),
            "p95_ns": float(np.percentile(array, 95)),
            "mean_ns": float(np.mean(array)),
            "total_ns": float(np.sum(array)),
            "natural_launch_specifications": sorted(launches[kernel]),
        }
    return {"samples": len(rows), "kernel_count": len(kernels), "kernels": kernels}


def plot(path: Path, rows: list[dict], summary: dict) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["Kernel"])].append(float(row["DurationNs"]) * 1.0e-6)
    ordered = sorted(grouped, key=lambda kernel: summary["kernels"][kernel]["median_ns"])
    positions = np.arange(len(ordered))
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, max(1, len(ordered))))

    figure = plt.figure(figsize=(17, max(8.5, 0.58 * len(ordered) + 3.0)), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.65, 1.0))
    distribution = figure.add_subplot(grid[:, 0])
    total_axis = figure.add_subplot(grid[0, 1])
    count_axis = figure.add_subplot(grid[1, 1])

    values = [np.asarray(grouped[kernel], dtype=float) for kernel in ordered]
    boxes = distribution.boxplot(values, vert=False, positions=positions, showfliers=False, patch_artist=True)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    distribution.set_yticks(positions, ordered)
    distribution.set_xscale("log")
    distribution.set_xlabel("Alpaka event duration per call [ms, log scale]")
    distribution.set_title("Per-call device timing distribution")
    distribution.grid(axis="x", alpha=0.24, which="both")

    totals = [summary["kernels"][kernel]["total_ns"] * 1.0e-9 for kernel in ordered]
    counts = [summary["kernels"][kernel]["samples"] for kernel in ordered]
    total_axis.barh(positions, totals, color=colors, alpha=0.85)
    total_axis.set_yticks(positions, ordered)
    total_axis.set_xscale("log")
    total_axis.set_xlabel("Accumulated measured device time [s, log scale]")
    total_axis.set_title("Total device-time contribution")
    total_axis.grid(axis="x", alpha=0.24, which="both")

    count_axis.barh(positions, counts, color=colors, alpha=0.85)
    count_axis.set_yticks(positions, ordered)
    count_axis.set_xscale("log")
    count_axis.set_xlabel("Measured calls [log scale]")
    count_axis.set_title("Kernel call count")
    count_axis.grid(axis="x", alpha=0.24, which="both")

    figure.suptitle("HASEonGPU all-kernel benchmark: unchanged default FrameSpecs", fontsize=16)
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def write_report(path: Path, summary: dict) -> None:
    lines = [
        "# HASE all-kernel timing-only benchmark",
        "",
        f"Validated **{summary['samples']:,}** Alpaka device-event samples across "
        f"**{summary['kernel_count']} kernels**.",
        "Every kernel used its unchanged application-provided `FrameSpec`; the tuning bundle was empty.",
        "",
        "## Kernels",
        "",
    ]
    ordered = sorted(summary["kernels"], key=lambda name: summary["kernels"][name]["total_ns"], reverse=True)
    for kernel in ordered:
        values = summary["kernels"][kernel]
        lines.append(
            f"- `{kernel}`: {values['samples']:,} calls, median {values['median_ns'] / 1.0e6:.6f} ms, "
            f"5th--95th percentile {values['p05_ns'] / 1.0e6:.6f}--{values['p95_ns'] / 1.0e6:.6f} ms, "
            f"total {values['total_ns'] / 1.0e9:.6f} s."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    raw = args.run_root / "raw" / "Cuda_NvidiaGpu_GpuCuda" / "timing_only"
    output = args.output_dir or args.run_root / "evaluations"
    output.mkdir(parents=True, exist_ok=True)

    rows, failures = collect_history(load_json(raw / "full-history.json"))
    failures.extend(validate_trace(rows, load_trace(raw / "trace.jsonl")))
    if failures:
        raise SystemExit("\n".join(f"validation failed: {failure}" for failure in failures))

    summary = summarize(rows)
    write_csv(output / "all-kernel-timing.csv", rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(output / "report.md", summary)
    plot(output / "all-kernel-timing.png", rows, summary)
    print(
        f"wrote {len(rows):,} rows across {summary['kernel_count']} kernels to "
        f"{output / 'all-kernel-timing.csv'}"
    )


if __name__ == "__main__":
    main()
