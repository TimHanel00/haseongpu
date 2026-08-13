#!/usr/bin/env python3
"""Validate and plot a timing-only HASE alpakaTune complete history."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_KERNEL = "AccumulateForwardPhiAse"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_trace(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def short_launch(value: object) -> str:
    text = str(value)
    return text if len(text) <= 44 else f"{text[:41]}..."


def collect_history(history: dict) -> tuple[list[dict], list[str]]:
    failures: list[str] = []
    if history.get("schema_version") != 11:
        failures.append(f"complete history schema is {history.get('schema_version')}, expected 11")

    rows: list[dict] = []
    contexts = history.get("contexts", {})
    if not contexts:
        failures.append("complete history contains no contexts")
        return rows, failures

    used_labels: Counter[str] = Counter()
    for fingerprint, context in sorted(contexts.items()):
        metadata = context.get("metadata", {})
        kernel = str(metadata.get("kernel", ""))
        if not kernel.endswith(EXPECTED_KERNEL):
            failures.append(f"{fingerprint[:12]}: unexpected kernel {kernel!r}")
        if metadata.get("mode") != "online_adaptive":
            failures.append(f"{fingerprint[:12]}: mode is not online_adaptive")
        if context.get("candidate_count") != 1:
            failures.append(f"{fingerprint[:12]}: candidate_count is not one")
        if context.get("policy", {}).get("queue") is not None:
            failures.append(f"{fingerprint[:12]}: active queue metadata is present")

        candidate_samples = context.get("candidate_samples", [])
        configurations = context.get("candidate_configurations", [])
        if len(candidate_samples) != 1 or not candidate_samples[0]:
            failures.append(f"{fingerprint[:12]}: expected one non-empty candidate history")
            continue

        launch = metadata.get("launch_specification", "unknown launch")
        base_label = short_launch(launch)
        used_labels[base_label] += 1
        label = base_label if used_labels[base_label] == 1 else f"{base_label} [{fingerprint[:8]}]"
        configuration = configurations[0] if configurations else {}
        configuration_text = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
        started = context.get("started_at_unix_seconds", "")
        for sample, runtime_seconds in enumerate(candidate_samples[0], start=1):
            if not isinstance(runtime_seconds, (int, float)) or not math.isfinite(runtime_seconds) or runtime_seconds < 0:
                failures.append(f"{fingerprint[:12]}: invalid runtime at sample {sample}")
                continue
            rows.append(
                {
                    "Sample": sample,
                    "Context": label,
                    "Kernel": EXPECTED_KERNEL,
                    "Device": metadata.get("device", "unknown"),
                    "LaunchSpecification": launch,
                    "CandidateConfiguration": configuration_text,
                    "StartedUnixSeconds": started,
                    "DurationNs": runtime_seconds * 1.0e9,
                    "RuntimeSource": "device_event",
                    "Fingerprint": fingerprint,
                }
            )
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
        failures.append("trace contains a tuning space larger than one candidate")
    if any(entry.get("active_queue") is not False for entry in trace):
        failures.append("trace contains an active candidate queue")

    trace_samples = sorted(
        float(entry["runtime_seconds"])
        for entry in trace
        if isinstance(entry.get("runtime_seconds"), (int, float))
    )
    history_samples = sorted(float(row["DurationNs"]) * 1.0e-9 for row in rows)
    if len(trace_samples) == len(history_samples) and not np.allclose(
        trace_samples, history_samples, rtol=1.0e-12, atol=1.0e-15
    ):
        failures.append("trace device durations do not match complete-history raw samples")
    return failures


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = [
        "Sample",
        "Context",
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


def rolling_median(values: np.ndarray, maximum_window: int = 101) -> tuple[np.ndarray, np.ndarray]:
    if values.size < 3:
        indexes = np.arange(1, values.size + 1)
        return indexes, values
    window = min(maximum_window, max(3, values.size // 30))
    if window % 2 == 0:
        window += 1
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    indexes = np.arange(window // 2 + 1, window // 2 + 1 + windows.shape[0])
    return indexes, np.median(windows, axis=1)


def plot(path: Path, rows: list[dict], title: str) -> None:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["Context"]), []).append(float(row["DurationNs"]) * 1.0e-6)
    ordered = sorted(grouped, key=lambda label: statistics.median(grouped[label]))

    figure, (timeline, distribution) = plt.subplots(
        1,
        2,
        figsize=(15.5, 7.5),
        gridspec_kw={"width_ratios": (2.25, 1.0)},
        constrained_layout=True,
    )
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, max(1, len(ordered))))
    for color, label in zip(colors, ordered):
        values = np.asarray(grouped[label], dtype=float)
        stride = max(1, math.ceil(values.size / 4000))
        sample = np.arange(1, values.size + 1)
        timeline.scatter(sample[::stride], values[::stride], s=3, alpha=0.12, color=color, rasterized=True)
        median_x, median_y = rolling_median(values)
        timeline.plot(median_x, median_y, linewidth=1.6, color=color, label=f"{label} (n={values.size:,})")

    timeline.set_title("Natural-launch device timings")
    timeline.set_xlabel("Sample within launch context")
    timeline.set_ylabel("Alpaka event duration [ms]")
    timeline.grid(alpha=0.22)
    timeline.legend(fontsize=8, frameon=False)

    box_values = [np.asarray(grouped[label][10:] or grouped[label], dtype=float) for label in ordered]
    boxes = distribution.boxplot(
        box_values,
        vert=False,
        tick_labels=ordered,
        showfliers=False,
        patch_artist=True,
        widths=0.65,
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    distribution.set_title("Steady-state distribution")
    distribution.set_xlabel("Alpaka event duration [ms]")
    distribution.grid(axis="x", alpha=0.22)
    figure.suptitle(title, fontsize=15)
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def summarize(rows: list[dict]) -> dict:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["Context"]), []).append(float(row["DurationNs"]))
    contexts = {}
    for label, values in sorted(grouped.items()):
        array = np.asarray(values, dtype=float)
        steady = array[10:] if array.size > 10 else array
        contexts[label] = {
            "samples": int(array.size),
            "median_ns": float(np.median(steady)),
            "p05_ns": float(np.percentile(steady, 5)),
            "p95_ns": float(np.percentile(steady, 95)),
            "mean_ns": float(np.mean(steady)),
        }
    return {"kernel": EXPECTED_KERNEL, "samples": len(rows), "contexts": contexts}


def write_report(path: Path, summary: dict) -> None:
    lines = [
        "# HASE timing-only full-history benchmark",
        "",
        f"Validated **{summary['samples']:,}** Alpaka device-event samples for `{EXPECTED_KERNEL}`.",
        "The tuning space contained only the application's original `FrameSpec`, with no active candidate queue.",
        "",
        "## Contexts",
        "",
    ]
    for label, values in summary["contexts"].items():
        lines.append(
            f"- `{label}`: {values['samples']:,} samples, median {values['median_ns'] / 1.0e6:.6f} ms, "
            f"5th--95th percentile {values['p05_ns'] / 1.0e6:.6f}--{values['p95_ns'] / 1.0e6:.6f} ms."
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
    history_path = raw / "full-history.json"
    trace_path = raw / "trace.jsonl"

    rows, failures = collect_history(load_json(history_path))
    trace = load_trace(trace_path)
    failures.extend(validate_trace(rows, trace))
    if failures:
        raise SystemExit("\n".join(f"validation failed: {failure}" for failure in failures))

    summary = summarize(rows)
    write_csv(output / "timing-history.csv", rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(output / "report.md", summary)
    plot(
        output / "timing-history.png",
        rows,
        f"HASEonGPU {EXPECTED_KERNEL}: timing-only full history",
    )
    print(f"wrote {len(rows):,} rows to {output / 'timing-history.csv'}")


if __name__ == "__main__":
    main()
