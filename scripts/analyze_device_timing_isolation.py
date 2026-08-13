#!/usr/bin/env python3
"""Validate and plot tuner-linked versus tuner-free device timing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BACKEND = "Cuda_NvidiaGpu_GpuCuda"
KERNELS = ("AccumulateForwardPhiAse", "TraceGeneralPump")
MODES = ("current_timing_default", "no_tuner_default", "no_tuner_selected")
MODE_LABELS = {
    "current_timing_default": "Current timing\n(tuner linked)",
    "no_tuner_default": "Isolated default\n(no tuner)",
    "no_tuner_selected": "Hand selected\n(no tuner)",
}
MODE_COLORS = {
    "current_timing_default": "#6c757d",
    "no_tuner_default": "#457b9d",
    "no_tuner_selected": "#2a9d8f",
}
EXPECTED_COUNTS = {
    ("AccumulateForwardPhiAse", 2, 512): 16_000,
    ("AccumulateForwardPhiAse", 5, 512): 16_000,
    ("AccumulateForwardPhiAse", 16, 512): 16_000,
    ("AccumulateForwardPhiAse", 52, 512): 16_000,
    ("AccumulateForwardPhiAse", 166, 512): 10_496,
    ("TraceGeneralPump", 97, 512): 8_004,
}
SELECTED_SHAPES = {
    ("AccumulateForwardPhiAse", 2, 512): (16, 64),
    ("AccumulateForwardPhiAse", 5, 512): (80, 32),
    ("AccumulateForwardPhiAse", 16, 512): (256, 32),
    ("AccumulateForwardPhiAse", 52, 512): (208, 128),
    ("AccumulateForwardPhiAse", 166, 512): (2656, 32),
    ("TraceGeneralPump", 97, 512): (194, 256),
}


def load_tuner_trace(path: Path) -> list[dict]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        result.append(
            {
                "kernel": source["kernel"],
                "mode": "current_timing_default",
                "original_num_frames": int(source["original_num_frames"][0]),
                "original_frame_extent": int(source["original_frame_extent"][0]),
                "selected_num_frames": int(source["selected_num_frames"][0]),
                "selected_frame_extent": int(source["selected_frame_extent"][0]),
                "duration_ns": float(source["evaluation_runtime_seconds"]) * 1.0e9,
                "runtime_source": source["evaluation_runtime_measurement_source"],
                "candidate_count": int(source["candidate_count"]),
                "measured": bool(source["evaluation_measured"]),
            }
        )
    return result


def load_direct_csv(path: Path) -> list[dict]:
    result = []
    with path.open(newline="", encoding="utf-8") as stream:
        for source in csv.DictReader(stream):
            result.append(
                {
                    "kernel": source["Kernel"],
                    "mode": source["Mode"],
                    "original_num_frames": int(source["OriginalNumFrames"]),
                    "original_frame_extent": int(source["OriginalFrameExtent"]),
                    "selected_num_frames": int(source["SelectedNumFrames"]),
                    "selected_frame_extent": int(source["SelectedFrameExtent"]),
                    "duration_ns": float(source["DurationNs"]),
                    "runtime_source": source["RuntimeSource"],
                    "candidate_count": 1,
                    "measured": True,
                }
            )
    return result


def context_key(record: dict) -> tuple[str, int, int]:
    return record["kernel"], record["original_num_frames"], record["original_frame_extent"]


def validate(records_by_mode: dict[str, list[dict]]) -> list[str]:
    failures: list[str] = []
    for mode, records in records_by_mode.items():
        if {record["mode"] for record in records} != {mode}:
            failures.append(f"{mode}: inconsistent mode labels")
        if {record["kernel"] for record in records} != set(KERNELS):
            failures.append(f"{mode}: unexpected kernel set")
        if any(record["runtime_source"] != "device_event" for record in records):
            failures.append(f"{mode}: a duration is not device-event timing")
        if any(
            not record["measured"] or not math.isfinite(record["duration_ns"]) or record["duration_ns"] <= 0.0
            for record in records
        ):
            failures.append(f"{mode}: invalid measured duration")
        counts = Counter(context_key(record) for record in records)
        if counts != Counter(EXPECTED_COUNTS):
            failures.append(f"{mode}: context counts {dict(counts)} differ from {EXPECTED_COUNTS}")
        if any(record["candidate_count"] != 1 for record in records):
            failures.append(f"{mode}: a tuning space was active")
        for record in records:
            original = (record["original_num_frames"], record["original_frame_extent"])
            selected = (record["selected_num_frames"], record["selected_frame_extent"])
            if mode == "no_tuner_selected":
                expected = SELECTED_SHAPES.get(context_key(record))
                if selected != expected:
                    failures.append(f"{mode}: {context_key(record)} selected {selected}, expected {expected}")
                    break
            elif selected != original:
                failures.append(f"{mode}: default FrameSpec changed from {original} to {selected}")
                break
            if selected[0] * selected[1] != original[0] * original[1]:
                failures.append(f"{mode}: worker coverage changed for {context_key(record)}")
                break
    return failures


def parse_elapsed_seconds(path: Path) -> float:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([^\s]+)", text)
    if match is None:
        raise ValueError(f"could not parse wall time from {path}")
    parts = [float(part) for part in match.group(1).split(":")]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    raise ValueError(f"unexpected elapsed-time spelling in {path}")


def summarize(records_by_mode: dict[str, list[dict]], wall_seconds: dict[str, float]) -> dict:
    kernels: dict[str, dict] = {}
    for kernel in KERNELS:
        kernels[kernel] = {}
        for mode in MODES:
            values = np.asarray(
                [record["duration_ns"] * 1.0e-6 for record in records_by_mode[mode] if record["kernel"] == kernel]
            )
            kernels[kernel][mode] = {
                "samples": int(values.size),
                "median_ms": float(np.median(values)),
                "mean_ms": float(np.mean(values)),
                "p05_ms": float(np.percentile(values, 5)),
                "p95_ms": float(np.percentile(values, 95)),
            }
        current = kernels[kernel]["current_timing_default"]["median_ms"]
        isolated = kernels[kernel]["no_tuner_default"]["median_ms"]
        selected = kernels[kernel]["no_tuner_selected"]["median_ms"]
        kernels[kernel]["comparisons"] = {
            "isolated_over_current": isolated / current,
            "selected_speedup_over_isolated": isolated / selected,
            "selected_speedup_over_current": current / selected,
        }
    return {
        "validation_failures": [],
        "kernels": kernels,
        "wall_seconds": wall_seconds,
        "wall_comparisons": {
            "isolated_over_current": wall_seconds["no_tuner_default"] / wall_seconds["current_timing_default"],
            "selected_speedup_over_isolated": wall_seconds["no_tuner_default"] / wall_seconds["no_tuner_selected"],
        },
    }


def write_combined_csv(path: Path, records_by_mode: dict[str, list[dict]]) -> None:
    columns = [
        "Sample",
        "Kernel",
        "Mode",
        "KernelMode",
        "OriginalNumFrames",
        "OriginalFrameExtent",
        "SelectedNumFrames",
        "SelectedFrameExtent",
        "TunerMeasured",
        "DurationNs",
        "RuntimeSource",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for mode in MODES:
            samples: Counter[str] = Counter()
            for record in records_by_mode[mode]:
                samples[record["kernel"]] += 1
                writer.writerow(
                    {
                        "Sample": samples[record["kernel"]],
                        "Kernel": record["kernel"],
                        "Mode": mode,
                        "KernelMode": f'{record["kernel"]} | {mode}',
                        "OriginalNumFrames": record["original_num_frames"],
                        "OriginalFrameExtent": record["original_frame_extent"],
                        "SelectedNumFrames": record["selected_num_frames"],
                        "SelectedFrameExtent": record["selected_frame_extent"],
                        "TunerMeasured": "false",
                        "DurationNs": record["duration_ns"],
                        "RuntimeSource": "device_event",
                    }
                )


def plot_device(path: Path, records_by_mode: dict[str, list[dict]], summary: dict) -> None:
    arrays, labels, colors, positions = [], [], [], []
    position = 1.0
    for kernel_index, kernel in enumerate(KERNELS):
        for mode in MODES:
            arrays.append(
                np.asarray(
                    [
                        record["duration_ns"] * 1.0e-6
                        for record in records_by_mode[mode]
                        if record["kernel"] == kernel
                    ]
                )
            )
            labels.append(f"{kernel}\n{MODE_LABELS[mode]}")
            colors.append(MODE_COLORS[mode])
            positions.append(position)
            position += 1.0
        if kernel_index + 1 < len(KERNELS):
            position += 0.8
    figure, axis = plt.subplots(figsize=(15, 8.5), constrained_layout=True)
    boxes = axis.boxplot(
        arrays,
        positions=positions,
        widths=0.68,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#111111", "linewidth": 2.0},
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.86)
    axis.set_xticks(positions, labels, rotation=17, ha="right")
    axis.set_ylabel("Device-event runtime per kernel call [ms]")
    axis.set_xlabel("Kernel + instrumentation/FrameSpec mode")
    axis.set_title("HASEonGPU A100: tuner overhead isolation and static FrameSpec replay")
    axis.grid(axis="y", alpha=0.25)
    for x, values in zip(positions, [summary["kernels"][kernel][mode] for kernel in KERNELS for mode in MODES]):
        axis.annotate(
            f'{values["median_ms"]:.3f}',
            (x, values["median_ms"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    figure.savefig(path, dpi=190)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def plot_wall(path: Path, wall_seconds: dict[str, float]) -> None:
    figure, axis = plt.subplots(figsize=(9.5, 6.2), constrained_layout=True)
    values = [wall_seconds[mode] for mode in MODES]
    bars = axis.bar([MODE_LABELS[mode] for mode in MODES], values, color=[MODE_COLORS[mode] for mode in MODES])
    axis.bar_label(bars, fmt="%.2f s", padding=4)
    axis.set_ylabel("Whole-application wall time [s]")
    axis.set_title("Host-visible runtime, including instrumentation bookkeeping")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=190)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def write_report(path: Path, summary: dict) -> None:
    lines = [
        "# Device-timing isolation campaign",
        "",
        "All three modes ran sequentially in one Slurm allocation on the same GPU. Durations are Alpaka device-event timings; wall time additionally includes host-side instrumentation work.",
        "",
        "## Kernel medians",
        "",
    ]
    for kernel in KERNELS:
        lines.extend([f"### `{kernel}`", ""])
        for mode in MODES:
            values = summary["kernels"][kernel][mode]
            lines.append(
                f'- {MODE_LABELS[mode].replace(chr(10), " ")}: {values["median_ms"]:.6f} ms median, '
                f'5th--95th percentile {values["p05_ms"]:.6f}--{values["p95_ms"]:.6f} ms, '
                f'{values["samples"]:,} measurements.'
            )
        comparisons = summary["kernels"][kernel]["comparisons"]
        lines.append(
            f'- Isolated/current default median ratio: {comparisons["isolated_over_current"]:.6f}.'
        )
        lines.append(
            f'- Hand-selected speedup over isolated default: {comparisons["selected_speedup_over_isolated"]:.6f}x.'
        )
        lines.append("")
    lines.extend(["## Whole-application wall time", ""])
    for mode in MODES:
        lines.append(f'- {MODE_LABELS[mode].replace(chr(10), " ")}: {summary["wall_seconds"][mode]:.2f} s.')
    lines.extend(["", "## Validation", "", "- All context, count, coverage, FrameSpec, and device-event checks passed."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    raw = args.run_root / "raw" / BACKEND
    records_by_mode = {
        "current_timing_default": load_tuner_trace(raw / "current_timing_default" / "trace.jsonl"),
        "no_tuner_default": load_direct_csv(raw / "no_tuner_default" / "timings.csv"),
        "no_tuner_selected": load_direct_csv(raw / "no_tuner_selected" / "timings.csv"),
    }
    failures = validate(records_by_mode)
    wall_seconds = {mode: parse_elapsed_seconds(raw / mode / "time.txt") for mode in MODES}
    summary = summarize(records_by_mode, wall_seconds)
    summary["validation_failures"] = failures
    output = args.run_root / "evaluations"
    output.mkdir(parents=True, exist_ok=True)
    write_combined_csv(output / "device-timing-isolation.csv", records_by_mode)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(output / "report.md", summary)
    plot_device(output / "device-timing-isolation.png", records_by_mode, summary)
    plot_wall(output / "wall-time-isolation.png", wall_seconds)
    if failures:
        raise SystemExit("validation failed: " + "; ".join(failures))
    print(json.dumps({"validation_failures": failures, "wall_seconds": wall_seconds}, sort_keys=True))


if __name__ == "__main__":
    main()
