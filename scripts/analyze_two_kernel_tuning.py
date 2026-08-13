#!/usr/bin/env python3
"""Validate and plot equal-count two-kernel tuning campaigns."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BACKEND = "Cuda_NvidiaGpu_GpuCuda"
KERNELS = ("AccumulateForwardPhiAse", "TraceGeneralPump")
RAW_MODES = ("offline", "online_fixed", "online_adaptive")
PLOT_MODES = ("baseline", "offline", "online_fixed", "online_adaptive")
MODE_LABELS = {
    "baseline": "Baseline",
    "offline": "Offline",
    "online_fixed": "Online fixed",
    "online_adaptive": "Online adaptive",
}
MODE_COLORS = {
    "baseline": "#5c677d",
    "offline": "#2a9d8f",
    "online_fixed": "#e9c46a",
    "online_adaptive": "#e76f51",
}
EXPECTED_CONTEXT_SAMPLES = {
    ("AccumulateForwardPhiAse", (2,), (512,)): 15_992,
    ("AccumulateForwardPhiAse", (5,), (512,)): 15_992,
    ("AccumulateForwardPhiAse", (16,), (512,)): 15_992,
    ("AccumulateForwardPhiAse", (52,), (512,)): 15_992,
    ("AccumulateForwardPhiAse", (166,), (512,)): 10_496,
    ("TraceGeneralPump", (97,), (512,)): 8_000,
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_baseline_reference(path: Path) -> list[dict]:
    result: list[dict] = []
    for source in load_jsonl(path):
        if source.get("kernel") not in KERNELS:
            continue
        frames = source["num_frames"]
        extent = source["frame_extent"]
        coverage = frames[0] * extent[0]
        result.append(
            {
                "mode": "baseline",
                "kernel": source["kernel"],
                "evaluation_runtime_measurement_source": source["runtime_measurement_source"],
                "evaluation_runtime_seconds": source["runtime_seconds"],
                "evaluation_measured": source["measured"],
                "coverage": coverage,
                "selected_coverage": coverage,
                "original_num_frames": frames,
                "original_frame_extent": extent,
                "selected_num_frames": frames,
                "selected_frame_extent": extent,
                "candidate_count": source["candidate_count"],
                "legal_candidate_count": 1,
                "tuner_measured": False,
                "loaded_from_cache": False,
                "learned_status": None,
                "tuning_complete": False,
            }
        )
    return result


def context_key(record: dict) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    return (
        str(record["kernel"]),
        tuple(int(value) for value in record["original_num_frames"]),
        tuple(int(value) for value in record["original_frame_extent"]),
    )


def validate_mode(mode: str, records: list[dict]) -> list[str]:
    failures: list[str] = []
    if not records:
        return [f"{mode}: empty trace"]
    if {record.get("mode") for record in records} != {mode}:
        failures.append(f"{mode}: trace mode label differs")
    if {record.get("kernel") for record in records} != set(KERNELS):
        failures.append(f"{mode}: unexpected kernel set")
    if any(record.get("evaluation_runtime_measurement_source") != "device_event" for record in records):
        failures.append(f"{mode}: non-device evaluation timing present")
    measured_records = [record for record in records if record.get("evaluation_measured")]
    if any(
        not isinstance(record.get("evaluation_runtime_seconds"), (int, float))
        or not math.isfinite(record["evaluation_runtime_seconds"])
        or record["evaluation_runtime_seconds"] <= 0.0
        for record in measured_records
    ):
        failures.append(f"{mode}: invalid evaluation runtime")
    if any(record.get("selected_coverage") != record.get("coverage") for record in records):
        failures.append(f"{mode}: a selected FrameSpec changed worker coverage")

    if mode == "baseline":
        if len(measured_records) != len(records):
            failures.append(f"{mode}: baseline contains an unmeasured evaluation launch")
        if any(record.get("candidate_count") != 1 for record in records):
            failures.append(f"{mode}: baseline contains a tuning space")
        if any(record.get("selected_frame_extent") != record.get("original_frame_extent") for record in records):
            failures.append(f"{mode}: baseline changed frame extent")
        if any(record.get("selected_num_frames") != record.get("original_num_frames") for record in records):
            failures.append(f"{mode}: baseline changed frame count")
        if any(record.get("tuner_measured") for record in records):
            failures.append(f"{mode}: baseline unexpectedly entered the tuner")
    else:
        if any(record.get("candidate_count") != 25 for record in records):
            failures.append(f"{mode}: raw candidate count is not 25")
        if any(record.get("legal_candidate_count") != 5 for record in records):
            failures.append(f"{mode}: legal candidate count is not five")
        if any(record["selected_frame_extent"][0] not in (32, 64, 128, 256, 512) for record in records):
            failures.append(f"{mode}: selected extent is outside the declared space")

    if mode == "offline":
        if any(record.get("tuner_measured") for record in records):
            failures.append("offline: replay unexpectedly performed tuner measurements")
        if any(not record.get("loaded_from_cache") for record in records):
            failures.append("offline: a context did not load the exhaustive history")
        for key, group in group_contexts(records).items():
            if sum(not record.get("evaluation_measured") for record in group) != 1:
                failures.append(f"offline: {key} did not have exactly one cache-bootstrap launch")
    if mode in ("online_fixed", "online_adaptive"):
        statuses = {record.get("learned_status") for record in records}
        if statuses != {"active"}:
            failures.append(f"{mode}: learned strategy status is {sorted(map(str, statuses))}")
        if any(not record.get("loaded_from_cache") for record in records):
            failures.append(f"{mode}: a context did not load its seed history")
    if mode == "online_fixed":
        if not any(record.get("tuner_measured") for record in records):
            failures.append("online_fixed: no tuning measurements were taken")
        for key, group in group_contexts(records).items():
            measured = sum(bool(record.get("tuner_measured")) for record in group)
            if measured != 45:
                failures.append(f"online_fixed: {key} took {measured} tuner measurements instead of 45")
            if not any(record.get("tuning_complete") for record in group):
                failures.append(f"online_fixed: {key} never completed")
            if sum(not record.get("evaluation_measured") for record in group) != 1:
                failures.append(f"online_fixed: {key} did not have exactly one winner-cache launch")
    if mode == "online_adaptive":
        if any(not record.get("tuner_measured") for record in records):
            failures.append("online_adaptive: not every application call adapted and measured")
        if len(measured_records) != len(records):
            failures.append("online_adaptive: an evaluation runtime is missing")
    return failures


def group_contexts(records: list[dict]) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        grouped[context_key(record)].append(record)
    return dict(grouped)


def equal_count_records(records: list[dict], drop_warmup: bool) -> list[dict]:
    contexts = group_contexts([record for record in records if record.get("evaluation_measured")])
    if set(contexts) != set(EXPECTED_CONTEXT_SAMPLES):
        raise ValueError("trace does not contain the six expected launch contexts")
    result: list[dict] = []
    for key in sorted(contexts):
        values = contexts[key]
        required = EXPECTED_CONTEXT_SAMPLES[key]
        first = 1 if drop_warmup else 0
        needed = required + first
        if len(values) < needed:
            raise ValueError(f"{key} has {len(values)} measured calls; need at least {needed}")
        # Tuning modes run one extra application step, so drop one measured
        # warm-up and retain the prior campaign's exact context count.
        result.extend(values[first:needed])
    return result


def summarize(records_by_mode: dict[str, list[dict]]) -> dict:
    result: dict[str, dict] = {}
    for kernel in KERNELS:
        result[kernel] = {}
        for mode in PLOT_MODES:
            records = [record for record in records_by_mode[mode] if record["kernel"] == kernel]
            values = np.asarray([record["evaluation_runtime_seconds"] * 1.0e3 for record in records])
            selections = Counter(
                (record["selected_num_frames"][0], record["selected_frame_extent"][0]) for record in records
            )
            result[kernel][mode] = {
                "samples": int(values.size),
                "median_ms": float(np.median(values)),
                "mean_ms": float(np.mean(values)),
                "p05_ms": float(np.percentile(values, 5)),
                "p95_ms": float(np.percentile(values, 95)),
                "selections": [
                    {"num_frames": frames, "frame_extent": extent, "calls": calls}
                    for (frames, extent), calls in selections.most_common()
                ],
            }
        baseline = result[kernel]["baseline"]["median_ms"]
        for mode in PLOT_MODES[1:]:
            result[kernel][mode]["median_speedup"] = baseline / result[kernel][mode]["median_ms"]
            result[kernel][mode]["beats_baseline_median"] = result[kernel][mode]["median_ms"] < baseline
    return result


def write_csv(path: Path, records_by_mode: dict[str, list[dict]]) -> None:
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
        for mode in PLOT_MODES:
            counts: Counter[str] = Counter()
            for record in records_by_mode[mode]:
                kernel = record["kernel"]
                counts[kernel] += 1
                writer.writerow(
                    {
                        "Sample": counts[kernel],
                        "Kernel": kernel,
                        "Mode": mode,
                        "KernelMode": f"{kernel} | {mode}",
                        "OriginalNumFrames": record["original_num_frames"][0],
                        "OriginalFrameExtent": record["original_frame_extent"][0],
                        "SelectedNumFrames": record["selected_num_frames"][0],
                        "SelectedFrameExtent": record["selected_frame_extent"][0],
                        "TunerMeasured": str(bool(record.get("tuner_measured"))).lower(),
                        "DurationNs": record["evaluation_runtime_seconds"] * 1.0e9,
                        "RuntimeSource": "device_event",
                    }
                )


def plot(path: Path, records_by_mode: dict[str, list[dict]], summary: dict) -> None:
    arrays: list[np.ndarray] = []
    labels: list[str] = []
    colors: list[str] = []
    positions: list[float] = []
    position = 1.0
    for kernel_index, kernel in enumerate(KERNELS):
        for mode in PLOT_MODES:
            values = [
                record["evaluation_runtime_seconds"] * 1.0e3
                for record in records_by_mode[mode]
                if record["kernel"] == kernel
            ]
            arrays.append(np.asarray(values))
            labels.append(f"{kernel}\n{MODE_LABELS[mode]}")
            colors.append(MODE_COLORS[mode])
            positions.append(position)
            position += 1.0
        if kernel_index + 1 < len(KERNELS):
            position += 0.8

    figure, axis = plt.subplots(figsize=(16, 8.5), constrained_layout=True)
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
    axis.set_xticks(positions, labels, rotation=18, ha="right")
    axis.set_ylabel("Device-event runtime per kernel call [ms]")
    axis.set_xlabel("Kernel + execution mode")
    axis.set_title("HASEonGPU A100 two-kernel tuning: equal-count runtime distributions")
    axis.grid(axis="y", alpha=0.25)

    for x, kernel in zip((np.mean(positions[:4]), np.mean(positions[4:])), KERNELS):
        axis.text(
            x,
            1.01,
            f"{summary[kernel]['baseline']['samples']:,} samples per mode",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            color="#444444",
        )
    for x, values in zip(positions, [summary[kernel][mode] for kernel in KERNELS for mode in PLOT_MODES]):
        axis.annotate(
            f"{values['median_ms']:.3f}",
            (x, values["median_ms"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    figure.savefig(path, dpi=190)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def write_report(path: Path, summary: dict, failures: list[str]) -> dict[str, bool]:
    goals = {
        "minimum_offline": all(summary[kernel]["offline"]["beats_baseline_median"] for kernel in KERNELS),
        "medium_online_fixed": all(
            summary[kernel]["online_fixed"]["beats_baseline_median"] for kernel in KERNELS
        ),
        "maximum_online_adaptive": all(
            summary[kernel]["online_adaptive"]["beats_baseline_median"] for kernel in KERNELS
        ),
    }
    lines = [
        "# Two-kernel A100 tuning campaign",
        "",
        "All plotted modes use equal per-kernel sample counts. Baseline samples come from the prior fixed-default all-kernel campaign; no new baseline was run. Every duration is measured only with Alpaka device events: tuner-owned event pairs during learning and direct event pairs during cached replay. FrameSpec candidates preserve the original worker coverage exactly.",
        "",
        "## Median results",
        "",
    ]
    for kernel in KERNELS:
        lines.append(f"### `{kernel}`")
        lines.append("")
        for mode in PLOT_MODES:
            values = summary[kernel][mode]
            suffix = ""
            if mode != "baseline":
                suffix = f", {values['median_speedup']:.4f}x baseline"
            lines.append(
                f"- {MODE_LABELS[mode]}: {values['median_ms']:.6f} ms median, "
                f"5th--95th percentile {values['p05_ms']:.6f}--{values['p95_ms']:.6f} ms, "
                f"{values['samples']:,} measurements{suffix}."
            )
        lines.append("")
    lines.extend(["## Requirements", ""])
    lines.extend(
        [
            f"- {'PASS' if goals['minimum_offline'] else 'FAIL'} minimum: offline median beats baseline for both kernels.",
            f"- {'PASS' if goals['medium_online_fixed'] else 'FAIL'} medium: learned online-fixed median beats baseline for both kernels.",
            f"- {'PASS' if goals['maximum_online_adaptive'] else 'FAIL'} maximum: learned online-adaptive median beats baseline for both kernels.",
        ]
    )
    lines.extend(["", "## Validation", ""])
    lines.extend([f"- FAIL: {failure}" for failure in failures] or ["- All structural and equal-count checks passed."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return goals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--baseline-trace", required=True, type=Path)
    args = parser.parse_args()
    raw = args.run_root / "raw" / BACKEND
    traces = {mode: load_jsonl(raw / mode / "trace.jsonl") for mode in RAW_MODES}
    baseline = load_baseline_reference(args.baseline_trace)

    failures: list[str] = []
    failures.extend(validate_mode("baseline", baseline))
    for mode, records in traces.items():
        failures.extend(validate_mode(mode, records))
    comparable = {mode: equal_count_records(records, True) for mode, records in traces.items()}
    comparable["baseline"] = equal_count_records(baseline, False)
    reference_counts = Counter(record["kernel"] for record in comparable["baseline"])
    for mode in RAW_MODES:
        counts = Counter(record["kernel"] for record in comparable[mode])
        if counts != reference_counts:
            failures.append(f"{mode}: sample counts {dict(counts)} differ from baseline {dict(reference_counts)}")

    records_by_mode = {
        "baseline": comparable["baseline"],
        "offline": comparable["offline"],
        "online_fixed": comparable["online_fixed"],
        "online_adaptive": comparable["online_adaptive"],
    }
    summary = summarize(records_by_mode)
    output = args.run_root / "evaluations"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "two-kernel-tuning.csv", records_by_mode)
    goals = write_report(output / "report.md", summary, failures)
    document = {"validation_failures": failures, "requirements": goals, "kernels": summary}
    (output / "summary.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    plot(output / "two-kernel-tuning.png", records_by_mode, summary)
    if failures:
        raise SystemExit("validation failed; see evaluations/report.md")
    print(json.dumps(goals, sort_keys=True))


if __name__ == "__main__":
    main()
