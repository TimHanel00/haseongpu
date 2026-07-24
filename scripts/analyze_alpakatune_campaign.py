#!/usr/bin/env python3
"""Validate and plot the Rosi laserPumpCladding alpakaTune campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODES = (
    "baseline",
    "random",
    "learned_hybrid",
)
BACKENDS = ("Cuda_NvidiaGpu_GpuCuda",)
PLOT_SAMPLE_LIMIT = 10_000


@dataclass
class BoundedSeries:
    """Deterministic reservoir sample for a potentially very large trace series."""

    label: str
    limit: int = PLOT_SAMPLE_LIMIT
    seen: int = 0
    points: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        digest = hashlib.sha256(self.label.encode("utf-8")).digest()
        self._random = random.Random(int.from_bytes(digest[:8], "little"))

    def add(self, point: tuple[float, float]):
        self.seen += 1
        if len(self.points) < self.limit:
            self.points.append(point)
            return
        replacement = self._random.randrange(self.seen)
        if replacement < self.limit:
            self.points[replacement] = point


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_campaign(run_root: Path):
    runs = []
    runtime_samples = {}
    latency_samples = {}
    recommendation_counts = defaultdict(int)
    for backend in BACKENDS:
        for mode in MODES:
            for repetition in range(3):
                directory = run_root / "raw" / backend / mode / f"rep-{repetition}"
                summary_path = directory / "summary.json"
                if not summary_path.exists():
                    raise RuntimeError(f"missing campaign summary: {summary_path}")
                summary = load_json(summary_path)
                trace_path = directory / "trace.jsonl"
                trace_count = 0
                if trace_path.exists():
                    with trace_path.open(encoding="utf-8") as handle:
                        for line in handle:
                            if not line.strip():
                                continue
                            trace = json.loads(line)
                            trace_count += 1
                            learned_status = str(trace.get("learned_status", ""))
                            if learned_status.startswith("fallback_"):
                                raise RuntimeError(f"learned fallback in {directory}: {learned_status}")

                            recommendation = trace.get("recommendation_seconds")
                            if recommendation is not None:
                                recommendation_counts[(backend, mode)] += 1
                                key = (backend, mode)
                                series = latency_samples.get(key)
                                if series is None:
                                    series = BoundedSeries(f"latency:{backend}:{mode}")
                                    latency_samples[key] = series
                                series.add(
                                    (
                                        recommendation_counts[(backend, mode)],
                                        float(recommendation) * 1.0e6,
                                    )
                                )

                            if not trace.get("measured") or trace.get("runtime_seconds") is None:
                                continue
                            kernel = trace["kernel"]
                            key = (backend, mode, kernel)
                            series = runtime_samples.get(key)
                            if series is None:
                                series = BoundedSeries(f"runtime:{backend}:{mode}:{kernel}")
                                runtime_samples[key] = series
                            series.add(
                                (
                                    float(trace["elapsed_seconds"]),
                                    float(trace["runtime_seconds"]) * 1.0e6,
                                )
                            )
                if mode != "baseline" and trace_count == 0:
                    raise RuntimeError(f"missing tuner trace: {trace_path}")
                print(f"loaded {backend}/{mode}/rep-{repetition}: {trace_count} traces", flush=True)
                runs.append(
                    {
                        "backend": backend,
                        "mode": mode,
                        "repetition": repetition,
                        "directory": directory,
                        "summary": summary,
                        "trace_count": trace_count,
                    }
                )
    return runs, runtime_samples, latency_samples


def validate_numerics(runs):
    failures = []
    references = {}
    for backend in BACKENDS:
        completed = [
            run
            for run in runs
            if run["backend"] == backend
            and run["mode"] == "baseline"
            and not run["summary"]["timed_out"]
            and run["summary"]["phi_ase_sum"] is not None
            and run["summary"]["beta_cells_sum"] is not None
        ]
        if completed:
            references[backend] = {
                metric: statistics.median(run["summary"][metric] for run in completed)
                for metric in ("phi_ase_sum", "beta_cells_sum")
            }

    for run in runs:
        summary = run["summary"]
        label = f'{run["backend"]}/{run["mode"]}/rep-{run["repetition"]}'
        if not summary["finite"]:
            failures.append(f"{label}: non-finite numerical output")
        if summary["timed_out"] or run["backend"] not in references:
            continue
        for metric, reference in references[run["backend"]].items():
            value = summary[metric]
            if value is None or reference is None:
                failures.append(f"{label}: {metric} is missing from a completed run")
                continue
            scale = max(abs(reference), np.finfo(float).tiny)
            relative = abs(value - reference) / scale
            if relative > 0.05:
                failures.append(f"{label}: {metric} differs from baseline by {relative:.2%}")
    return failures, references


def write_run_table(runs, output: Path):
    fields = (
        "backend",
        "mode",
        "repetition",
        "completed_steps",
        "elapsed_seconds",
        "timed_out",
        "finite",
        "phi_ase_sum",
        "beta_cells_sum",
        "beta_volume_sum",
        "trace_count",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {key: run["summary"].get(key) for key in fields}
            row.update(
                backend=run["backend"],
                mode=run["mode"],
                repetition=run["repetition"],
                trace_count=run["trace_count"],
            )
            writer.writerow(row)


def plot_kernel_runtime(runtime_samples, output_dir: Path):
    for backend in BACKENDS:
        kernels = sorted({
            kernel
            for sample_backend, _, kernel in runtime_samples
            if sample_backend == backend
        })
        if not kernels:
            continue
        figure, axes = plt.subplots(len(kernels), 1, figsize=(10, 3.3 * len(kernels)), squeeze=False)
        for axis, kernel in zip(axes[:, 0], kernels, strict=True):
            for mode in MODES[1:]:
                series = runtime_samples.get((backend, mode, kernel))
                if series:
                    selected = sorted(series.points)
                    axis.scatter(
                        [point[0] for point in selected],
                        [point[1] for point in selected],
                        s=9,
                        alpha=0.55,
                        label=f"{mode} (sample {len(selected)}/{series.seen})",
                    )
            axis.set_title(kernel)
            axis.set_xlabel("Elapsed invocation time (s)")
            axis.set_ylabel("Scored kernel runtime (us)")
            axis.grid(alpha=0.2)
        axes[0, 0].legend(loc="best", fontsize="small")
        figure.suptitle(backend)
        figure.tight_layout()
        figure.savefig(output_dir / f"kernel-runtime-{backend}.png", dpi=180)
        plt.close(figure)


def plot_recommendation_latency(latency_samples, output_dir: Path):
    for backend in BACKENDS:
        figure, axis = plt.subplots(figsize=(10, 5))
        for mode in MODES[1:]:
            series = latency_samples.get((backend, mode))
            if series:
                selected = sorted(series.points)
                axis.plot(
                    [point[0] for point in selected],
                    [point[1] for point in selected],
                    ".",
                    ms=3,
                    alpha=0.65,
                    label=f"{mode} (sample {len(selected)}/{series.seen})",
                )
        axis.set_yscale("symlog", linthresh=1.0)
        axis.set_xlabel("Natural HASE kernel invocation")
        axis.set_ylabel("Recommendation and re-sort latency (us)")
        axis.set_title(backend)
        axis.grid(alpha=0.2)
        axis.legend(loc="best", fontsize="small")
        figure.tight_layout()
        figure.savefig(output_dir / f"recommendation-latency-{backend}.png", dpi=180)
        plt.close(figure)


def plot_application_time(runs, output_dir: Path):
    figure, axes = plt.subplots(1, len(BACKENDS), figsize=(14, 5), squeeze=False)
    for axis, backend in zip(axes[0], BACKENDS, strict=True):
        values = defaultdict(list)
        for run in runs:
            if run["backend"] == backend:
                values[run["mode"]].append(run["summary"]["elapsed_seconds"])
        medians = [statistics.median(values[mode]) for mode in MODES]
        axis.bar(np.arange(len(MODES)), medians)
        axis.set_xticks(np.arange(len(MODES)), MODES, rotation=35, ha="right")
        axis.set_ylabel("Application wall time (s)")
        axis.set_title(backend)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "application-wall-time.png", dpi=180)
    plt.close(figure)


def write_report(runs, references, failures, output: Path):
    lines = [
        "# HASEonGPU alpakaTune validation",
        "",
        "Each cell is `completed runs / timed-out runs`; every cell contains three repetitions.",
        "",
        "| Backend | " + " | ".join(MODES) + " |",
        "|---|" + "---|" * len(MODES),
    ]
    for backend in BACKENDS:
        cells = []
        for mode in MODES:
            selected = [run for run in runs if run["backend"] == backend and run["mode"] == mode]
            timed_out = sum(run["summary"]["timed_out"] for run in selected)
            cells.append(f"{len(selected) - timed_out} / {timed_out}")
        lines.append(f"| {backend} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Numerical validation", ""])
    if failures:
        lines.extend(f"- FAIL: {failure}" for failure in failures)
    else:
        lines.append("All available outputs are finite; completed tuned runs stay within 5% of the matching backend baseline integrals.")
    if not references:
        lines.append("No backend completed a baseline run, so integral comparisons are censored.")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = args.run_root / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs, runtime_samples, latency_samples = load_campaign(args.run_root)
    failures, references = validate_numerics(runs)
    write_run_table(runs, output_dir / "runs.csv")
    plot_kernel_runtime(runtime_samples, output_dir)
    plot_recommendation_latency(latency_samples, output_dir)
    plot_application_time(runs, output_dir)
    write_report(runs, references, failures, output_dir / "report.md")
    if failures:
        raise SystemExit("numerical validation failed; see evaluations/report.md")


if __name__ == "__main__":
    main()
