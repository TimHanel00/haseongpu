#!/usr/bin/env python3
# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plot the expanded static-winner wall-time campaign."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


DEFAULT_COLOR = "#4C78A8"
SELECTED_COLOR = "#F58518"
POINT_EDGE = "#202020"
KERNEL_ORDER = {
    "AccumulateForwardPhiAseReservoir": 0,
    "AccumulateReflectedForwardPhiAse": 1,
    "TraceGeneralPump": 2,
}
EXPECTED_LABELS = {
    *(f"default_{index}" for index in range(6)),
    *(f"selected_{index}" for index in range(6)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired application and kernel boxplots for the expanded static-wall campaign."
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: RUN_ROOT/evaluations).",
    )
    return parser.parse_args()


def load_runs(run_root: Path) -> list[dict[str, str]]:
    path = run_root / "evaluations" / "runs.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    labels = {row["label"] for row in rows}
    if len(rows) != 12 or labels != EXPECTED_LABELS:
        raise ValueError(f"{path} does not contain the exact six paired runs per mode")
    return rows


def paired_values(rows: list[dict[str, str]], key: str) -> tuple[list[float], list[float]]:
    by_label = {row["label"]: float(row[key]) for row in rows}
    return (
        [by_label[f"default_{index}"] for index in range(6)],
        [by_label[f"selected_{index}"] for index in range(6)],
    )


def style_boxplot(axis, boxes, colors: tuple[str, ...]) -> None:
    for patch, color in zip(boxes["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.48)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
    for median in boxes["medians"]:
        median.set_color(POINT_EDGE)
        median.set_linewidth(1.8)
    for artist in (*boxes["whiskers"], *boxes["caps"]):
        artist.set_color("#555555")
        artist.set_linewidth(1.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)


def plot_application_boxplots(rows: list[dict[str, str]], output_dir: Path) -> None:
    metrics = (
        ("wall_seconds", "External wall time", "GNU time elapsed (s)"),
        ("application_seconds", "Application runtime", "Application elapsed (s)"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 5.3), constrained_layout=True)
    for axis, (key, title, ylabel) in zip(axes, metrics, strict=True):
        default, selected = paired_values(rows, key)
        boxes = axis.boxplot(
            [default, selected],
            positions=(1.0, 2.0),
            widths=0.48,
            patch_artist=True,
            showfliers=False,
        )
        style_boxplot(axis, boxes, (DEFAULT_COLOR, SELECTED_COLOR))
        for default_value, selected_value in zip(default, selected, strict=True):
            axis.plot(
                (1.0, 2.0),
                (default_value, selected_value),
                color="#9A9A9A",
                linewidth=0.9,
                alpha=0.62,
                zorder=1,
            )
        axis.scatter(
            [1.0] * len(default),
            default,
            s=32,
            color=DEFAULT_COLOR,
            edgecolor=POINT_EDGE,
            linewidth=0.45,
            zorder=3,
        )
        axis.scatter(
            [2.0] * len(selected),
            selected,
            s=32,
            color=SELECTED_COLOR,
            edgecolor=POINT_EDGE,
            linewidth=0.45,
            zorder=3,
        )
        default_mean = sum(default) / len(default)
        selected_mean = sum(selected) / len(selected)
        change = 100.0 * (selected_mean - default_mean) / default_mean
        outcome = f"{abs(change):.1f}% {'slower' if change >= 0.0 else 'faster'}"
        axis.set_title(title, fontsize=12, weight="semibold")
        axis.set_ylabel(ylabel)
        axis.set_xticks((1.0, 2.0), ("Default", "Selected"))
        axis.text(
            0.5,
            0.97,
            f"selected mean: {outcome}",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            weight="semibold",
        )
    figure.suptitle("Ordinary 1,000-step builds on one A100 allocation", fontsize=14, weight="bold")
    figure.supxlabel("Boxes: IQR and median; points and gray lines: six paired runs", fontsize=9)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"runtime_boxplots.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def scalar_vector(value: object) -> int:
    text = str(value)
    if not (text.startswith("{") and text.endswith("}")):
        raise ValueError(f"invalid scalar vector: {value!r}")
    return int(text[1:-1])


def kernel_samples(run_root: Path) -> list[dict[str, object]]:
    history = json.loads((run_root / "discovery" / "complete-history.json").read_text(encoding="utf-8"))
    winners = json.loads((run_root / "winners.json").read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for winner in winners["contexts"]:
        context = history["contexts"][winner["context_id"]]
        configurations = context["candidate_configurations"]
        default_index = next(
            index
            for index, configuration in enumerate(configurations)
            if scalar_vector(configuration["numFrames"]) == winner["original_num_frames"]
            and scalar_vector(configuration["frameExtent"]) == winner["original_frame_extent"]
        )
        selected_index = int(winner["winner_candidate_index"])
        default_estimate = float(context["candidate_estimates"][default_index])
        selected_estimate = float(context["candidate_estimates"][selected_index])
        records.append(
            {
                "kernel": winner["kernel"],
                "original_frames": int(winner["original_num_frames"]),
                "original_extent": int(winner["original_frame_extent"]),
                "selected_frames": int(winner["selected_num_frames"]),
                "selected_extent": int(winner["selected_frame_extent"]),
                "default": [1.0e3 * float(value) for value in context["candidate_samples"][default_index]],
                "selected": [1.0e3 * float(value) for value in context["candidate_samples"][selected_index]],
                "reduction": 100.0 * (default_estimate - selected_estimate) / default_estimate,
            }
        )
    return sorted(
        records,
        key=lambda record: KERNEL_ORDER[str(record["kernel"])],
    )


def plot_kernel_boxplots(run_root: Path, output_dir: Path) -> None:
    records = kernel_samples(run_root)
    if len(records) != 3 or {str(record["kernel"]) for record in records} != set(KERNEL_ORDER):
        raise ValueError("expected the exact three reflection-on tuned launch contexts")

    figure, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    centers = list(range(1, len(records) + 1))
    default_positions = [position - 0.18 for position in centers]
    selected_positions = [position + 0.18 for position in centers]
    default_data = [record["default"] for record in records]
    selected_data = [record["selected"] for record in records]
    default_boxes = axis.boxplot(
        default_data,
        positions=default_positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
    )
    selected_boxes = axis.boxplot(
        selected_data,
        positions=selected_positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
    )
    style_boxplot(axis, default_boxes, (DEFAULT_COLOR,) * len(records))
    style_boxplot(axis, selected_boxes, (SELECTED_COLOR,) * len(records))

    jitter = (-0.045, -0.030, -0.015, 0.0, 0.015, 0.030, 0.045)
    for index, record in enumerate(records):
        for position, values, color in (
            (default_positions[index], record["default"], DEFAULT_COLOR),
            (selected_positions[index], record["selected"], SELECTED_COLOR),
        ):
            axis.scatter(
                [position + offset for offset in jitter],
                values,
                s=22,
                color=color,
                edgecolor=POINT_EDGE,
                linewidth=0.35,
                zorder=3,
            )
        group_maximum = max(*record["default"], *record["selected"])
        axis.text(
            centers[index],
            group_maximum + 0.055,
            f"{record['reduction']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="semibold",
        )

    labels = []
    for record in records:
        kernel = {
            "AccumulateForwardPhiAseReservoir": "Forward ASE\ndirect reservoir pass",
            "AccumulateReflectedForwardPhiAse": "Forward ASE\nreflected passes",
            "TraceGeneralPump": "Pump",
        }[str(record["kernel"])]
        labels.append(
            f"{kernel}\n"
            f"{record['original_frames']}×{record['original_extent']} → "
            f"{record['selected_frames']}×{record['selected_extent']}"
        )
    axis.set_xticks(centers, labels)
    axis.tick_params(axis="x", labelsize=9)
    axis.set_xlim(0.48, len(records) + 0.52)
    axis.set_ylim(top=max(max(*record["default"], *record["selected"]) for record in records) + 0.24)
    axis.set_ylabel("Measured kernel runtime (ms)")
    axis.set_title(
        "Random-search samples: default FrameSpec versus selected winner",
        fontsize=14,
        weight="bold",
    )
    axis.text(
        0.5,
        0.98,
        "Labels above groups show reduction of the tuner estimate; lower is better",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    axis.legend(
        handles=(
            Patch(facecolor=DEFAULT_COLOR, edgecolor=DEFAULT_COLOR, alpha=0.48, label="Default"),
            Patch(facecolor=SELECTED_COLOR, edgecolor=SELECTED_COLOR, alpha=0.48, label="Selected"),
        ),
        loc="upper right",
        frameon=False,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"kernel_winner_boxplots.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_root / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_runs(args.run_root)
    plot_application_boxplots(rows, output_dir)
    plot_kernel_boxplots(args.run_root, output_dir)


if __name__ == "__main__":
    main()
