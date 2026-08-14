#!/usr/bin/env python3
"""Validate and summarize the ordinary-build default versus static-winner campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


BACKEND = "Cuda_NvidiaGpu_GpuCuda"
EXPECTED_CONTEXTS = {
    ("AccumulateForwardPhiAseReservoir", 977, 128),
    ("AccumulateReflectedForwardPhiAse", 977, 128),
    ("TraceGeneralPump", 98, 512),
}
ORDER = (
    "default_0", "selected_0", "selected_1", "default_1",
    "selected_2", "default_2", "default_3", "selected_3",
    "default_4", "selected_4", "selected_5", "default_5",
)
FORBIDDEN_ARTIFACTS = {"trace.jsonl", "metrics.jsonl", "step-timings.jsonl"}
SUMMARY_KEYS = {
    "completed_steps",
    "timed_out",
    "finite",
    "elapsed_seconds",
    "phi_ase_sum",
    "beta_volume_sum",
}
ELAPSED = re.compile(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)")


def time_seconds(path: Path) -> float:
    match = ELAPSED.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"missing GNU time elapsed value in {path}")
    fields = [float(value) for value in match.group(1).split(":" )]
    if len(fields) == 2:
        return fields[0] * 60.0 + fields[1]
    if len(fields) == 3:
        return fields[0] * 3600.0 + fields[1] * 60.0 + fields[2]
    raise ValueError(f"unexpected elapsed value in {path}")


def metadata_value(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}=(.+)$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"missing metadata key {key}")
    return match.group(1)


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "sample_sd_seconds": statistics.stdev(values),
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    raw = args.run_root / "raw" / BACKEND
    failures: list[str] = []
    for marker in (
        "discovery-build-completed",
        "discovery-completed",
        "static-build-completed",
        "static-wall-completed",
    ):
        if not (args.run_root / marker).is_file():
            failures.append(f"stage marker is absent: {marker}")
    actual = sorted(path.name for path in raw.iterdir() if path.is_dir())
    if set(actual) != set(ORDER):
        failures.append(f"run directories differ from the exact 12-run design: {actual}")

    winners = json.loads((args.run_root / "winners.json").read_text(encoding="utf-8"))
    winners_sha = hashlib.sha256((args.run_root / "winners.json").read_bytes()).hexdigest()
    if winners.get("schema_version") != 1:
        failures.append("winner contract schema is not version 1")
    winner_contexts = winners.get("contexts", [])
    winner_keys = {
        (item.get("kernel"), item.get("original_num_frames"), item.get("original_frame_extent"))
        for item in winner_contexts
        if isinstance(item, dict)
    }
    if winner_keys != EXPECTED_CONTEXTS or len(winner_contexts) != len(EXPECTED_CONTEXTS):
        failures.append("winner contract does not contain the exact three required contexts")
    for item in winner_contexts:
        if not isinstance(item, dict):
            failures.append("winner contract contains a non-object context")
            continue
        if item.get("candidate_count") != 25 or item.get("measured_candidate_count") != 25:
            failures.append(f"winner context {item.get('context_id')} is not complete over 25 candidates")
        if item.get("winner_sample_count") != 7:
            failures.append(f"winner context {item.get('context_id')} does not have seven winner samples")
        if any(not isinstance(item.get(key), int) or item[key] <= 0 for key in ("selected_num_frames", "selected_frame_extent")):
            failures.append(f"winner context {item.get('context_id')} has an invalid selected FrameSpec")
    discovery_history = args.run_root / "discovery" / "complete-history.json"
    if not discovery_history.is_file() or hashlib.sha256(discovery_history.read_bytes()).hexdigest() != winners.get(
        "source_history_sha256"
    ):
        failures.append("winner contract is not tied to the retrieved complete discovery history")
    discovery_metadata_path = args.run_root / "discovery" / "metadata.txt"
    if not discovery_metadata_path.is_file():
        failures.append("discovery metadata is absent")
    elif metadata_value(discovery_metadata_path.read_text(encoding="utf-8"), "forward_ray_count") != "1000000":
        failures.append("discovery did not use the required fixed forward ray count")
    manifest = json.loads((args.run_root / "source-manifest.json").read_text(encoding="utf-8"))
    manifest_sha = hashlib.sha256((args.run_root / "source-manifest.json").read_bytes()).hexdigest()
    source_revision = manifest.get("source", {}).get("head")
    source_dirty = manifest.get("source", {}).get("dirty")
    if not isinstance(source_revision, str) or len(source_revision) != 40:
        failures.append("local source revision is missing from source-manifest.json")
    if not isinstance(source_dirty, bool):
        failures.append("local dirty state is missing from source-manifest.json")
    dependencies = manifest.get("dependencies")
    expected_dependency_heads = {
        "1641d11c5983b76d4f49585b7f775eae6e34a4a4",
        "bbf6e93eb803a09a5b4d5b9736a5b519163051d2",
    }
    if not isinstance(dependencies, list) or len(dependencies) != 2:
        failures.append("source manifest does not record both Alpaka dependency repositories")
    else:
        for dependency in dependencies:
            if not isinstance(dependency, dict):
                failures.append("source manifest contains an invalid dependency entry")
                continue
            if not isinstance(dependency.get("head"), str) or len(dependency["head"]) != 40:
                failures.append("a dependency revision is missing from source-manifest.json")
            if not isinstance(dependency.get("dirty"), bool):
                failures.append("a dependency dirty state is missing from source-manifest.json")
        if {dependency.get("head") for dependency in dependencies if isinstance(dependency, dict)} != expected_dependency_heads:
            failures.append("source manifest dependency revisions do not match the pinned campaign inputs")

    runs: list[dict[str, object]] = []
    reference: dict[str, object] | None = None
    nodes: set[str] = set()
    gpu_rows: set[str] = set()
    job_ids: set[str] = set()
    binary_digests: dict[str, set[str]] = {"default": set(), "selected": set()}
    for position, label in enumerate(ORDER):
        directory = raw / label
        mode = label.split("_", 1)[0]
        required = [directory / name for name in ("summary.json", "time.txt", "metadata.txt", "stdout.txt")]
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
        if missing:
            failures.append(f"{label}: missing or empty artifacts: {missing}")
            continue
        forbidden = sorted(path.name for path in directory.rglob("*") if path.name in FORBIDDEN_ARTIFACTS)
        if forbidden:
            failures.append(f"{label}: forbidden instrumentation artifacts exist: {forbidden}")
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if set(summary) != SUMMARY_KEYS:
            failures.append(f"{label}: summary schema differs from the cell-centered campaign contract")
        metadata = (directory / "metadata.txt").read_text(encoding="utf-8")
        if metadata_value(metadata, "mode") != mode:
            failures.append(f"{label}: metadata mode mismatch")
        if metadata_value(metadata, "forward_ray_count") != "1000000":
            failures.append(f"{label}: fixed forward ray count mismatch")
        if metadata_value(metadata, "source_manifest_sha256") != manifest_sha:
            failures.append(f"{label}: source manifest digest mismatch")
        if metadata_value(metadata, "winners_sha256") != winners_sha:
            failures.append(f"{label}: winner contract digest mismatch")
        if "HASE_ENABLE_ALPAKATUNE:BOOL=OFF" not in metadata:
            failures.append(f"{label}: alpakaTune is not proven disabled")
        if "HASE_ENABLE_DEVICE_TIMING:BOOL=OFF" not in metadata:
            failures.append(f"{label}: device timing is not proven disabled")
        if f"HASE_STATIC_FRAMESPEC_MODE:STRING={mode}" not in metadata:
            failures.append(f"{label}: compiled static FrameSpec mode mismatch")
        if f"HASE_STATIC_FRAMESPEC_WINNERS_FILE:FILEPATH={args.run_root / 'winners.json'}" not in metadata:
            failures.append(f"{label}: compiled winner path mismatch")
        if f"HASE_STATIC_FRAMESPEC_WINNERS_SHA256:STRING={winners_sha}" not in metadata:
            failures.append(f"{label}: compiled winner digest mismatch")
        if re.search(r"libalpakatune|/alpakaTune/", metadata, re.IGNORECASE):
            failures.append(f"{label}: binary dependency metadata mentions alpakaTune")
        binary_rows = re.findall(r"^([0-9a-f]{64})  \S+/calcPhiASE$", metadata, re.MULTILINE)
        if len(binary_rows) != 1:
            failures.append(f"{label}: expected exactly one executable digest")
        else:
            binary_digests[mode].add(binary_rows[0])
        nodes.add(metadata_value(metadata, "node"))
        job_ids.add(metadata_value(metadata, "job_id"))
        gpu_lines = [line for line in metadata.splitlines() if "GPU-" in line]
        if len(gpu_lines) != 1:
            failures.append(f"{label}: expected exactly one GPU UUID row")
        else:
            gpu_rows.add(gpu_lines[0])
        if summary.get("completed_steps") != 1000 or summary.get("timed_out") or not summary.get("finite"):
            failures.append(f"{label}: incomplete, timed out, or non-finite")
        application_seconds = summary.get("elapsed_seconds")
        if not isinstance(application_seconds, (int, float)) or not math.isfinite(application_seconds) or application_seconds <= 0:
            failures.append(f"{label}: invalid application elapsed time")
        if reference is None:
            reference = summary
        for key in ("phi_ase_sum", "beta_volume_sum"):
            value = summary.get(key)
            baseline = reference.get(key) if reference else None
            if not isinstance(value, (int, float)) or not isinstance(baseline, (int, float)) or not math.isclose(
                value, baseline, rel_tol=1.0e-12, abs_tol=0.0
            ):
                failures.append(f"{label}: {key} differs from the first default run")
        wall_seconds = time_seconds(directory / "time.txt")
        if not math.isfinite(wall_seconds) or wall_seconds <= 0:
            failures.append(f"{label}: invalid external wall time")
        runs.append(
            {
                "position": position,
                "label": label,
                "mode": mode,
                "wall_seconds": wall_seconds,
                "application_seconds": (
                    float(application_seconds) if isinstance(application_seconds, (int, float)) else math.nan
                ),
            }
        )

    if len(nodes) != 1 or len(gpu_rows) != 1 or len(job_ids) != 1:
        failures.append("measured runs did not share exactly one allocation, node, and GPU UUID")
    if any(len(digests) != 1 for digests in binary_digests.values()):
        failures.append("a measured mode used more or fewer than one executable")
    elif binary_digests["default"] == binary_digests["selected"]:
        failures.append("default and selected modes used the same executable digest")
    by_label = {str(run["label"]): run for run in runs}
    if len(by_label) != len(ORDER):
        failures.append("not all 12 designed runs were readable")

    output = args.run_root / "evaluations"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("position", "label", "mode", "wall_seconds", "application_seconds"))
        writer.writeheader()
        writer.writerows(runs)

    modes: dict[str, dict[str, object]] = {}
    pairs: list[dict[str, float | int]] = []
    if len(by_label) == len(ORDER):
        for mode in ("default", "selected"):
            selected = [run for run in runs if run["mode"] == mode]
            modes[mode] = {
                "wall": summarize([float(run["wall_seconds"]) for run in selected]),
                "application": summarize([float(run["application_seconds"]) for run in selected]),
            }
        for index in range(6):
            default = by_label[f"default_{index}"]
            selected = by_label[f"selected_{index}"]
            pairs.append(
                {
                    "pair": index,
                    "wall_delta_seconds": float(selected["wall_seconds"]) - float(default["wall_seconds"]),
                    "application_delta_seconds": float(selected["application_seconds"]) - float(default["application_seconds"]),
                }
            )
    result = {
        "validation_failures": failures,
        "source_revision": source_revision,
        "source_dirty": source_dirty,
        "gpu": next(iter(gpu_rows), None),
        "order": list(ORDER),
        "modes": modes,
        "paired_deltas": pairs,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    report = [
        "# Expanded random discovery: static winner wall-time validation",
        "",
        "Random search was used only to select one FrameSpec for each reflection-on forward/pump launch context at a fixed global ASE ray count of 1,000,000. This report compares two ordinary, separately compiled binaries: unchanged defaults and those static winners. All 12 measured processes shared one GPU allocation and used the counterbalanced order `D-S-S-D / S-D-D-S / D-S-S-D` after one warm-up per binary. No adaptive ray-count sequence or online amortization is tested or claimed.",
        "",
        f"Local source revision: `{source_revision}` (dirty snapshot: `{source_dirty}`).",
        f"GPU identity: `{next(iter(gpu_rows), 'unavailable')}`.",
        "",
        "| Mode | Mean external wall (s) | SD | Mean application runtime (s) | SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in ("default", "selected"):
        if mode in modes:
            report.append(
                f"| {mode} | {modes[mode]['wall']['mean_seconds']:.3f} | "
                f"{modes[mode]['wall']['sample_sd_seconds']:.3f} | "
                f"{modes[mode]['application']['mean_seconds']:.3f} | "
                f"{modes[mode]['application']['sample_sd_seconds']:.3f} |"
            )
    if pairs:
        report.extend(
            [
                "",
                f"Mean paired selected-minus-default wall delta: {statistics.mean(float(pair['wall_delta_seconds']) for pair in pairs):+.3f} s.",
                f"Mean paired selected-minus-default application delta: {statistics.mean(float(pair['application_delta_seconds']) for pair in pairs):+.3f} s.",
            ]
        )
    report.extend(["", "## Completeness and correctness", ""])
    report.extend([f"- FAIL: {failure}" for failure in failures] or ["All completeness, allocation, instrumentation, and numerical-equivalence checks passed."])
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("validation failed; see evaluations/report.md")


if __name__ == "__main__":
    main()
