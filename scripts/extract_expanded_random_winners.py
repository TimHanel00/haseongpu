#!/usr/bin/env python3
"""Validate random-search history and emit one static FrameSpec per launch context."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


KERNELS = {
    "AccumulateForwardPhiAseReservoir",
    "AccumulateReflectedForwardPhiAse",
    "TraceGeneralPump",
}
EXPECTED_CONTEXTS = {
    ("AccumulateForwardPhiAseReservoir", 977, 128),
    ("AccumulateReflectedForwardPhiAse", 977, 128),
    ("TraceGeneralPump", 98, 512),
}
VECTOR = re.compile(r"^\{([0-9]+)\}$")
LAUNCH = re.compile(r"^FrameSpec\{\{([0-9]+)\},\{([0-9]+)\}, executor=([^}]+)\}$")


def scalar_vector(value: object, label: str) -> int:
    match = VECTOR.fullmatch(str(value))
    if match is None:
        raise ValueError(f"{label} is not a one-dimensional integral vector: {value!r}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-count", type=int, default=25)
    parser.add_argument("--runs-per-candidate", type=int, default=7)
    args = parser.parse_args()

    raw = args.history.read_bytes()
    history = json.loads(raw)
    if history.get("schema_version") != 11:
        raise ValueError(f"unsupported history schema: {history.get('schema_version')!r}")
    contexts = history.get("contexts")
    if not isinstance(contexts, dict):
        raise ValueError("history contexts are missing or invalid")

    winners: list[dict[str, object]] = []
    seen: set[tuple[str, int, int]] = set()
    for context_id, context in sorted(contexts.items()):
        metadata = context.get("metadata", {})
        kernel = str(metadata.get("kernel", "")).rsplit("::", 1)[-1]
        if kernel not in KERNELS:
            raise ValueError(f"unexpected kernel in context {context_id}: {kernel!r}")
        launch = LAUNCH.fullmatch(str(metadata.get("launch_specification", "")))
        if launch is None:
            raise ValueError(f"invalid launch specification in context {context_id}")
        original_frames, original_extent = int(launch.group(1)), int(launch.group(2))
        key = kernel, original_frames, original_extent
        if key in seen:
            raise ValueError(f"duplicate launch context: {key}")
        seen.add(key)

        configurations = context.get("candidate_configurations")
        samples = context.get("candidate_samples")
        estimates = context.get("candidate_estimates")
        best = context.get("best_candidate_index")
        candidate_count = context.get("candidate_count")
        if candidate_count != args.candidate_count:
            raise ValueError(
                f"context {context_id} candidate count is {candidate_count!r}, expected {args.candidate_count}"
            )
        if not all(isinstance(value, list) and len(value) == candidate_count for value in (configurations, samples, estimates)):
            raise ValueError(f"context {context_id} has inconsistent candidate arrays")
        if metadata.get("mode") != "online_fixed" or metadata.get("strategy") != "random":
            raise ValueError(f"context {context_id} is not an online-fixed random-search context")
        if context.get("completion_reason") != "all_configurations":
            raise ValueError(f"context {context_id} did not complete all candidate configurations")
        if context.get("execution_budget_reached") is not False:
            raise ValueError(f"context {context_id} unexpectedly terminated at its execution budget")
        expected_executions = args.candidate_count * args.runs_per_candidate
        if context.get("execution_count") != expected_executions:
            raise ValueError(
                f"context {context_id} execution count is {context.get('execution_count')!r}, "
                f"expected {expected_executions}"
            )
        if any(len(candidate_samples) != args.runs_per_candidate for candidate_samples in samples):
            raise ValueError(f"context {context_id} does not have the required samples for every candidate")
        if any(configuration is None for configuration in configurations):
            raise ValueError(f"context {context_id} has an unmeasured candidate configuration")
        if any(not isinstance(estimate, (int, float)) or not math.isfinite(estimate) or estimate <= 0 for estimate in estimates):
            raise ValueError(f"context {context_id} has an invalid candidate estimate")
        if not isinstance(best, int) or not 0 <= best < candidate_count:
            raise ValueError(f"context {context_id} has no valid winning candidate")
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            for candidate_samples in samples
            for value in candidate_samples
        ):
            raise ValueError(f"context {context_id} has invalid runtime samples")
        if not isinstance(estimates[best], (int, float)) or not math.isfinite(estimates[best]) or estimates[best] <= 0:
            raise ValueError(f"context {context_id} winner has an invalid runtime estimate")
        if estimates[best] != min(estimates):
            raise ValueError(f"context {context_id} best candidate does not have the minimum estimate")

        selected_frames = scalar_vector(configurations[best]["numFrames"], "numFrames")
        selected_extent = scalar_vector(configurations[best]["frameExtent"], "frameExtent")
        winners.append(
            {
                "context_id": context_id,
                "kernel": kernel,
                "executor": launch.group(3),
                "original_num_frames": original_frames,
                "original_frame_extent": original_extent,
                "selected_num_frames": selected_frames,
                "selected_frame_extent": selected_extent,
                "candidate_count": candidate_count,
                "measured_candidate_count": sum(bool(value) for value in samples),
                "winner_candidate_index": best,
                "winner_estimate_seconds": estimates[best],
                "winner_sample_count": len(samples[best]),
            }
        )

    if seen != EXPECTED_CONTEXTS:
        missing = sorted(EXPECTED_CONTEXTS - seen)
        unexpected = sorted(seen - EXPECTED_CONTEXTS)
        raise ValueError(f"history context set mismatch; missing={missing}, unexpected={unexpected}")
    document = {
        "schema_version": 1,
        "purpose": "static FrameSpecs selected by random-search discovery; physical worker coverage may differ",
        "source_history": str(args.history),
        "source_history_sha256": hashlib.sha256(raw).hexdigest(),
        "contexts": winners,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
