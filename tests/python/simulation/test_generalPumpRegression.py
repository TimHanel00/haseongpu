# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3] / "example"))

import numpy as np
import pytest

import laserPumpCladdingApi as example


LEGACY_VOLUME_UNIT_SI = 1.0e-6


def _normalized_wasserstein_distance(first_coordinates, first_weights, second_coordinates, second_weights):
    first_coordinates = np.asarray(first_coordinates, dtype=np.float64).reshape(-1)
    second_coordinates = np.asarray(second_coordinates, dtype=np.float64).reshape(-1)
    first_weights = np.asarray(first_weights, dtype=np.float64).reshape(-1)
    second_weights = np.asarray(second_weights, dtype=np.float64).reshape(-1)
    first_weights = first_weights / np.sum(first_weights)
    second_weights = second_weights / np.sum(second_weights)

    coordinates = np.unique(np.concatenate((first_coordinates, second_coordinates)))
    first_order = np.argsort(first_coordinates)
    second_order = np.argsort(second_coordinates)
    first_cdf = np.concatenate(([0.0], np.cumsum(first_weights[first_order])))
    second_cdf = np.concatenate(([0.0], np.cumsum(second_weights[second_order])))
    first_indices = np.searchsorted(first_coordinates[first_order], coordinates[:-1], side="right")
    second_indices = np.searchsorted(second_coordinates[second_order], coordinates[:-1], side="right")
    distance = np.sum(
        np.abs(first_cdf[first_indices] - second_cdf[second_indices]) * np.diff(coordinates)
    )
    coordinate_span = coordinates[-1] - coordinates[0]
    return float(distance / coordinate_span) if coordinate_span > 0.0 else 0.0


def _deposition_diagnostics(topology, states, reference, legacy_lumped_volume):
    point_coordinates = np.asarray(topology.points, dtype=np.float64)
    cell_coordinates = np.asarray(topology.cellCenters, dtype=np.float64)
    cell_volumes = np.asarray(topology.cellVolumes, dtype=np.float64)
    point_volumes = np.asarray(legacy_lumped_volume, dtype=np.float64).reshape(-1, order="F")
    point_radius = np.linalg.norm(point_coordinates[:, :2], axis=1)
    cell_radius = np.linalg.norm(cell_coordinates[:, :2], axis=1)
    diagnostics = []
    for step, (state, legacy_rate) in enumerate(zip(states, reference["dndtPump"], strict=True), start=1):
        cell_measure = np.asarray(state.dndtPump, dtype=np.float64).reshape(-1) * cell_volumes
        point_measure = np.asarray(legacy_rate, dtype=np.float64).reshape(-1, order="F") * point_volumes
        cell_total = np.sum(cell_measure)
        point_total = np.sum(point_measure)
        diagnostics.append(
            {
                "step": step,
                "total_relative_error": float(abs(cell_total - point_total) / point_total),
                "axial_profile_distance": _normalized_wasserstein_distance(
                    cell_coordinates[:, 2], cell_measure, point_coordinates[:, 2], point_measure
                ),
                "radial_profile_distance": _normalized_wasserstein_distance(
                    cell_radius, cell_measure, point_radius, point_measure
                ),
                "cell_mean_z": float(np.sum(cell_coordinates[:, 2] * cell_measure) / cell_total),
                "legacy_mean_z": float(np.sum(point_coordinates[:, 2] * point_measure) / point_total),
                "cell_mean_radius": float(np.sum(cell_radius * cell_measure) / cell_total),
                "legacy_mean_radius": float(np.sum(point_radius * point_measure) / point_total),
            }
        )
    return diagnostics


@pytest.mark.integration
def test_general_pump_reproduces_legacy_crystal_inversion(openPmdFileBackend, alpakaRuntimeBackend):
    reference = np.load(
        Path(__file__).parents[2] / "data" / "pump" / "legacy_one_dimensional_reference.npz"
    )
    simulation = example.buildSimulation(
        backend=alpakaRuntimeBackend,
        openpmdBackend=openPmdFileBackend,
        simulationSteps=3,
        pumpSteps=3,
        aseSteps=0,
        spectralResolution=191,
        pumpRayCount=50_000,
        pumpRngSeed=5489,
        useCladding=False,
    )
    states = []
    simulation.onStep(states.append).step(3)
    topology = simulation._simulationState.topology

    beta_volume = np.stack([np.asarray(state.betaVolume) for state in states])
    relative_field_error = np.linalg.norm(beta_volume - reference["betaVolume"]) / np.linalg.norm(
        reference["betaVolume"]
    )

    cell_volumes = np.asarray(topology.cellVolumes)
    legacy_lumped_volume_cm3 = np.bincount(
        np.asarray(topology.cellPointIndices).reshape(-1),
        weights=np.repeat(cell_volumes / LEGACY_VOLUME_UNIT_SI / 4.0, 4),
        minlength=topology.numberOfPoints,
    ).reshape(reference["dndtPump"].shape[1:], order="F")
    new_total = np.asarray([np.sum(np.asarray(state.dndtPump) * cell_volumes) for state in states])
    legacy_total_m3 = LEGACY_VOLUME_UNIT_SI * np.asarray(
        [np.sum(values * legacy_lumped_volume_cm3) for values in reference["dndtPump"]]
    )
    np.testing.assert_allclose(new_total, legacy_total_m3, rtol=0.01, atol=1e-12)
    diagnostics = _deposition_diagnostics(topology, states, reference, legacy_lumped_volume_cm3)
    assert relative_field_error < 0.05, f"deposition diagnostics: {diagnostics}"
