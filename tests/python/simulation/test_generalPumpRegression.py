# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path
import numpy as np
import pytest
from example import laserPumpCladding as example
from pyInclude.geometry.vtk import _parseVtk


@pytest.mark.integration
def test_general_pump_reproduces_legacy_crystal_inversion(
    openPmdFileBackend,
    alpakaRuntimeBackend,
    tmp_path,
):
    reference = np.load(
        Path(__file__).parents[2] / "data" / "pump" / "legacy_one_dimensional_reference.npz"
    )
    example.run_example(
        backend=alpakaRuntimeBackend,
        openPmdBackend=openPmdFileBackend,
        timeSteps=3,
        pumpSteps=3,
        vtkOutputDir=tmp_path,
        enableAse=False,
        prePump=True,
        spectralResolution=191,
    )

    snapshots = [
        _parseVtk(tmp_path / f"laserPumpCladding_{step:03d}.vtk")
        for step in range(1, 4)
    ]

    beta_volume = np.stack([cell_data["betaVolume"] for *_, cell_data, _fields in snapshots])
    relative_field_error = np.linalg.norm(beta_volume - reference["betaVolume"]) / np.linalg.norm(
        reference["betaVolume"]
    )
    assert relative_field_error < 0.05

    topology = example.laser_pump_cladding_mesh()
    cell_points = np.asarray(topology.cellPointIndices).reshape(-1)
    lumped_volume = np.bincount(
        cell_points,
        weights=np.repeat(np.asarray(topology.cellVolumes) / 4.0, 4),
        minlength=topology.numberOfSamplePoints,
    )
    new_total = np.asarray(
        [np.sum(point_data["dndtPump"] * lumped_volume) for *_, point_data, _cell_data, _fields in snapshots]
    )
    old_total = np.asarray([np.sum(values * lumped_volume) for values in reference["dndtPump"]])
    np.testing.assert_allclose(new_total, old_total, rtol=0.01, atol=1e-12)
