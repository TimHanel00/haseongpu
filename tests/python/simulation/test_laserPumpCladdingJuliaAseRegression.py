# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Five-step laserPumpCladding regression against pinned JuliaASE results."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pyInclude.geometry.vtk import _parseVtk


repoRoot = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repoRoot / "example"))
import laserPumpCladding  # noqa: E402
sys.path.insert(0, str(repoRoot / "scripts"))
import regenerate_juliaase_laser_pump_cladding_fixture as fixtureGenerator  # noqa: E402


REFERENCE_PATH = (
    repoRoot
    / "tests"
    / "data"
    / "juliaASE"
    / "laser_pump_cladding_5_step_reference"
    / "reference.json"
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256Text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def testJuliaAseFiveStepReferenceRecordsGeneratorAndBothModes():
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    generation = reference["referenceGeneration"]
    generator = generation["generator"]
    assert reference["schema"] == "hase.juliaASE.laserPumpCladdingFiveStep.v1"
    assert generation["repository"]["commit"] == (
        "f6e19290ac06f7fd6f9492e6bb14a86973a50166"
    )
    assert generator["kind"] == "embeddedJuliaSource"
    assert _sha256Text(fixtureGenerator.JULIA_DRIVER_SOURCE) == generator["sha256"]
    assert _sha256(repoRoot / generator["wrapperPath"]) == generator["wrapperSha256"]
    assert set(reference["aseResults"]) == {
        "withoutReflections",
        "withReflections",
    }
    assert reference["tolerances"]["relative"] == 0.05
    for result in reference["aseResults"].values():
        rows = result["simulationSteps"]
        assert [row["step"] for row in rows] == [1, 2, 3, 4, 5]
        np.testing.assert_allclose(
            [row["time"] for row in rows],
            np.arange(1, 6) * reference["parameters"]["timeStepSeconds"],
            rtol=0.0,
            atol=1.0e-19,
        )
        assert rows[0]["phiAseVolumeIntegral"] == 0.0
        assert rows[0]["aseStatus"] == "prePump"
        assert rows[0]["asePasses"] == 0
        assert all(row["phiAseVolumeIntegral"] > 0.0 for row in rows[1:])
        assert all(row["aseStatus"] == "converged" for row in rows[1:])

    withoutReflections = reference["aseResults"]["withoutReflections"][
        "simulationSteps"
    ]
    withReflections = reference["aseResults"]["withReflections"]["simulationSteps"]
    assert all(
        reflected["phiAseVolumeIntegral"] > direct["phiAseVolumeIntegral"]
        for direct, reflected in zip(
            withoutReflections[1:], withReflections[1:], strict=True
        )
    )


def _tetVolumes(points, cells):
    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.uint32)
    a = points[cells[:, 0]]
    b = points[cells[:, 1]]
    c = points[cells[:, 2]]
    d = points[cells[:, 3]]
    return np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6.0


def _readStepObservables(path, *, allowMissingPhiAse=False):
    points, cells, _cellTypes, _pointData, cellData, _fields = _parseVtk(path)
    volumes = _tetVolumes(points, cells)
    beta = np.asarray(cellData["betaVolume"], dtype=np.float64)
    if allowMissingPhiAse and "phiASE" not in cellData:
        phi = np.zeros_like(beta)
    else:
        phi = np.asarray(cellData["phiASE"], dtype=np.float64)
    return {
        "betaVolumeMean": float(np.sum(beta * volumes) / np.sum(volumes)),
        # HASE's legacy example geometry uses centimetres.  Its PhiASE volume
        # integral is photons*cm/s; JuliaASE's SI mesh produces photons*m/s.
        "phiAseVolumeIntegral": float(np.sum(phi * volumes) * 1.0e-2),
    }


@pytest.mark.integration
@pytest.mark.parametrize(
    ("useReflections", "resultName"),
    ((False, "withoutReflections"), (True, "withReflections")),
    ids=("without-reflections", "with-reflections"),
)
def testLaserPumpCladdingFiveStepsMatchJuliaAse(
    tmp_path,
    monkeypatch,
    openPmdRuntimeBackend,
    openPmdRuntimeExecutable,
    alpakaRuntimeBackend,
    useReflections,
    resultName,
):
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    parameters = reference["parameters"]
    assert parameters["simulationSteps"] == 5
    monkeypatch.setenv("HASE_CALCPHIASE", str(openPmdRuntimeExecutable))

    outputDir = tmp_path / resultName
    capturedStates = []
    originalWriteVtkFields = laserPumpCladding.writeVtkFields

    def captureState(state, *args, **kwargs):
        capturedStates.append(state)
        return originalWriteVtkFields(state, *args, **kwargs)

    monkeypatch.setattr(laserPumpCladding, "writeVtkFields", captureState)
    finalState = laserPumpCladding.runExample(
        backend=alpakaRuntimeBackend,
        openpmdBackend=openPmdRuntimeBackend,
        timeSlices=parameters["simulationSteps"],
        # Fifty is the example's physical pump duration and also keeps the
        # pump active throughout this deliberately short five-step prefix.
        pumpSteps=parameters["pumpConfiguredSteps"],
        vtkOutputDir=outputDir,
        enableASE=True,
        prePump=parameters["prePump"],
        spectralResolution=parameters["spectralResolution"],
        pumpRayCount=parameters["pumpRayCount"],
        pumpRngSeed=parameters["pumpRngSeed"],
        rngSeed=parameters["aseRngSeed"],
        useReflections=useReflections,
        minRays=parameters["aseRayCount"],
        maxRays=parameters["aseRayCount"],
        repetitions=1,
        adaptiveSteps=1,
        relativeStandardErrorThreshold=1.0,
        reflectionMaxIterations=parameters["reflectionMaxIterations"],
        reflectionTolerance=parameters["reflectionTolerance"],
        surfaceReservoirSize=parameters["surfaceReservoirSize"],
    )

    expectedRows = reference["aseResults"][resultName]["simulationSteps"]
    actualRows = [
        _readStepObservables(
            outputDir / f"laserPumpCladding_{step:03d}.vtk",
            allowMissingPhiAse=step == 1,
        )
        for step in range(1, parameters["simulationSteps"] + 1)
    ]
    assert [row["step"] for row in expectedRows] == [1, 2, 3, 4, 5]
    assert [state.step for state in capturedStates] == [1, 2, 3, 4, 5]
    assert finalState.step == 5
    if useReflections:
        srmResults = [
            (state.aseResult.srmStatus, state.aseResult.srmPasses)
            for state in capturedStates[1:]
        ]
        expectedSrmStatuses = [row["aseStatus"] for row in expectedRows[1:]]
        assert [status for status, _passes in srmResults] == expectedSrmStatuses
        assert all(
            0 < passes <= parameters["reflectionMaxIterations"]
            for _status, passes in srmResults
        ), srmResults
    else:
        assert all(
            state.aseResult.srmStatus == "disabled"
            and state.aseResult.srmPasses == 0
            for state in capturedStates
        )

    relativeTolerance = reference["tolerances"]["relative"]
    np.testing.assert_allclose(
        [row["betaVolumeMean"] for row in actualRows],
        [row["betaVolumeMean"] for row in expectedRows],
        rtol=relativeTolerance,
        atol=reference["tolerances"]["betaVolumeMeanAbsolute"],
    )
    # Step one intentionally has no ASE because prePump=true. Comparing it too
    # ensures a pre-pump regression cannot be hidden by slicing it away.
    np.testing.assert_allclose(
        [row["phiAseVolumeIntegral"] for row in actualRows],
        [row["phiAseVolumeIntegral"] for row in expectedRows],
        rtol=relativeTolerance,
        atol=reference["tolerances"]["phiAseVolumeIntegralAbsolute"],
    )
