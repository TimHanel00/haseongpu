# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from HASEonGPU import (
    CrossSectionData,
    GainMedium,
    PhiASE,
    VolumeTopology,
    backendFlat,
)


FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "juliaASE"
    / "reflection_surface_reference"
)
REFERENCE_PATH = FIXTURE_DIR / "reference.json"


def _mediumFromReference(reference):
    topology = VolumeTopology.fromTetrahedra(
        np.asarray(reference["points"], dtype=np.float64),
        np.asarray(reference["cells"], dtype=np.uint32),
        faceBoundaries=np.asarray(reference["faceBoundaries"], dtype=np.int32),
        metadata={
            "source": str(FIXTURE_DIR / reference["meshFile"]),
            "format": "gmsh",
        },
    )
    material = reference["material"]
    legacy = reference["legacyOpticsFallback"]
    optics = reference["surfaceOptics"]
    return GainMedium(topology).withPhysicalProperties(
        betaVolume=backendFlat(
            np.asarray(reference["initialBetaVolume"], dtype=np.float64)
        ),
        betaCells=backendFlat(np.asarray(reference["betaCells"], dtype=np.float64)),
        claddingCellTypes=np.asarray(
            reference["claddingCellTypes"], dtype=np.uint32
        ),
        refractiveIndices=np.asarray(
            legacy["refractiveIndices"], dtype=np.float32
        ),
        reflectivities=backendFlat(
            np.asarray(legacy["reflectivities"], dtype=np.float32)
        ),
        surfaceReflectivity=np.asarray(
            optics["surfaceReflectivity"], dtype=np.float32
        ),
        surfaceRefractiveIndexInside=np.asarray(
            optics["surfaceRefractiveIndexInside"], dtype=np.float32
        ),
        surfaceRefractiveIndexOutside=np.asarray(
            optics["surfaceRefractiveIndexOutside"], dtype=np.float32
        ),
        nTot=float(material["nTot"]),
        crystalTFluo=float(material["crystalTFluo"]),
        claddingNumber=int(material["claddingNumber"]),
        claddingAbsorption=float(material["claddingAbsorption"]),
    )


def _crossSectionsFromReference(reference):
    crossSections = reference["crossSections"]
    return CrossSectionData(
        wavelengthsAbsorption=np.asarray(
            crossSections["wavelengthsAbsorption"], dtype=np.float64
        ),
        crossSectionAbsorption=np.asarray(
            crossSections["crossSectionAbsorption"], dtype=np.float64
        ),
        wavelengthsEmission=np.asarray(
            crossSections["wavelengthsEmission"], dtype=np.float64
        ),
        crossSectionEmission=np.asarray(
            crossSections["crossSectionEmission"], dtype=np.float64
        ),
        resolution=int(crossSections["resolution"]),
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("useReflections", "resultName"),
    ((False, "withoutReflections"), (True, "withReflections")),
    ids=("without-reflections", "with-reflections"),
)
def test_haseForwardMatchesCommittedJuliaaseSurfaceFixture(
    monkeypatch,
    openPmdRuntimeBackend,
    openPmdRuntimeExecutable,
    alpakaRuntimeBackend,
    useReflections,
    resultName,
):
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    monkeypatch.setenv("HASE_CALCPHIASE", str(openPmdRuntimeExecutable))

    phiAse = PhiASE(
        crossSections=_crossSectionsFromReference(reference),
        propagationMode="forward",
        minRays=int(reference["rayCount"]),
        maxRays=int(reference["rayCount"]),
        forwardRayCount=int(reference["rayCount"]),
        relativeStandardErrorThreshold=1.0,
        repetitions=1,
        adaptiveSteps=1,
        useReflections=useReflections,
        reflectionMaxIterations=int(reference["reflectionMaxIterations"]),
        reflectionTolerance=float(reference["reflectionTolerance"]),
        surfaceReservoirSize=int(reference["surfaceReservoirSize"]),
        monochromatic=True,
        backend=alpakaRuntimeBackend,
        openpmdBackend=openPmdRuntimeBackend,
        parallelMode="single",
        numDevices=1,
        rngSeed=int(reference["seed"]),
    )

    phiAse.run(gainMedium=_mediumFromReference(reference))

    result = phiAse.getResults()
    actualPhiAse = np.asarray(result.phiAse, dtype=np.float64)
    actualDndtAse = np.asarray(result.dndtAse, dtype=np.float64)
    actualFinalBeta = np.asarray(
        reference["initialBetaVolume"], dtype=np.float64
    ) - float(reference["timeStep"]) * actualDndtAse
    expected = reference["aseResults"][resultName]
    tolerances = reference["tolerances"]

    if useReflections:
        assert result.srmStatus == "converged"
        assert result.srmPasses == 1
    else:
        assert result.srmStatus == "disabled"
        assert result.srmPasses == 0

    np.testing.assert_allclose(
        actualPhiAse,
        np.asarray(expected["phiAse"], dtype=np.float64),
        rtol=float(tolerances["phiAseRtol"]),
        atol=float(tolerances["phiAseAtol"]),
    )
    np.testing.assert_allclose(
        actualDndtAse,
        np.asarray(expected["dndtAse"], dtype=np.float64),
        rtol=float(tolerances["phiAseRtol"]),
        atol=float(tolerances["phiAseAtol"]),
    )
    np.testing.assert_allclose(
        actualFinalBeta,
        np.asarray(expected["finalBetaVolume"], dtype=np.float64),
        rtol=float(tolerances["betaVolumeRtol"]),
        atol=float(tolerances["betaVolumeAtol"]),
    )
