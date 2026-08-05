# Copyright 2026 HASEonGPU contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import hashlib
import json
import math
import struct
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = (
    REPO_ROOT / "tests" / "data" / "juliaASE" / "reflection_surface_reference"
)
REFERENCE_PATH = FIXTURE_ROOT / "reference.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def test_juliaase_reflection_fixture_records_reproducible_generator():
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    generation = reference["referenceGeneration"]
    repository = generation["repository"]
    environment = generation["environment"]
    driver = generation["driver"]

    assert repository == {
        "url": "https://codebase.helmholtz.cloud/penelope-julia/julia_ase.git",
        "visibility": "private",
        "accessContact": "Daniel Albach",
        "commit": "f6e19290ac06f7fd6f9492e6bb14a86973a50166",
        "dirty": False,
    }
    assert generation["result"]["withoutReflections"] == {
        "status": "max_passes",
        "passes": 0,
    }
    assert generation["result"]["withReflections"] == {
        "status": "converged",
        "passes": 1,
    }

    manifest_path = REPO_ROOT / environment["manifestFile"]
    wrapper_path = REPO_ROOT / driver["wrapperPath"]
    assert driver["kind"] == "embeddedJuliaSource"
    assert (
        driver["sha256"]
        == "ecaddde49a66b9c59b26f79243b2842f2a27e5ed826b26ba784e3c4d8b8921d0"
    )
    assert sha256(manifest_path) == environment["manifestSha256"]
    assert sha256(wrapper_path) == driver["wrapperSha256"]

    with manifest_path.open("rb") as stream:
        manifest = tomllib.load(stream)
    assert manifest["julia_version"] == environment["juliaVersion"] == "1.10.9"


def test_juliaase_reflection_fixture_parameters_and_derived_values_are_consistent():
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    assert reference["schemaVersion"] == 2
    assert set(reference["aseResults"]) == {
        "withoutReflections",
        "withReflections",
    }
    generation = reference["referenceGeneration"]
    parameters = generation["parameters"]

    assert parameters["rayCount"] == reference["referenceRayCount"]
    assert parameters["seed"] == reference["seed"]
    assert parameters["maxPasses"] == reference["reflectionMaxIterations"]
    assert parameters["epsilon"] == reference["reflectionTolerance"]
    assert parameters["srmReservoirSize"] == reference["surfaceReservoirSize"]

    beta_float32 = float32(0.18)
    absorption_float32 = float32(0.01)
    emission_float32 = float32(0.02)
    gain_float32 = float32(
        float32(beta_float32 * float32(absorption_float32 + emission_float32))
        - absorption_float32
    )
    assert parameters["gainFloat32"] == gain_float32
    assert parameters["sourceRateTotalFloat64"] == beta_float32 / 6.0

    beta = reference["initialBetaVolume"][0]
    absorption = max(reference["crossSections"]["crossSectionAbsorption"])
    emission = max(reference["crossSections"]["crossSectionEmission"])
    for result in reference["aseResults"].values():
        expected_dndt = (
            beta * (emission + absorption) - absorption
        ) * result["phiAse"][0]
        assert math.isclose(
            result["dndtAse"][0], expected_dndt, rel_tol=0.0, abs_tol=0.0
        )
        expected_final_beta = beta - reference["timeStep"] * expected_dndt
        assert math.isclose(
            result["finalBetaVolume"][0],
            expected_final_beta,
            rel_tol=0.0,
            abs_tol=0.0,
        )
