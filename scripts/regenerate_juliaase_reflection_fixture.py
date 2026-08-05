#!/usr/bin/env python3
"""Regenerate the committed JuliaASE reflection-surface fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "data" / "juliaASE" / "reflection_surface_reference"
REFERENCE_PATH = FIXTURE_DIR / "reference.json"
JULIAASE_MANIFEST = FIXTURE_DIR / "juliaase-manifest.toml"

JULIA_DRIVER_SOURCE = r'''
using Random

const juliaase_root = length(ARGS) == 1 ? ARGS[1] : error("usage: embedded JuliaASE driver JULIAASE_ROOT")
const ray_count = parse(Int, get(ENV, "HASE_JULIAASE_FIXTURE_RAYS", "4096"))
const max_reflection_passes = parse(Int, get(ENV, "HASE_JULIAASE_FIXTURE_MAX_PASSES", "4"))
include(joinpath(juliaase_root, "src", "ForwardASE.jl"))

const T = ForwardASE.Types
const S = ForwardASE.SRM
const G = ForwardASE.GPUTransfer
const Sim = ForwardASE.Simulation
const Tallies = ForwardASE.Tallies

const points = Float64[
    0.0 1.0 0.0 0.0
    0.0 0.0 1.0 0.0
    0.0 0.0 0.0 1.0
]
const connectivity = reshape(Int32[1, 2, 3, 4], 4, 1)
const face_normals = cat(
    reshape(Float32[inv(sqrt(3.0)), inv(sqrt(3.0)), inv(sqrt(3.0))], 3, 1),
    reshape(Float32[-1.0, 0.0, 0.0], 3, 1),
    reshape(Float32[0.0, -1.0, 0.0], 3, 1),
    reshape(Float32[0.0, 0.0, -1.0], 3, 1);
    dims = 2,
)
const mesh = T.TetMesh(
    points,
    Int8[T.TET4],
    connectivity,
    zeros(Int32, 6, 1),
    Int32[1],
    Float32[1.0 / 6.0],
    reshape(Float32[0.25, 0.25, 0.25], 3, 1),
    reshape(face_normals, 3, 4, 1),
    reshape(Float32[sqrt(3.0) / 2.0, 0.5, 0.5, 0.5], 4, 1),
    fill(Int32(-1), 4, 1),
    reshape(Int32[T.BOUND_STOP, T.BOUND_STOP, T.BOUND_STOP, 11], 4, 1),
    fill(NaN, 3, 3, 1),
)

const boundary_faces = T.BoundaryFaceList(
    Int32[1, 1, 1, 1],
    Int8[1, 2, 3, 4],
    Int32[T.BOUND_STOP, T.BOUND_STOP, T.BOUND_STOP, 11],
    face_normals,
    Float32[sqrt(3.0) / 2.0, 0.5, 0.5, 0.5],
    Float32[
        1.0 / 3.0 0.0 1.0 / 3.0 1.0 / 3.0
        1.0 / 3.0 1.0 / 3.0 0.0 1.0 / 3.0
        1.0 / 3.0 1.0 / 3.0 1.0 / 3.0 0.0
    ],
    Int32[0, 0, 0, 1],
)

const coating = T.CoatingTable(
    "hase-surface-reflectivity-0.65",
    Float32[0.0, 90.0],
    Float32[1030.0, 1030.1],
    fill(0.65f0, 2, 2),
    fill(0.65f0, 2, 2),
    fill(0.35f0, 2, 2),
    fill(0.35f0, 2, 2),
)

const beta = 0.18f0
const sigma_absorption = 0.01f0
const sigma_emission = 0.02f0
const gain = beta * (sigma_absorption + sigma_emission) - sigma_absorption
const source_rate_total = Float64(beta) / 6.0
const state = G.init_simulation_state(
    mesh,
    boundary_faces,
    S.init_srm(length(boundary_faces.tet_ind), 64),
    T.TIER3,
    Float32[gain],
    zeros(Float32, 3, 1),
    zeros(Float32, 1),
    fill(beta, 4),
    source_rate_total;
    domain_n = Float32[1.5],
    bfl_outside_n = Float32[1.0, 1.0, 1.0, 1.0],
    coating_tables = T.CoatingTable[coating],
    face_coating_ind = Int32[0, 0, 0, 1],
    track_polarization = false,
)

const run = Sim.run_passes!(
    state,
    ray_count,
    trues(1),
    1.0,
    1030.0f0,
    MersenneTwister(12345);
    epsilon = 1.0e-5,
    max_passes = max_reflection_passes,
    diverge_streak = 3,
    nthreads = 1,
    n_chunks = 1,
)
const phi_ase = Tallies.compute_phi_ase(state, ray_count)

function json_array(values)
    return "[" * join(string.(Float64.(values)), ",") * "]"
end

println("{\"status\":\"", run.status,
        "\",\"passes\":", run.n_passes,
        ",\"rayCount\":", ray_count,
        ",\"phiAse\":", json_array(phi_ase),
        ",\"initialReflectedWeight\":", Float64(sum(run.srm_W_cumulative)),
        ",\"reflectedPassWeightFractions\":", json_array(run.W_fracs),
        "}")
'''.strip()

JULIAASE_REPOSITORY = "https://codebase.helmholtz.cloud/penelope-julia/julia_ase.git"
JULIAASE_COMMIT = "f6e19290ac06f7fd6f9492e6bb14a86973a50166"
JULIAASE_PROJECT_SHA256 = (
    "8e79c420e591978d179233554597cbaa896debd18baf119c4de325b71695f7d7"
)
JULIAASE_MANIFEST_SHA256 = (
    "6bc0d15a228289b5c6f6c2b50e7ed6e1b862729a2ac23657dccecfab8251347a"
)
HISTORICAL_DRIVER_COMMIT = "fa15f8e6d980bf2442635376b41ae51a6441d57c"
JULIA_VERSION = "1.10.9"
JULIA_DISTRIBUTION_URL = (
    "https://julialang-s3.julialang.org/bin/linux/x64/1.10/"
    "julia-1.10.9-linux-x86_64.tar.gz"
)
JULIA_DISTRIBUTION_SHA256 = (
    "5a2d2c5224594b683c97e7304cb72407fbcf0be4a0187789cba1a2f73f0cbf09"
)
JULIA_EXECUTABLE_SHA256 = (
    "1436babaecb80defc11086ab2587c8dbfda4705cbe6608aa21bc9ddccd15e60f"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def juliaase_root() -> Path:
    configured = os.environ.get("JULIAASE_ROOT")
    root = (
        Path(configured).expanduser().resolve()
        if configured
        else REPO_ROOT.parent / "juliaASE"
    )
    if not (root / "Project.toml").is_file():
        raise SystemExit(
            "JuliaASE checkout not found. Set JULIAASE_ROOT to the private checkout; "
            "contact Daniel Albach if access is required."
        )
    return root


def git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_juliaase_checkout(root: Path) -> None:
    revision = git_output(root, "rev-parse", "HEAD")
    if revision != JULIAASE_COMMIT:
        raise SystemExit(
            f"JuliaASE must be checked out at {JULIAASE_COMMIT}, got {revision}"
        )
    status = git_output(root, "status", "--porcelain")
    if status:
        raise SystemExit(
            f"JuliaASE checkout must be clean; git status reported:\n{status}"
        )
    project_hash = sha256(root / "Project.toml")
    if project_hash != JULIAASE_PROJECT_SHA256:
        raise SystemExit(
            f"JuliaASE Project.toml hash mismatch: expected {JULIAASE_PROJECT_SHA256}, "
            f"got {project_hash}"
        )


def find_julia() -> tuple[str, str]:
    julia = shutil.which("julia")
    if julia is None:
        raise SystemExit(f"Julia {JULIA_VERSION} was not found on PATH")
    version = subprocess.run(
        [julia, "--startup-file=no", "-e", "print(VERSION)"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if version != JULIA_VERSION:
        raise SystemExit(f"Julia {JULIA_VERSION} is required, got {version}")
    executable_hash = sha256(Path(julia).resolve())
    if executable_hash != JULIA_EXECUTABLE_SHA256:
        raise SystemExit(
            f"Julia executable hash mismatch: expected {JULIA_EXECUTABLE_SHA256}, "
            f"got {executable_hash}"
        )
    return julia, version


def exact_environment() -> dict[str, str]:
    return dict(
        os.environ,
        HASE_JULIAASE_FIXTURE_MAX_PASSES="4",
        JULIA_NUM_THREADS="1",
        JULIA_LOAD_PATH="@:@stdlib",
        JULIA_PKG_PRECOMPILE_AUTO="0",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )


def prepare_project(
    root: Path, project: Path, julia: str, environment: dict[str, str]
) -> None:
    manifest_hash = sha256(JULIAASE_MANIFEST)
    if manifest_hash != JULIAASE_MANIFEST_SHA256:
        raise SystemExit(
            f"JuliaASE manifest hash mismatch: expected {JULIAASE_MANIFEST_SHA256}, "
            f"got {manifest_hash}"
        )
    shutil.copyfile(root / "Project.toml", project / "Project.toml")
    shutil.copyfile(JULIAASE_MANIFEST, project / "Manifest.toml")
    subprocess.run(
        [
            julia,
            "--startup-file=no",
            f"--project={project}",
            "-e",
            "using Pkg; Pkg.instantiate()",
        ],
        check=True,
        cwd=root,
        env=environment,
    )
    instantiated_hash = sha256(project / "Manifest.toml")
    if instantiated_hash != JULIAASE_MANIFEST_SHA256:
        raise SystemExit(
            "Pkg.instantiate() changed the pinned JuliaASE manifest: "
            f"expected {JULIAASE_MANIFEST_SHA256}, got {instantiated_hash}"
        )


def generate_reference(
    root: Path,
    project: Path,
    julia: str,
    ray_count: int,
    environment: dict[str, str],
) -> tuple[dict, str]:
    environment = dict(environment, HASE_JULIAASE_FIXTURE_RAYS=str(ray_count))
    completed = subprocess.run(
        [
            julia,
            "--startup-file=no",
            f"--project={project}",
            "-e",
            JULIA_DRIVER_SOURCE,
            str(root),
        ],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=environment,
    )
    try:
        generated = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"JuliaASE fixture driver produced invalid JSON:\n{completed.stdout}"
        ) from exc
    if generated["status"] != "converged":
        raise SystemExit(f"JuliaASE fixture did not converge: {generated['status']}")
    normalized_stderr = completed.stderr.replace(str(root), "$JULIAASE_ROOT").replace(
        str(REPO_ROOT), "$HASE_ROOT"
    )
    return generated, normalized_stderr


def simulation_parameters(ray_count: int) -> dict:
    return {
        "rayCount": ray_count,
        "randomNumberGenerator": "Random.MersenneTwister",
        "seed": 12345,
        "sourceMask": [True],
        "radiativeLifetime": 1.0,
        "wavelengthNm": 1030.0,
        "epsilon": 1e-5,
        "maxPasses": 4,
        "divergeStreak": 3,
        "maxSteps": 10_000,
        "nthreads": 1,
        "nChunks": 1,
        "useGpu": False,
        "gpuTier": "TIER3",
        "srmReservoirSize": 64,
        "trackPolarization": False,
        "stateScalarType": "Float32",
        "betaLiteral": "0.18f0",
        "sigmaAbsorptionForGainLiteral": "0.01f0",
        "sigmaEmissionForGainLiteral": "0.02f0",
        "gainFloat32": -0.004599999636411667,
        "sourceRateTotalFloat64": 0.030000001192092896,
        "domainRefractiveIndex": [1.5],
        "outsideRefractiveIndex": [1.0, 1.0, 1.0, 1.0],
        "faceCoatingIndex": [0, 0, 0, 1],
        "coating": {
            "scalarType": "Float32",
            "angleDegrees": [0.0, 90.0],
            "wavelengthNm": [1030.0, 1030.1],
            "reflectivityS": 0.65,
            "reflectivityP": 0.65,
            "transmissivityS": 0.35,
            "transmissivityP": 0.35,
        },
    }


def generation_metadata(
    root: Path,
    julia: str,
    julia_version: str,
    ray_count: int,
    generated: dict,
    driver_stderr: str,
) -> dict:
    return {
        "implementation": "JuliaASE",
        "generatedAtUtc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "repository": {
            "url": JULIAASE_REPOSITORY,
            "visibility": "private",
            "accessContact": "Daniel Albach",
            "commit": git_output(root, "rev-parse", "HEAD"),
            "dirty": bool(git_output(root, "status", "--porcelain")),
        },
        "driver": {
            "kind": "embeddedJuliaSource",
            "sha256": hashlib.sha256(JULIA_DRIVER_SOURCE.encode("utf-8")).hexdigest(),
            "historicalCommit": HISTORICAL_DRIVER_COMMIT,
            "wrapperPath": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "wrapperSha256": sha256(Path(__file__).resolve()),
        },
        "command": (
            "JULIAASE_ROOT=/path/to/juliaASE JULIA_NUM_THREADS=1 "
            "JULIA_LOAD_PATH=@:@stdlib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "python3 scripts/regenerate_juliaase_reflection_fixture.py "
            f"--ray-count {ray_count}"
        ),
        "parameters": simulation_parameters(ray_count),
        "environment": {
            "juliaVersion": julia_version,
            "juliaExecutableSha256": sha256(Path(julia).resolve()),
            "juliaDistributionUrl": JULIA_DISTRIBUTION_URL,
            "juliaDistributionSha256": JULIA_DISTRIBUTION_SHA256,
            "projectSha256": JULIAASE_PROJECT_SHA256,
            "manifestFile": str(JULIAASE_MANIFEST.relative_to(REPO_ROOT)),
            "manifestSha256": sha256(JULIAASE_MANIFEST),
            "pythonVersion": platform.python_version(),
            "numpyVersion": np.__version__,
            "operatingSystem": platform.freedesktop_os_release()["PRETTY_NAME"],
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "juliaNumThreads": 1,
            "juliaLoadPath": "@:@stdlib",
            "ompNumThreads": 1,
            "openblasNumThreads": 1,
            "startupFile": "disabled",
        },
        "result": {
            "status": generated["status"],
            "passes": generated["passes"],
            "driverStderr": driver_stderr,
        },
        "postprocessing": {
            "dndtConvention": "HASE/openPMD",
            "dndtAseFormula": (
                "(beta * (sigmaEmission + sigmaAbsorption) - sigmaAbsorption) * phiAse"
            ),
            "finalBetaVolumeFormula": "beta - timeStep * dndtAse",
        },
    }


def load_reference() -> dict:
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


def write_reference(reference: dict, *, dry_run: bool) -> None:
    text = json.dumps(reference, indent=2) + "\n"
    if dry_run:
        print(text, end="")
    else:
        REFERENCE_PATH.write_text(text, encoding="utf-8")


def require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise SystemExit(
            f"fixture validation failed for {label}: {actual!r} != {expected!r}"
        )


def validate_stored_reference() -> None:
    reference = load_reference()
    generation = reference["referenceGeneration"]
    repository = generation["repository"]
    driver = generation["driver"]
    environment = generation["environment"]

    require_equal(repository["commit"], JULIAASE_COMMIT, "JuliaASE commit")
    require_equal(repository["dirty"], False, "JuliaASE dirty flag")
    require_equal(repository["visibility"], "private", "repository visibility")
    require_equal(
        repository["accessContact"], "Daniel Albach", "repository access contact"
    )
    require_equal(driver["historicalCommit"], HISTORICAL_DRIVER_COMMIT, "driver commit")
    require_equal(driver.get("kind"), "embeddedJuliaSource", "driver kind")
    require_equal(
        driver["sha256"],
        hashlib.sha256(JULIA_DRIVER_SOURCE.encode("utf-8")).hexdigest(),
        "embedded driver hash",
    )
    require_equal(
        sha256(REPO_ROOT / driver["wrapperPath"]),
        driver["wrapperSha256"],
        "wrapper file hash",
    )
    require_equal(environment["juliaVersion"], JULIA_VERSION, "Julia version")
    require_equal(
        environment["juliaExecutableSha256"],
        JULIA_EXECUTABLE_SHA256,
        "Julia executable hash",
    )
    require_equal(
        environment["juliaDistributionSha256"],
        JULIA_DISTRIBUTION_SHA256,
        "Julia distribution hash",
    )
    require_equal(environment["projectSha256"], JULIAASE_PROJECT_SHA256, "project hash")
    require_equal(
        environment["manifestSha256"],
        JULIAASE_MANIFEST_SHA256,
        "recorded manifest hash",
    )
    require_equal(
        sha256(REPO_ROOT / environment["manifestFile"]),
        JULIAASE_MANIFEST_SHA256,
        "manifest file hash",
    )
    require_equal(
        generation["parameters"],
        simulation_parameters(reference["referenceRayCount"]),
        "JuliaASE parameters",
    )
    require_equal(generation["result"]["status"], "converged", "driver status")
    require_equal(generation["result"]["passes"], 1, "driver pass count")

    beta = np.asarray(reference["initialBetaVolume"], dtype=np.float64)
    absorption = float(max(reference["crossSections"]["crossSectionAbsorption"]))
    emission = float(max(reference["crossSections"]["crossSectionEmission"]))
    expected_dndt = (beta * (emission + absorption) - absorption) * np.asarray(
        reference["phiAse"], dtype=np.float64
    )
    require_equal(reference["dndtAse"], expected_dndt.tolist(), "derived dndtAse")
    expected_final_beta = beta - float(reference["timeStep"]) * expected_dndt
    require_equal(
        reference["finalBetaVolume"],
        expected_final_beta.tolist(),
        "derived finalBetaVolume",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="print JSON instead of overwriting"
    )
    parser.add_argument(
        "--ray-count",
        type=int,
        default=1_000_000,
        help="JuliaASE histories used for the Monte Carlo reference (default: %(default)s)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate committed metadata and checksums without executing JuliaASE",
    )
    args = parser.parse_args(argv)
    if args.ray_count <= 0:
        raise SystemExit("--ray-count must be positive")
    if args.validate_only:
        validate_stored_reference()
        print("JuliaASE reflection fixture metadata is valid")
        return 0

    root = juliaase_root()
    verify_juliaase_checkout(root)
    julia, julia_version = find_julia()
    environment = exact_environment()
    with TemporaryDirectory(prefix="hase-juliaase-project-") as temporary:
        project = Path(temporary)
        prepare_project(root, project, julia, environment)
        generated, driver_stderr = generate_reference(
            root, project, julia, args.ray_count, environment
        )

    reference = load_reference()
    reference["referenceGeneration"] = generation_metadata(
        root, julia, julia_version, args.ray_count, generated, driver_stderr
    )
    reference["phiAse"] = generated["phiAse"]
    reference["reflectedPassWeightFractions"] = generated[
        "reflectedPassWeightFractions"
    ]
    reference["referenceRayCount"] = generated["rayCount"]
    reference["referenceInitialReflectedWeight"] = generated["initialReflectedWeight"]

    beta = np.asarray(reference["initialBetaVolume"], dtype=np.float64)
    absorption = float(max(reference["crossSections"]["crossSectionAbsorption"]))
    emission = float(max(reference["crossSections"]["crossSectionEmission"]))
    dndt = (beta * (emission + absorption) - absorption) * np.asarray(
        reference["phiAse"], dtype=np.float64
    )
    reference["dndtAse"] = dndt.tolist()
    reference["finalBetaVolume"] = (beta - float(reference["timeStep"]) * dndt).tolist()
    reference["referenceGeneration"]["postprocessing"]["timeStep"] = float(
        reference["timeStep"]
    )
    write_reference(reference, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
