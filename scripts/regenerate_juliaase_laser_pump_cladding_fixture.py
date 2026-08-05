#!/usr/bin/env python3
"""Regenerate the five-step JuliaASE laserPumpCladding reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = (
    REPO_ROOT
    / "tests"
    / "data"
    / "juliaASE"
    / "laser_pump_cladding_5_step_reference"
)
REFERENCE_PATH = FIXTURE_DIR / "reference.json"
MANIFEST_PATH = (
    REPO_ROOT
    / "tests"
    / "data"
    / "juliaASE"
    / "reflection_surface_reference"
    / "juliaase-manifest.toml"
)

JULIA_DRIVER_SOURCE = r'''
using Random
using Printf

length(ARGS) == 2 || error("usage: embedded JuliaASE driver JULIAASE_ROOT MODE")
const JULIAASE_ROOT = abspath(ARGS[1])
const REFLECTION_MODE = ARGS[2]
const HASE_REFRACTIVE_INDEX = 1.83f0
REFLECTION_MODE in ("withoutReflections", "withReflections") ||
    error("MODE must be withoutReflections or withReflections")

ENV["SIMRUN_FIDELITY"] = "debug"
ENV["SIMRUN_STAGE"] = "full"
ENV["SIMRUN_NSLICES"] = "5"
ENV["SIMRUN_ASE"] = "on"
ENV["SIMRUN_BACK"] = REFLECTION_MODE == "withReflections" ? "water_cavity_openedge" : "ar_ase"
ENV["SIMRUN_SPECTRAL"] = "on"
ENV["SIMRUN_RELAY_GPU"] = "off"
ENV["SIMRUN_PUMP_PROFILE"] = "supergauss"
ENV["SIMRUN_PUMP_PEXP"] = "40"
ENV["SIMRUN_RELAY_RAYS"] = "50000"
ENV["SIMRUN_SRM_K"] = "256"

const FULL_EXAMPLE = joinpath(JULIAASE_ROOT, "test", "simulation_run", "run_full_simulation.jl")
source = read(FULL_EXAMPLE, String)
trimmed = replace(source, r"\nmain\(\)\s*$" => "\n")
trimmed == source && error("could not remove the full example's main() call")
freeze_return = "return (node = project_tet_to_nodes(dn_tet, mesh), q_tet = q_tet)"
occursin(freeze_return, trimmed) || error("could not extend freeze_ase! convergence result")
trimmed = replace(
    trimmed,
    freeze_return => (
        "return (node = project_tet_to_nodes(dn_tet, mesh), q_tet = q_tet, " *
        "status = string(res.status), passes = res.n_passes)"
    ),
    count=1,
)
Base.include_string(Main, trimmed, FULL_EXAMPLE)

# HASE's reflective mode uses ideal AR end coatings: zero sub-critical
# reflectivity and total internal reflection against air. The cylindrical side
# is absorbing. The JuliaASE example's cavity setup installs the required AR
# coating table and SRM connectivity; this replacement selects the HASE faces.
@eval function set_ase_faces!(state, mesh)
    bfl = state.boundary_faces
    if REFLECTION_MODE == "withoutReflections"
        @inbounds for k in eachindex(bfl.bound_ind)
            b = bfl.bound_ind[k]
            if b == BI_FRONT_PUMP || b == Int32(BOUND_EXTERNAL)
                state.bfl_outside_n[k] = HASE_REFRACTIVE_INDEX
            end
            state.face_coating_ind[k] = Int32(0)
        end
    else
        bfl = state.boundary_faces
        @inbounds for k in eachindex(bfl.bound_ind)
            b = bfl.bound_ind[k]
            if b == BI_FRONT_PUMP || b == Int32(BOUND_TMM_A) || b == Int32(BOUND_EXTERNAL)
                state.bfl_outside_n[k] = N_FRONT_EXT
                state.face_coating_ind[k] = AR_COATING_IDX
            elseif b == Int32(BOUND_TMM_B)
                state.bfl_outside_n[k] = HASE_REFRACTIVE_INDEX
                state.face_coating_ind[k] = Int32(0)
            end
        end
    end
    return nothing
end

@eval function set_relay_faces!(state)
    bfl = state.boundary_faces
    @inbounds for k in eachindex(bfl.bound_ind)
        b = bfl.bound_ind[k]
        if b == BI_FRONT_PUMP || b == Int32(BOUND_EXTERNAL)
            state.bfl_outside_n[k] = HASE_REFRACTIVE_INDEX
        end
    end
    fill!(state.face_coating_ind, Int32(0))
    return nothing
end

function volume_weighted_beta(state, mesh, N0)
    conn = mesh.connectivity
    weighted = 0.0
    @inbounds for t in eachindex(mesh.volumes)
        beta = 0.25 * (
            Float64(state.N_inv_nodes[conn[1, t]]) +
            Float64(state.N_inv_nodes[conn[2, t]]) +
            Float64(state.N_inv_nodes[conn[3, t]]) +
            Float64(state.N_inv_nodes[conn[4, t]])
        ) / N0
        weighted += beta * Float64(mesh.volumes[t])
    end
    return weighted / sum(Float64.(mesh.volumes))
end

function phi_volume_integral(state, mesh)
    phi = Tallies.compute_phi_ase(state, N_RAYS_ASE)
    return sum(Float64.(phi) .* Float64.(mesh.volumes))
end

function fixture_main()
    mesh_path = joinpath(JULIAASE_ROOT, "test", "simulation_run", "mesh", "disk_debug.msh")
    isfile(mesh_path) || error("missing pinned JuliaASE debug mesh: $mesh_path")
    mesh, bfl = build_mesh(mesh_path)
    p = load_physics()
    state = build_state(mesh, bfl)
    fill!(state.domain_n, HASE_REFRACTIVE_INDEX)
    fill!(state.N_inv_nodes, 0f0)

    pump_spec = make_relay_spec(LAMBDA_PUMP, PUMP_POWER)
    exit_b = Int32[BOUND_EXTERNAL, BI_FRONT_PUMP]
    entry_b = Int32[BOUND_EXTERNAL]
    relay_surface = build_relay_surface(bfl, mesh, entry_b)
    retro = ImagingTransform(relay_surface.origin, relay_surface.e1, relay_surface.e2, relay_surface.normal)

    ase_rng = MersenneTwister(5489)
    pump_rng = MersenneTwister(5489)
    n_nodes = size(mesh.points, 2)
    rows = NamedTuple[]
    for step in 1:5
        # laserPumpCladding uses prePump=true: the first step establishes an
        # inversion before the first ASE transport evaluation.
        if step == 1
            ase_node = zeros(Float64, n_nodes)
            phi_integral = 0.0
            ase_status = "prePump"
            ase_passes = 0
        else
            ase = freeze_ase!(state, mesh, p, ase_rng, false, nothing)
            ase_node = ase.node
            phi_integral = phi_volume_integral(state, mesh)
            ase_status = ase.status
            ase_passes = ase.passes
            ase_status == "converged" || error(
                "ASE did not converge at step $step: status=$ase_status passes=$ase_passes"
            )
        end
        pump_node = freeze_pump!(
            state, mesh, p, pump_spec, retro, exit_b, entry_b, pump_rng;
            relay_gpu=false,
        )
        n2 = rk4_frozen_ase_step(
            Float64.(state.N_inv_nodes), DT_SLICE, ase_node .+ pump_node, p.tau_rad
        )
        clamp!(n2, 0.0, p.N0)
        state.N_inv_nodes .= Float32.(n2)
        push!(rows, (
            step=step,
            time=step * DT_SLICE,
            betaVolumeMean=volume_weighted_beta(state, mesh, p.N0),
            phiAseVolumeIntegral=phi_integral,
            aseStatus=ase_status,
            asePasses=ase_passes,
        ))
    end

    print("{\"mode\":\"", REFLECTION_MODE, "\",\"simulationSteps\":[")
    for (index, row) in enumerate(rows)
        index == 1 || print(",")
        @printf(
            "{\"step\":%d,\"time\":%.17g,\"betaVolumeMean\":%.17g,\"phiAseVolumeIntegral\":%.17g,\"aseStatus\":\"%s\",\"asePasses\":%d}",
            row.step,
            row.time,
            row.betaVolumeMean,
            row.phiAseVolumeIntegral,
            row.aseStatus,
            row.asePasses,
        )
    end
    println("]}")
end

fixture_main()
'''.strip()

JULIAASE_COMMIT = "f6e19290ac06f7fd6f9492e6bb14a86973a50166"
JULIAASE_REPOSITORY = "https://codebase.helmholtz.cloud/penelope-julia/julia_ase.git"
JULIAASE_PROJECT_SHA256 = "8e79c420e591978d179233554597cbaa896debd18baf119c4de325b71695f7d7"
JULIAASE_MANIFEST_SHA256 = "6bc0d15a228289b5c6f6c2b50e7ed6e1b862729a2ac23657dccecfab8251347a"
JULIAASE_EXAMPLE_SHA256 = "78396c070827ece084755d2198cee1f05ace43864f4c66011d64c90b0b3a66a7"
JULIAASE_MESH_SHA256 = "594d5ad477a660c4fab5a84fa6d72b3d08b483eec4510283bced51e64f830c92"
JULIAASE_MATERIAL_SHA256 = "16bb18cb803bbefb35bcf405d93b8f3eebf98094c9cc05a2faa0441ab687fa6a"
JULIA_VERSION = "1.10.9"
JULIA_EXECUTABLE_SHA256 = "1436babaecb80defc11086ab2587c8dbfda4705cbe6608aa21bc9ddccd15e60f"
HASE_COMMIT = "ba22ddc3692cf3ec8dcc10f90b95e81be154b9c7"
HASE_SOURCE_HASHES = {
    "example/laserPumpCladding.py": "eb94162df14663bba23fc053758cc87038b29da88b7abfc4e0ff56614a5157b9",
    "example/data/ptTet4.vtk": "c7ec39245a45188cc92c2cb10de15eea9c0b16c79102c0891c185127f3d83556",
    "example/input/lambda_a.txt": "4e206b66764c8558eaee67bb57a81a6a1acce3ac592b5bb05eec6ef7a7e58251",
    "example/input/sigma_a.txt": "85228a6a66291146b39d333803550f38c8b3cda1c31c3318e6133e8ab84343f5",
    "example/input/lambda_e.txt": "4e206b66764c8558eaee67bb57a81a6a1acce3ac592b5bb05eec6ef7a7e58251",
    "example/input/sigma_e.txt": "e6a5b5ec7bede97bf42dd01129482e3dfe9fae6a41ef2eb96cd7561e21d514e6",
}
MODES = ("withoutReflections", "withReflections")
GENERATION_COMMAND = (
    "JULIAASE_ROOT=/path/to/juliaASE JULIA_NUM_THREADS=1 "
    "JULIA_LOAD_PATH=@:@stdlib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
    "python3 scripts/regenerate_juliaase_laser_pump_cladding_fixture.py"
)
GENERATION_ENVIRONMENT = {
    "juliaNumThreads": 1,
    "juliaLoadPath": "@:@stdlib",
    "ompNumThreads": 1,
    "openblasNumThreads": 1,
    "startupFile": "disabled",
}
PARAMETERS = {
    "simulationSteps": 5,
    "timeStepSeconds": 2.0e-5,
    "prePump": True,
    "pumpActiveSteps": 5,
    "pumpConfiguredSteps": 50,
    "aseRayCount": 30_000,
    "pumpRayCount": 50_000,
    "aseRngSeed": 5489,
    "pumpRngSeed": 5489,
    "spectralResolution": 191,
    "refractiveIndex": 1.83,
    "surfaceReservoirSize": 256,
    "reflectionMaxIterations": 80,
    "reflectionTolerance": 1.0e-3,
    "pumpProfile": "supergaussian",
    "pumpProfileExponent": 40.0,
    "pumpRadiusMeters": 0.015,
    "pumpIntensityWattsPerSquareMeter": 1.6e8,
}
TOLERANCES = {
    "relative": 0.05,
    "betaVolumeMeanAbsolute": 1.0e-8,
    "phiAseVolumeIntegralAbsolute": 0.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def juliaase_root() -> Path:
    configured = os.environ.get("JULIAASE_ROOT")
    root = (
        Path(configured).expanduser().resolve()
        if configured
        else REPO_ROOT / "julia_ase"
    )
    if not (root / "Project.toml").is_file():
        raise SystemExit("JuliaASE checkout not found; set JULIAASE_ROOT")
    return root


def verify_source(root: Path) -> None:
    if git_output(root, "rev-parse", "HEAD") != JULIAASE_COMMIT:
        raise SystemExit(f"JuliaASE must be checked out at {JULIAASE_COMMIT}")
    if git_output(root, "status", "--porcelain"):
        raise SystemExit("JuliaASE checkout must be clean")
    expected = {
        root / "Project.toml": JULIAASE_PROJECT_SHA256,
        root / "test/simulation_run/run_full_simulation.jl": JULIAASE_EXAMPLE_SHA256,
        root / "test/simulation_run/mesh/disk_debug.msh": JULIAASE_MESH_SHA256,
        root / "materials/Yb_YAGref.h5": JULIAASE_MATERIAL_SHA256,
        MANIFEST_PATH: JULIAASE_MANIFEST_SHA256,
    }
    for path, expected_hash in expected.items():
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise SystemExit(
                f"SHA-256 mismatch for {path}: expected {expected_hash}, got {actual_hash}"
            )


def find_julia() -> Path:
    executable = shutil.which("julia")
    if executable is None:
        raise SystemExit("Julia 1.10.9 is required on PATH")
    path = Path(executable).resolve()
    version = subprocess.run(
        [str(path), "--startup-file=no", "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if version != f"julia version {JULIA_VERSION}":
        raise SystemExit(f"expected Julia {JULIA_VERSION}, got {version}")
    actual_hash = sha256(path)
    if actual_hash != JULIA_EXECUTABLE_SHA256:
        raise SystemExit("Julia executable SHA-256 does not match the pinned distribution")
    return path


def normalized_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "JULIA_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "JULIA_LOAD_PATH": "@:@stdlib",
        }
    )
    return env


def parse_generator_result(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith('{"mode"'):
            return json.loads(line)
    raise SystemExit(f"Julia generator did not emit a JSON result:\n{stdout}")


def generate(root: Path, julia: Path) -> dict[str, dict]:
    env = normalized_environment()
    with TemporaryDirectory(prefix="hase-juliaase-laser-pump-") as temporary:
        project = Path(temporary)
        shutil.copy2(root / "Project.toml", project / "Project.toml")
        shutil.copy2(MANIFEST_PATH, project / "Manifest.toml")
        subprocess.run(
            [
                str(julia),
                "--startup-file=no",
                f"--project={project}",
                "-e",
                "using Pkg; Pkg.instantiate()",
            ],
            check=True,
            env=env,
        )
        results = {}
        for mode in MODES:
            completed = subprocess.run(
                [
                    str(julia),
                    "--startup-file=no",
                    f"--project={project}",
                    "-e",
                    JULIA_DRIVER_SOURCE,
                    str(root),
                    mode,
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            result = parse_generator_result(completed.stdout)
            result["generatorStderr"] = completed.stderr.replace(str(root), "$JULIAASE_ROOT")
            results[mode] = result
        return results


def fixture(results: dict[str, dict], julia: Path) -> dict:
    return {
        "schema": "hase.juliaASE.laserPumpCladdingFiveStep.v1",
        "description": (
            "Five simulation steps of the JuliaASE full thin-disk example, "
            "configured as the corresponding physical HASE laserPumpCladding case "
            "with and without ASE reflections."
        ),
        "referenceGeneration": {
            "implementation": "JuliaASE",
            "generatedAtUtc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "repository": {
                "url": JULIAASE_REPOSITORY,
                "commit": JULIAASE_COMMIT,
                "visibility": "private",
                "accessContact": "Daniel Albach",
                "dirty": False,
            },
            "generator": {
                "kind": "embeddedJuliaSource",
                "sha256": sha256_text(JULIA_DRIVER_SOURCE),
                "wrapperPath": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                "wrapperSha256": sha256(Path(__file__).resolve()),
                "juliaaseExamplePath": "test/simulation_run/run_full_simulation.jl",
                "juliaaseExampleSha256": JULIAASE_EXAMPLE_SHA256,
            },
            "julia": {
                "version": JULIA_VERSION,
                "executableSha256": sha256(julia),
            },
            "platform": platform.platform(),
            "command": GENERATION_COMMAND,
            "environment": GENERATION_ENVIRONMENT,
            "manifestPath": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
            "manifestSha256": sha256(MANIFEST_PATH),
        },
        "parameters": PARAMETERS,
        "sourceData": {
            "meshPath": "test/simulation_run/mesh/disk_debug.msh",
            "meshSha256": JULIAASE_MESH_SHA256,
            "materialPath": "materials/Yb_YAGref.h5",
            "materialSha256": JULIAASE_MATERIAL_SHA256,
            "haseCommit": HASE_COMMIT,
            "haseFiles": HASE_SOURCE_HASHES,
        },
        "tolerances": TOLERANCES,
        "aseResults": results,
    }


def validate(reference: dict) -> None:
    if reference["schema"] != "hase.juliaASE.laserPumpCladdingFiveStep.v1":
        raise SystemExit("unexpected fixture schema")
    if reference["parameters"] != PARAMETERS:
        raise SystemExit("stored simulation parameters are stale")
    if reference["tolerances"] != TOLERANCES:
        raise SystemExit("stored regression tolerances are stale")
    for mode in MODES:
        rows = reference["aseResults"][mode]["simulationSteps"]
        if [row["step"] for row in rows] != [1, 2, 3, 4, 5]:
            raise SystemExit(f"{mode} does not contain steps 1 through 5")
        if rows[0]["phiAseVolumeIntegral"] != 0.0:
            raise SystemExit(f"{mode} first step must be the ASE-free pre-pump step")
        if rows[0]["aseStatus"] != "prePump" or rows[0]["asePasses"] != 0:
            raise SystemExit(f"{mode} first step has invalid ASE status")
        for row in rows:
            if not all(
                math.isfinite(float(row[name]))
                for name in ("time", "betaVolumeMean", "phiAseVolumeIntegral")
            ):
                raise SystemExit(f"{mode} contains a non-finite observable")
        if any(row["aseStatus"] != "converged" for row in rows[1:]):
            raise SystemExit(f"{mode} contains an unconverged ASE evaluation")
    generation = reference["referenceGeneration"]
    repository = generation["repository"]
    if repository != {
        "url": JULIAASE_REPOSITORY,
        "commit": JULIAASE_COMMIT,
        "visibility": "private",
        "accessContact": "Daniel Albach",
        "dirty": False,
    }:
        raise SystemExit("stored JuliaASE repository provenance is stale")
    if generation["generator"].get("kind") != "embeddedJuliaSource":
        raise SystemExit("stored Julia generator kind is stale")
    if generation["generator"]["sha256"] != sha256_text(JULIA_DRIVER_SOURCE):
        raise SystemExit("stored embedded Julia generator checksum is stale")
    if generation["generator"]["wrapperSha256"] != sha256(Path(__file__).resolve()):
        raise SystemExit("stored regeneration wrapper checksum is stale")
    if generation["manifestSha256"] != sha256(MANIFEST_PATH):
        raise SystemExit("stored Julia manifest checksum is stale")
    if generation["generator"]["juliaaseExampleSha256"] != JULIAASE_EXAMPLE_SHA256:
        raise SystemExit("stored JuliaASE full-example checksum is stale")
    if generation["julia"]["version"] != JULIA_VERSION:
        raise SystemExit("stored Julia version is stale")
    if generation["julia"]["executableSha256"] != JULIA_EXECUTABLE_SHA256:
        raise SystemExit("stored Julia executable checksum is stale")
    if generation["command"] != GENERATION_COMMAND:
        raise SystemExit("stored generation command is stale")
    if generation["environment"] != GENERATION_ENVIRONMENT:
        raise SystemExit("stored generation environment is stale")
    source_data = reference["sourceData"]
    if source_data["meshSha256"] != JULIAASE_MESH_SHA256:
        raise SystemExit("stored JuliaASE mesh checksum is stale")
    if source_data["materialSha256"] != JULIAASE_MATERIAL_SHA256:
        raise SystemExit("stored JuliaASE material checksum is stale")
    if source_data["haseCommit"] != HASE_COMMIT:
        raise SystemExit("stored HASE commit is stale")
    if source_data["haseFiles"] != HASE_SOURCE_HASHES:
        raise SystemExit("stored HASE input checksums are stale")
    for path, expected_hash in HASE_SOURCE_HASHES.items():
        if sha256(REPO_ROOT / path) != expected_hash:
            raise SystemExit(f"HASE source checksum is stale: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        validate(json.loads(REFERENCE_PATH.read_text(encoding="utf-8")))
        print("JuliaASE laserPumpCladding fixture metadata is valid")
        return 0
    root = juliaase_root()
    verify_source(root)
    julia = find_julia()
    reference = fixture(generate(root, julia), julia)
    validate(reference)
    rendered = json.dumps(reference, indent=2) + "\n"
    if args.dry_run:
        print(rendered, end="")
    else:
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        REFERENCE_PATH.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
