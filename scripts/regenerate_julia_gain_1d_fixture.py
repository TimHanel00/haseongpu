#!/usr/bin/env python3
"""Regenerate the committed private JULIA_gain_1D SSG fixture."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = REPO_ROOT / "tests" / "data" / "julia1D.csv"
METADATA_PATH = REPO_ROOT / "tests" / "data" / "julia1D.metadata.json"
MANIFEST_PATH = REPO_ROOT / "tests" / "data" / "juliaGain1D-manifest.toml"

JULIA_DRIVER_SOURCE = r'''
using Printf

const julia_gain_1d_root =
    length(ARGS) == 1 ? ARGS[1] : error("usage: embedded JULIA_gain_1D driver JULIA_GAIN_1D_ROOT")

include(joinpath(julia_gain_1d_root, "src", "JULIA_absorption_1D.jl"))
using .JULIA_absorption_1D

const config = SimConfig(
    material_file = "Yb_YAGref.h5",
    doping_at_pct = 2.0,
    crys_l = 0.7,
    pump_power = [16e3],
    pump_wl = [940.0],
    pump_center = [940.0],
    pump_fwhm = [6.0],
    pump_passes = [1 -1],
    pump_dur = 1e-3,
    pump_window = 2e-3,
    ssg_wl = 1030.0,
    ssg_passes = 2,
    tloss_system = 1.0,
    temperatures = [300.0],
    consistent_pump_rk4 = true,
    steps_time = 101,
    steps_crystal = 10,
    pulses = PulseConfig[],
)

const times, ssg = ssg_history(config)

println("step,time_s,SSG")
for (index, (time, gain)) in enumerate(zip(times, ssg))
    @printf "%d,%.12e,%.12e\n" index - 1 time gain
end
'''.strip()

JULIA_GAIN_1D_REPOSITORY = (
    "https://codebase.helmholtz.cloud/penelope-julia/julia_gain_1d.git"
)
JULIA_GAIN_1D_COMMIT = "7e01494a4984ab02dd14fec6f4b7f055e01f49b9"
JULIA_GAIN_1D_PROJECT_SHA256 = (
    "b0f90d4dcee42923bc23b4880cdd48b0e3072841461d4b31436967a8fce01ce1"
)
MATERIAL_RELATIVE_PATH = Path("materials") / "Yb_YAGref.h5"
MATERIAL_SHA256 = (
    "01d5d143fa32e3cb8bea257e04b26e706355237dc99dcefa0298e6ba2aa800aa"
)
MANIFEST_SHA256 = (
    "af026be026bec685aff29219755d11c21ed18563f38e423d100d868a4e3fc014"
)
JULIA_VERSION = "1.12.6"
JULIA_EXECUTABLE_SHA256 = (
    "fd670aabc838e93f178cbcf7304c5e9c50aadcce3735e86d5e3218b1bb04e602"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def julia_gain_1d_root() -> Path:
    configured = os.environ.get("JULIA_GAIN_1D_ROOT")
    root = (
        Path(configured).expanduser().resolve()
        if configured
        else REPO_ROOT / "julia1D"
    )
    if not (root / "Project.toml").is_file():
        raise SystemExit(
            "Private JULIA_gain_1D checkout not found. Set JULIA_GAIN_1D_ROOT; "
            "contact Dr. Daniel Albach to request repository access."
        )
    return root


def verify_checkout(root: Path) -> None:
    revision = git_output(root, "rev-parse", "HEAD")
    if revision != JULIA_GAIN_1D_COMMIT:
        raise SystemExit(
            f"JULIA_gain_1D must be checked out at {JULIA_GAIN_1D_COMMIT}, "
            f"got {revision}"
        )
    status = git_output(root, "status", "--porcelain")
    if status:
        raise SystemExit(
            f"JULIA_gain_1D checkout must be clean; git status reported:\n{status}"
        )
    project_hash = sha256(root / "Project.toml")
    if project_hash != JULIA_GAIN_1D_PROJECT_SHA256:
        raise SystemExit(
            "JULIA_gain_1D Project.toml hash mismatch: "
            f"expected {JULIA_GAIN_1D_PROJECT_SHA256}, got {project_hash}"
        )
    material_hash = sha256(root / MATERIAL_RELATIVE_PATH)
    if material_hash != MATERIAL_SHA256:
        raise SystemExit(
            f"material hash mismatch: expected {MATERIAL_SHA256}, got {material_hash}"
        )


def verify_committed_inputs() -> None:
    checks = ((MANIFEST_PATH, MANIFEST_SHA256, "Julia manifest"),)
    for path, expected, label in checks:
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"{label} hash mismatch: expected {expected}, got {actual}")


def find_julia() -> tuple[str, str, Path]:
    julia = shutil.which("julia")
    if julia is None:
        raise SystemExit(f"Julia {JULIA_VERSION} was not found on PATH")
    completed = subprocess.run(
        [
            julia,
            "--startup-file=no",
            "-e",
            "println(VERSION); println(joinpath(Sys.BINDIR, Base.julia_exename()))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    version, executable_text = completed.stdout.splitlines()
    if version != JULIA_VERSION:
        raise SystemExit(f"Julia {JULIA_VERSION} is required, got {version}")
    executable = Path(executable_text).resolve()
    executable_hash = sha256(executable)
    if executable_hash != JULIA_EXECUTABLE_SHA256:
        raise SystemExit(
            "Julia executable hash mismatch: "
            f"expected {JULIA_EXECUTABLE_SHA256}, got {executable_hash}"
        )
    return julia, version, executable


def exact_environment() -> dict[str, str]:
    return dict(
        os.environ,
        JULIA_NUM_THREADS="1",
        JULIA_PKG_PRECOMPILE_AUTO="0",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )


def prepare_project(
    root: Path, project: Path, julia: str, environment: dict[str, str]
) -> None:
    shutil.copyfile(root / "Project.toml", project / "Project.toml")
    shutil.copyfile(MANIFEST_PATH, project / "Manifest.toml")
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
    if instantiated_hash != MANIFEST_SHA256:
        raise SystemExit(
            "Pkg.instantiate() changed the pinned manifest: "
            f"expected {MANIFEST_SHA256}, got {instantiated_hash}"
        )


def generate_reference(
    root: Path, project: Path, julia: str, environment: dict[str, str]
) -> bytes:
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
        env=environment,
    )
    return completed.stdout


def validate_csv(content: bytes) -> list[dict[str, str]]:
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
    if not rows or list(rows[0]) != ["step", "time_s", "SSG"]:
        raise SystemExit("generated Julia fixture has unexpected columns")
    if len(rows) != 101:
        raise SystemExit(f"generated Julia fixture must contain 101 data rows, got {len(rows)}")
    for expected_step, row in enumerate(rows):
        if int(row["step"]) != expected_step:
            raise SystemExit(f"unexpected step at row {expected_step}: {row['step']}")
        expected_time = expected_step * 2.0e-5
        if abs(float(row["time_s"]) - expected_time) > 1.0e-15:
            raise SystemExit(
                f"unexpected time at step {expected_step}: {row['time_s']}"
            )
    return rows


def parameters() -> dict:
    return {
        "materialFile": "Yb_YAGref.h5",
        "dopingAtomicPercent": 2.0,
        "crystalLengthCm": 0.7,
        "pumpPowerWPerCm2": [16000.0],
        "pumpWavelengthNm": [940.0],
        "pumpCenterNm": [940.0],
        "pumpFwhmNm": [6.0],
        "pumpPasses": [[1, -1]],
        "pumpDurationSeconds": 0.001,
        "pumpWindowSeconds": 0.002,
        "ssgWavelengthNm": 1030.0,
        "ssgPasses": 2,
        "systemTransmission": 1.0,
        "temperaturesK": [300.0],
        "consistentPumpRk4": True,
        "timePoints": 101,
        "crystalPoints": 10,
        "extractionPulses": 0,
    }


def build_metadata(
    reference_content: bytes,
    rows: list[dict[str, str]],
    julia_version: str,
    julia_executable: Path,
) -> dict:
    return {
        "schemaVersion": 1,
        "description": (
            "JULIA_gain_1D one-dimensional small-signal-gain reference used by "
            "the disabled-ASE laserPumpCladding pump regression."
        ),
        "referenceGeneration": {
            "implementation": "JULIA_gain_1D",
            "generatedAtUtc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "repository": {
                "url": JULIA_GAIN_1D_REPOSITORY,
                "visibility": "private",
                "accessContact": "Dr. Daniel Albach",
                "commit": JULIA_GAIN_1D_COMMIT,
                "dirty": False,
            },
            "driver": {
                "kind": "embeddedJuliaSource",
                "sha256": sha256_bytes(JULIA_DRIVER_SOURCE.encode("utf-8")),
                "wrapperPath": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                "wrapperSha256": sha256(Path(__file__).resolve()),
            },
            "command": (
                "JULIA_GAIN_1D_ROOT=/path/to/julia_gain_1d "
                "JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                "python3 scripts/regenerate_julia_gain_1d_fixture.py"
            ),
            "parameters": parameters(),
            "environment": {
                "juliaVersion": julia_version,
                "juliaExecutable": str(julia_executable),
                "juliaExecutableSha256": sha256(julia_executable),
                "projectSha256": JULIA_GAIN_1D_PROJECT_SHA256,
                "manifestFile": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
                "manifestSha256": sha256(MANIFEST_PATH),
                "materialFile": str(MATERIAL_RELATIVE_PATH),
                "materialSha256": MATERIAL_SHA256,
                "operatingSystem": platform.platform(),
                "architecture": platform.machine(),
                "juliaNumThreads": 1,
                "ompNumThreads": 1,
                "openblasNumThreads": 1,
                "startupFile": "disabled",
            },
        },
        "reference": {
            "file": str(REFERENCE_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_bytes(reference_content),
            "columns": ["step", "time_s", "SSG"],
            "dataRows": len(rows),
            "firstStep": 0,
            "lastStep": 100,
            "timeStepSeconds": 2.0e-5,
        },
        "haseComparison": {
            "example": "example/laserPumpCladding.py",
            "aseEnabled": False,
            "timeSteps": 100,
            "pumpSteps": 50,
            "pumpRayCount": 10000,
            "spectralResolution": 191,
            "referenceFirstComparedRow": 1,
            "referenceLastComparedRow": 100,
            "referenceStepZeroUse": (
                "initial-condition sanity check only; HASE VTK snapshots start after "
                "the first completed step"
            ),
            "referenceQuantity": "SSG",
            "haseQuantity": "net_gain_factor",
            "relativeTolerance": 0.1,
            "absoluteTolerance": 1.0e-8,
        },
    }


def validate_committed_metadata() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    repository = metadata["referenceGeneration"]["repository"]
    environment = metadata["referenceGeneration"]["environment"]
    driver = metadata["referenceGeneration"]["driver"]
    if repository != {
        "url": JULIA_GAIN_1D_REPOSITORY,
        "visibility": "private",
        "accessContact": "Dr. Daniel Albach",
        "commit": JULIA_GAIN_1D_COMMIT,
        "dirty": False,
    }:
        raise SystemExit("committed JULIA_gain_1D repository metadata is stale")
    checks = (
        (REPO_ROOT / driver["wrapperPath"], driver["wrapperSha256"], "wrapper"),
        (REPO_ROOT / environment["manifestFile"], environment["manifestSha256"], "manifest"),
        (REFERENCE_PATH, metadata["reference"]["sha256"], "reference CSV"),
    )
    for path, expected, label in checks:
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"{label} hash mismatch: expected {expected}, got {actual}")
    if driver.get("kind") != "embeddedJuliaSource":
        raise SystemExit("committed JULIA_gain_1D driver kind is stale")
    embedded_driver_hash = sha256_bytes(JULIA_DRIVER_SOURCE.encode("utf-8"))
    if driver.get("sha256") != embedded_driver_hash:
        raise SystemExit(
            "embedded Julia driver hash mismatch: "
            f"expected {driver.get('sha256')}, got {embedded_driver_hash}"
        )
    validate_csv(REFERENCE_PATH.read_bytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate committed metadata and checksums without executing Julia",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        validate_committed_metadata()
        print("JULIA_gain_1D fixture metadata is valid")
        return

    root = julia_gain_1d_root()
    verify_checkout(root)
    verify_committed_inputs()
    julia, julia_version, julia_executable = find_julia()
    environment = exact_environment()
    with TemporaryDirectory(prefix="hase-julia-gain-1d-project-") as temporary:
        project = Path(temporary)
        prepare_project(root, project, julia, environment)
        reference_content = generate_reference(root, project, julia, environment)
    rows = validate_csv(reference_content)
    metadata = build_metadata(
        reference_content, rows, julia_version, julia_executable
    )
    REFERENCE_PATH.write_bytes(reference_content)
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {REFERENCE_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {METADATA_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
