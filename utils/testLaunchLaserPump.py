#!/usr/bin/env python3
"""Launch the laserPumpCladding example with CI smoke-test defaults."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(os.environ.get("GITHUB_WORKSPACE", Path(__file__).resolve().parents[1]))


def launch_command(openpmd_backend: str, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(repo_root() / "example" / "laserPumpCladding.py"),
        "--backend",
        "Host_Cpu_CpuSerial",
        "--openpmd-backend",
        openpmd_backend,
        "--time-steps",
        "1",
        "--pump-steps",
        "1",
        "--vtk-output-dir",
        str(output_dir),
        "--rng-seed",
        "5489",
    ]
    phi_ase_config = os.environ.get("HASE_PHIASE_CONFIG")
    if phi_ase_config:
        command.extend(("--phi-ase-config", phi_ase_config))
    return command


def launch_laser_pump(openpmd_backend: str, output_dir: Path) -> int:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        launch_command(openpmd_backend, output_dir),
        cwd=output_dir,
        check=False,
    ).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("openpmd_backend")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    return launch_laser_pump(args.openpmd_backend, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
