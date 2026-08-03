# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Laser-pump-cladding example using the public composition frontend."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

try:
    from ._source_tree_import import ensure_hase_importable
except ImportError:
    from _source_tree_import import ensure_hase_importable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PHI_ASE_CONFIG_PATH = Path(
    os.environ.get(
        "HASE_PHIASE_CONFIG",
        SCRIPT_DIR.parent / "config/hase-phiase.yaml",
    )
)

ensure_hase_importable()

from HASEonGPU import (  # noqa: E402
    AbsorbingSurface,
    ConstantReflectivitySurface,
    FrozenPhiAseRungeKutta4,
    InitialState,
    MonteCarloASESolver,
    MonteCarloPumpSolver,
    PlanarPumpRelay,
    Pump,
    integratePumpProfile,
    PumpAngularDistribution,
    PumpSpectrum,
    Simulation,
    SuperGaussianPumpProfile,
    SurfacePumpInjector,
    UnstructuredMesh,
    writeParaviewState,
    writeVtkState,
    materialLibrary,
    units,
)


def print_state(state):
    print(
        f"step={state.step:03d} "
        f"time={float(state.time.toValue(units.s)):.3e}s "
        f"mean_beta={state.sampledExcitationFraction.mean():.6e} "
        f"mean_phi={state.phiAse.mean():.6e}"
    )


def _local_gain(state, material):
    cross_sections = material.crossSections
    peak = int(np.argmax(cross_sections.emission.magnitude))
    sigma_absorption = np.asarray(cross_sections.absorption.magnitude)[peak]
    sigma_emission = np.asarray(cross_sections.emission.magnitude)[peak]
    beta = state.sampledExcitationFraction
    gain = (
        beta * (sigma_absorption + sigma_emission) - sigma_absorption
    ) * cross_sections.absorption.unit * material.activeIonDensity
    return gain.toValue(units.cm**-1)


def write_vtk_fields(
    state,
    vtkOutputDir=SCRIPT_DIR,
    cladding_absorption=1.0,
    material=None,
):
    if state.phiAse is None:
        raise ValueError("VTK export requires state.phiAse")
    if material is None:
        raise ValueError("VTK export requires material for gain")

    fields = {
        "betaCells": state.sampledExcitationFraction,
        "betaVolume": state.excitationFraction,
        "phiASE": state.phiAse,
        "dndtAse": state.sampledDExcitationDtAse,
        "dndtPump": state.dExcitationDtPump,
        "cladAbs": state.phiAse * float(cladding_absorption.toValue(units.cm**-1)),
        "localGain": _local_gain(state, material),
    }
    if state.volumePhiAse is not None:
        fields["volumePhiASE"] = state.volumePhiAse
    if state.volumeDExcitationDtAse is not None:
        fields["volumeDndtAse"] = state.volumeDExcitationDtAse
    path = Path(vtkOutputDir) / f"laserPumpCladding_{state.step:03d}.vtk"
    return writeVtkState(path, state, fields=fields)


BOTTOM_ASE_SURFACE_ID = 1
TOP_ASE_SURFACE_ID = 2
CLADDING_SURFACE_ID = 3
NUMBER_OF_Z_LAYERS = 10


def _assign_surface_domains(topology):
    """Name the bottom, top, and cylindrical exterior regions."""
    points = np.asarray(topology.points, dtype=np.float64)
    exterior = topology.neighborCells < 0
    z = points[:, 2]
    face_z = z[np.asarray(topology.facePointIndices, dtype=np.uint32)]
    bottom = exterior & np.all(np.isclose(face_z, np.min(z)), axis=2)
    top = exterior & np.all(np.isclose(face_z, np.max(z)), axis=2)
    side = exterior & ~bottom & ~top
    if not (np.any(bottom) and np.any(top) and np.any(side)):
        raise ValueError(
            "ptTet4.vtk must contain bottom, top, and cladding exterior faces"
        )

    topology = topology.withCellDomains(
        domain=1,
        name="crystal_volume",
        where="all",
    ).withSurfaceDomains(
        [
            {
                "domain": BOTTOM_ASE_SURFACE_ID,
                "name": "ase_bottom",
                "faceIndices": np.argwhere(bottom),
            },
            {
                "domain": TOP_ASE_SURFACE_ID,
                "name": "ase_top",
                "faceIndices": np.argwhere(top),
            },
            {
                "domain": CLADDING_SURFACE_ID,
                "name": "cladding",
                "faceIndices": np.argwhere(side),
            },
        ]
    )
    return topology


def laser_pump_cladding_mesh():
    return _assign_surface_domains(
        UnstructuredMesh.fromFile(
            SCRIPT_DIR / "data" / "ptTet4.vtk",
            coordinateUnit=units.cm,
        )
    )


def laser_pump_cladding_material(spectralResolution=1000):
    return materialLibrary["YbYAG"].at(
        temperature=293.15 * units.K,
        activeIonDensity=2 * 1.388e20 * (1 / units.cm**3),
        spectralResolution=spectralResolution, #applies linear interpolation on the underlying spectral decomposition
    )


def run_example(
    phiAseConfigPath=DEFAULT_PHI_ASE_CONFIG_PATH,
    backend=None,
    timeSteps=150,
    pumpSteps=50,
    vtkOutputDir=SCRIPT_DIR,
    openPmdOutputDir=None,
    openPmdBackend=None,
    enableAse=True,
    prePump=True,
    spectralResolution=1000,
    pumpRayCount=50000,
    pumpRngSeed=5489,
    reportTimings=False,
    **aseOverrides,
):
    mesh = laser_pump_cladding_mesh()
    material = laser_pump_cladding_material(spectralResolution)
    pump_wavelength = 940 * units.nm
    absorption = 5.5 * (1 / units.cm)

    if backend is not None:
        aseOverrides["backend"] = backend
    if openPmdBackend is not None:
        aseOverrides["openPmdBackend"] = openPmdBackend
    aseSolver = MonteCarloASESolver.fromYaml(
        phiAseConfigPath,
        **aseOverrides,
    )

    pump_profile = SuperGaussianPumpProfile(
        radiusU=1.5 * units.cm,
        radiusV=1.5 * units.cm,
        exponent=40,
    )
    pump_aperture = mesh.surface("ase_bottom")
    pump = Pump(
        totalPower=(16 * units.kW / units.cm**2)
        * integratePumpProfile(mesh, pump_aperture, pump_profile),
        spectrum=PumpSpectrum.monochromatic(pump_wavelength),
        angularDistribution=PumpAngularDistribution.collimated(),
        profile=pump_profile,
    )
    pumpSolver = MonteCarloPumpSolver(
        rayCount=pumpRayCount,
        seed=pumpRngSeed,
        maxSteps=pumpSteps,
    )

    print(f"Running simulation with backend {aseSolver.backend}")
    print(f"Using openPMD backend {aseSolver.openPmdBackend}")
    simulation = Simulation(
        mesh=mesh,
        aseSolver=aseSolver,
        pumpSolver=pumpSolver,
        timeIntegrator=FrozenPhiAseRungeKutta4(),
        timeStepSize=20 * units.us,
        initialState=InitialState(excitationFraction=0 * units.one),
        enableAse=enableAse,
        prePump=prePump,
        reportTimings=reportTimings,
    )
    simulation.addMaterial(material, domains=mesh.volume("crystal_volume"))
    simulation.addBoundary(
        ConstantReflectivitySurface(),
        domains=mesh.surface("ase_bottom", "ase_top"),
    )
    simulation.addBoundary(
        AbsorbingSurface(),
        domains=mesh.surface("cladding"),
    )
    simulation.addPump(
        pump,
        injectionMethod=SurfacePumpInjector(surface=pump_aperture),
        relays=(PlanarPumpRelay.retroreflect(mesh.surface("ase_top")),),
    )
    simulation.onStep(print_state)
    simulation.onStep(
        write_vtk_fields,
        vtkOutputDir,
        absorption,
        material,
    )
    if openPmdOutputDir is not None:
        simulation.onStep(writeParaviewState, openPmdOutputDir, absorption)
    simulation.step(timeSteps, pumpSteps=pumpSteps)
    return simulation.getLastState()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Modern HASEonGPU laser-pump cladding example")
    parser.add_argument("--backend")
    parser.add_argument("--openpmd-backend")
    parser.add_argument("--time-steps", type=int, default=150)
    parser.add_argument(
        "--pump-steps",
        type=int,
        default=100,
        help=(
            "Number of outer simulation steps with pump contribution. "
            "Default: 100. Use a value matching --time-steps to pump for the full run. "
            "This is distinct from MonteCarloPumpSolver.rayCount, which controls "
            "the Monte Carlo pump sampling resolution."),
    )
    parser.add_argument(
        "--phi-ase-config",
        type=Path,
        default=DEFAULT_PHI_ASE_CONFIG_PATH,
        help="Monte Carlo ASE run-control YAML. Defaults to config/hase-phiase.yaml.",
    )
    parser.add_argument("--vtk-output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--openpmd-output-dir", type=Path, default=None)
    parser.add_argument(
        "--disable-ase",
        action="store_true",
        help="Disable ASE depletion during the time-stepped pump simulation.",
    )
    parser.add_argument(
        "--disable-pre-pump",
        action="store_true",
        help="Run ASE during the first pump time step instead of seeding beta without ASE.",
    )
    parser.add_argument("--rng-seed", type=int, default=None)
    parser.add_argument(
        "--pump-ray-count", type=int, default=50000,
        help="Equal-power launch rays per pump source. Default: 50000.",
    )
    parser.add_argument("--pump-rng-seed", type=int, default=5489)
    parser.add_argument(
        "--spectral-resolution",
        type=int,
        default=1000,
        help="Frontend material cross-section interpolation resolution. Default: 1000.",
    )
    parser.add_argument(
        "--disable-reflections",
        action="store_true",
        help="Disable ASE surface reflections.",
    )
    parser.add_argument(
        "--timings",
        action="store_true",
        help="Print frontend timing split for compiled transport, snapshots, and callbacks.",
    )
    args = parser.parse_args(argv)

    aseOverrides = {}
    if args.rng_seed is not None:
        aseOverrides["rngSeed"] = args.rng_seed
    if args.disable_reflections:
        aseOverrides["useReflections"] = False

    state = run_example(
        phiAseConfigPath=args.phi_ase_config,
        backend=args.backend,
        timeSteps=args.time_steps,
        pumpSteps=args.pump_steps,
        vtkOutputDir=args.vtk_output_dir,
        openPmdOutputDir=args.openpmd_output_dir,
        openPmdBackend=args.openpmd_backend,
        enableAse=not args.disable_ase,
        prePump=not args.disable_pre_pump,
        spectralResolution=args.spectral_resolution,
        reportTimings=args.timings,
        pumpRayCount=args.pump_ray_count,
        pumpRngSeed=args.pump_rng_seed,
        **aseOverrides,
    )
    print(f"phiAse shape: {state.phiAse.shape}")
    print(f"betaCells shape: {state.sampledExcitationFraction.shape}")


if __name__ == "__main__":
    main()
