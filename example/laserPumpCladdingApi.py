# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Programmatic construction of the schema-v3 laser-pump example."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_tree_import import ensure_hase_importable


scriptDir = Path(__file__).resolve().parent
materialDatabase = scriptDir.parent / "material_library" / "data" / "materials.h5"

ensure_hase_importable()

from HASEonGPU import (  # noqa: E402
    Domain,
    FrozenPhiAseRungeKutta4,
    GainMedium,
    Material,
    OpticalComponent,
    PhiASE,
    PlanarPumpRelay,
    Pump,
    PumpAngularDistribution,
    PumpSpectrum,
    Simulation,
    SuperGaussianPumpProfile,
    SurfaceOptics,
    SurfacePumpInjector,
    VolumeTopology,
    calcGainFromState,
    integrate_pump_profile,
    units,
    vtkWedge,
)
from pyInclude.openpmd.paraview import writeParaviewState  # noqa: E402


def laserPumpCladdingMaterial(spectralResolution=1000):
    """Resolve the mutable Yb:YAG material used by the example."""
    return Material.fromHdf5(
        materialDatabase,
        key="YbYAG",
        temperature=293.15 * units.K,
        activeIonDensity=2.776e20 / units.cm**3,
        spectralResolution=spectralResolution,
        interpolation="exact",
        name="yb_yag",
    )


def laserPumpCladdingComponent(material=None):
    """Build the crystal and return it with its typed pump boundary domains."""
    material = laserPumpCladdingMaterial() if material is None else material
    topology = VolumeTopology.fromFile(scriptDir / "data" / "ptTet4.vtk", format="vtk")
    volume = Domain.fromTopology(topology)
    allExterior = Domain.where(topology, "all_exterior")
    bottom = Domain.where(topology, "z_min")
    top = Domain.where(topology, "z_max")
    side = allExterior - bottom - top

    component = OpticalComponent(domain=volume, material=material, name="crystal")
    component.assignSurfaceOptics(
        side,
        SurfaceOptics(reflectivity=0.0, n_inside=1.0, n_outside=1.0),
    )
    for domain in (bottom, top):
        component.assignSurfaceOptics(
            domain,
            SurfaceOptics(
                reflectivity=0.0,
                n_inside=material.refractiveIndex,
                n_outside=1.0,
            ),
        )
    return component, bottom, top


def buildSimulation(
    backend="Host_Cpu_CpuOmpBlocks",
    simulationSteps=150,
    pumpSteps=50,
    aseSteps=150,
    openpmdBackend="auto",
    prePump=True,
    spectralResolution=1000,
    pumpRayCount=50000,
    pumpRngSeed=5489,
    reportTimings=False,
    outputSteps=None,
    **aseOverrides,
):
    """Construct the complete five-object physical graph without running it."""
    material = laserPumpCladdingMaterial(spectralResolution)
    component, bottom, top = laserPumpCladdingComponent(material)
    gainMedium = GainMedium([component], name="amplifier")
    phiAseParameters = {
        "propagationMode": "forward",
        "minRays": 10000,
        "maxRays": 1000000,
        "relativeStandardErrorThreshold": 0.1,
        "repetitions": 4,
        "adaptiveSteps": 4,
        "useReflections": True,
        "reflectionMaxIterations": 40,
        "reflectionTolerance": 1.0e-4,
        "surfaceReservoirSize": 32,
        "monochromatic": False,
        "backend": backend,
        "openpmdBackend": openpmdBackend,
        "parallelMode": "single",
        "numDevices": 1,
        "nPerNode": 1,
        "ase_steps": aseSteps,
    }
    phiAseParameters.update(aseOverrides)

    pumpProfile = SuperGaussianPumpProfile(
        radius_u=1.5,
        radius_v=1.5,
        exponent=40,
    )
    profileArea = integrate_pump_profile(
        component.domain.topologies[0],
        bottom,
        pumpProfile,
    )
    pump = Pump(
        total_power=16e3 * profileArea,
        spectrum=PumpSpectrum.monochromatic(940e-9),
        ray_count=pumpRayCount,
        pump_steps=pumpSteps,
        rng_seed=pumpRngSeed,
        angular_distribution=PumpAngularDistribution.collimated(),
        profile=pumpProfile,
    )
    simulation = Simulation(
        opticalComponents=[component],
        gainMedium=gainMedium,
        initialExcitation=0.0,
        phiASE=PhiASE(**phiAseParameters),
        timeIntegrator=FrozenPhiAseRungeKutta4(),
        timeStepSize=2e-5,
        simulationSteps=simulationSteps,
        prePump=prePump,
        reportTimings=reportTimings,
        outputSteps=None if outputSteps is None else tuple(int(step) for step in outputSteps),
    )
    return simulation.addPump(
        pump,
        SurfacePumpInjector(bottom),
        relays=(PlanarPumpRelay.retroreflect(top),),
    )


def printState(state):
    print(
        f"step={state.step:03d} "
        f"time={state.time:.3e}s "
        f"mean_beta={state.betaVolume.mean():.6e} "
        f"mean_phi={state.phiAse.mean():.6e}"
    )


def _writeScalarArray(handle, name, values, count):
    arr = np.asarray(values).reshape(-1, order="F")
    if arr.size != count:
        raise ValueError(f"{name} has {arr.size} values, expected {count}")
    handle.write(f"SCALARS {name} double 1\n")
    handle.write("LOOKUP_TABLE default\n")
    for value in arr:
        handle.write(f"{float(value):.17g}\n")


def _writeTet4StateVtk(path, state, fields):
    topology = state.topology
    points = np.asarray(topology.points, dtype=np.float64)
    cells = np.asarray(topology.cellPointIndices, dtype=np.uint32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pointCount = points.shape[0]
    cellCount = cells.shape[0]
    pointFields = {name: value for name, value in fields.items() if np.asarray(value).size == pointCount}
    cellFields = {name: value for name, value in fields.items() if np.asarray(value).size == cellCount}
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# vtk DataFile Version 2.0\n")
        handle.write("HASEonGPU laserPumpCladding Tet4 state\n")
        handle.write("ASCII\nDATASET UNSTRUCTURED_GRID\n")
        handle.write(f"POINTS {pointCount} double\n")
        for x, y, z in points:
            handle.write(f"{x:.17g} {y:.17g} {z:.17g}\n")
        handle.write(f"CELLS {cellCount} {cellCount * 5}\n")
        for cell in cells:
            handle.write("4 " + " ".join(str(int(vertex)) for vertex in cell) + "\n")
        handle.write(f"CELL_TYPES {cellCount}\n")
        handle.write("10\n" * cellCount)
        if pointFields:
            handle.write(f"POINT_DATA {pointCount}\n")
            for name, values in pointFields.items():
                _writeScalarArray(handle, name, values, pointCount)
        if cellFields:
            handle.write(f"CELL_DATA {cellCount}\n")
            for name, values in cellFields.items():
                _writeScalarArray(handle, name, values, cellCount)
    return path


def writeVtkFields(state, vtkOutputDir=scriptDir, claddingAbsorption=1.0, crossSections=None, nTot=None):
    if state.phiAse is None:
        raise ValueError("VTK export requires state.phiAse")
    if crossSections is None or nTot is None:
        raise ValueError("VTK export requires crossSections and nTot for gain")
    fields = {
        "betaVolume": state.betaVolume,
        "phiASE": state.phiAse,
        "dndtAse": state.dndtAse,
        "dndtPump": state.dndtPump,
        "cladAbs": state.phiAse * np.float64(claddingAbsorption),
        "localGain": calcGainFromState(state, crossSections, nTot),
    }
    path = Path(vtkOutputDir) / f"laserPumpCladding_{state.step:03d}.vtk"
    if hasattr(state.topology, "cellPointIndices"):
        return _writeTet4StateVtk(path, state, fields)
    return vtkWedge(path, state, fields=fields)


def runExample(
    backend="Host_Cpu_CpuOmpBlocks",
    simulationSteps=150,
    pumpSteps=50,
    aseSteps=150,
    vtkOutputDir=scriptDir,
    openPmdOutputDir=None,
    openpmdBackend="auto",
    prePump=True,
    spectralResolution=1000,
    pumpRayCount=50000,
    pumpRngSeed=5489,
    reportTimings=False,
    outputSteps=None,
    **aseOverrides,
):
    simulation = buildSimulation(
        backend=backend,
        simulationSteps=simulationSteps,
        pumpSteps=pumpSteps,
        aseSteps=aseSteps,
        openpmdBackend=openpmdBackend,
        prePump=prePump,
        spectralResolution=spectralResolution,
        pumpRayCount=pumpRayCount,
        pumpRngSeed=pumpRngSeed,
        reportTimings=reportTimings,
        outputSteps=outputSteps,
        **aseOverrides,
    )
    material = simulation.gainMedium.components[0].material
    nTot = material.activeIonDensity.toValue(units.cm**-3)
    print(f"Running simulation with backend {simulation.phiASE.backend}")
    print(f"Using openPMD backend {simulation.phiASE.openpmdBackend}")
    simulation.onStep(printState)
    simulation.onStep(
        writeVtkFields,
        Path(vtkOutputDir),
        5.5,
        simulation.crossSections,
        nTot,
    )
    if openPmdOutputDir is not None:
        simulation.onStep(writeParaviewState, Path(openPmdOutputDir), 5.5)
    simulation.step()
    return simulation.getLastState()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inline-API HASEonGPU laser-pump cladding example")
    parser.add_argument("--backend", default="Host_Cpu_CpuOmpBlocks")
    parser.add_argument("--openpmd-backend", default="auto")
    parser.add_argument("--simulation-steps", type=int, default=150)
    parser.add_argument("--output-steps", type=int, nargs="+", default=None)
    parser.add_argument("--pump-steps", type=int, default=50)
    parser.add_argument("--ase-steps", type=int, default=150)
    parser.add_argument("--vtk-output-dir", type=Path, default=scriptDir)
    parser.add_argument("--openpmd-output-dir", type=Path, default=None)
    parser.add_argument("--disable-pre-pump", action="store_true")
    parser.add_argument("--rng-seed", type=int, default=None)
    parser.add_argument("--pump-ray-count", type=int, default=50000)
    parser.add_argument("--pump-rng-seed", type=int, default=5489)
    parser.add_argument("--spectral-resolution", type=int, default=1000)
    parser.add_argument("--timings", action="store_true")
    args = parser.parse_args(argv)

    overrides = {} if args.rng_seed is None else {"rngSeed": args.rng_seed}
    state = runExample(
        args.backend,
        simulationSteps=args.simulation_steps,
        pumpSteps=args.pump_steps,
        aseSteps=args.ase_steps,
        vtkOutputDir=args.vtk_output_dir,
        openPmdOutputDir=args.openpmd_output_dir,
        openpmdBackend=args.openpmd_backend,
        prePump=not args.disable_pre_pump,
        spectralResolution=args.spectral_resolution,
        reportTimings=args.timings,
        pumpRayCount=args.pump_ray_count,
        pumpRngSeed=args.pump_rng_seed,
        outputSteps=args.output_steps,
        **overrides,
    )
    print(f"phiAse shape: {state.phiAse.shape}")
    print(f"betaVolume shape: {state.betaVolume.shape}")


if __name__ == "__main__":
    main()
