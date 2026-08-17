# Copyright 2026 Tim Hanel
#
# This file is part of HASEonGPU
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Small programmatic example of the five-object frontend graph."""

from _source_tree_import import ensure_hase_importable

ensure_hase_importable()

from HASEonGPU import (  # noqa: E402
    CrossSectionTable,
    Domain,
    GainMedium,
    Material,
    OpticalComponent,
    PhiASE,
    Pump,
    PumpSpectrum,
    RungeKutta4,
    Simulation,
    SurfacePumpInjector,
    VolumeTopology,
    units,
)


def printState(state):
    print(
        f"step={state.step:03d} time={state.time:.3e}s "
        f"mean_beta={state.betaVolume.mean():.6e} "
        f"mean_phi={state.phiAse.mean():.6e}"
    )


def main():
    topology = VolumeTopology.fromTetrahedra(
        points=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        cellPointIndices=[[0, 1, 2, 3], [1, 2, 3, 4]],
    )
    material = Material(
        materialName="example gain material",
        temperature=293.15 * units.K,
        refractiveIndex=1.83,
        fluorescenceLifetime=0.941 * units.ms,
        crossSections=CrossSectionTable(
            [900.0, 1030.0] * units.nm,
            [1.1e-21, 1.0e-22] * units.cm**2,
            [1.0e-22, 2.48e-20] * units.cm**2,
        ),
        activeIonDensity=2.776e20 / units.cm**3,
    )
    component = OpticalComponent(
        domain=Domain.fromTopology(topology),
        material=material,
    )
    gainMedium = GainMedium([component])
    pumpInput = Domain.where(topology, "z_min")
    pump = Pump(
        total_power=16_000.0,
        spectrum=PumpSpectrum.monochromatic(940e-9),
        ray_count=100_000,
        pump_steps=3,
    )
    simulation = Simulation(
        opticalComponents=[component],
        gainMedium=gainMedium,
        initialExcitation=0.0,
        phiASE=PhiASE(
            forwardRayCount=1000,
            repetitions=1,
            backend="Host_Cpu_CpuSerial",
            ase_steps=3,
        ),
        timeIntegrator=RungeKutta4(),
        timeStepSize=1e-5,
        simulationSteps=3,
    ).addPump(pump, SurfacePumpInjector(pumpInput))
    simulation.onStep(printState)
    simulation.step()
    print(f"last completed step: {simulation.getLastState().step}")


if __name__ == "__main__":
    main()
