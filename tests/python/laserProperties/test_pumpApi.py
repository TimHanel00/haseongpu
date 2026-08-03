# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: GPL-3.0-or-later

import inspect

import numpy as np
import pytest

from HASEonGPU import (
    GaussianPump,
    MonteCarloPumpSolver,
    PlanarPumpRelay,
    Pump,
    PumpAngularDistribution,
    PumpSpectrum,
    Simulation as PublicSimulation,
    SuperGaussianPumpProfile,
    SurfacePumpInjector,
    UnstructuredMesh,
    UniformPumpProfile,
    units,
)

def test_public_pump_and_simulation_signatures_use_lower_camel_case():
    public_classes = (
        GaussianPump,
        MonteCarloPumpSolver,
        PlanarPumpRelay,
        Pump,
        PumpAngularDistribution,
        PumpSpectrum,
        PublicSimulation,
        SuperGaussianPumpProfile,
        SurfacePumpInjector,
        UniformPumpProfile,
    )
    for cls in public_classes:
        for name in inspect.signature(cls).parameters:
            assert "_" not in name, (cls.__name__, name)


def test_gaussian_pump_keeps_physics_separate_from_injection_and_solver():
    mesh = UnstructuredMesh.fromTetrahedra(
        np.eye(4, 3),
        [[0, 1, 2, 3]],
        surfaceDomains=[[1, 1, 1, 1]],
        coordinateUnit=units.cm,
    )
    pump = GaussianPump(
        totalPower=12.5 * units.W,
        spectrum=PumpSpectrum.monochromatic(940 * units.nm),
        waist=np.asarray([1.5, 1.25]) * units.cm,
        exponent=40,
        angularDistribution=PumpAngularDistribution.collimated(),
        name="lower_pump",
    )
    injector = SurfacePumpInjector(mesh.surface(1))
    solver = MonteCarloPumpSolver(rayCount=1234, seed=99, maxSteps=4)

    assert pump.totalPower.toValue(units.W) == pytest.approx(12.5)
    assert pump.profile.radiusU.toValue(units.cm) == pytest.approx(1.5)
    assert pump.profile.radiusV.toValue(units.cm) == pytest.approx(1.25)
    assert pump.profile.weightAt([[0.0, 0.0, 0.0]], units.cm)[0] == pytest.approx(1.0)
    np.testing.assert_array_equal(pump.spectrum.weights, [1.0])
    assert injector.surface.names == ("1",)
    assert solver == MonteCarloPumpSolver(rayCount=1234, seed=99, maxSteps=4)
    assert "crossSections" not in inspect.signature(Pump).parameters
    assert not hasattr(pump, "crossSections")


def test_uniform_cone_uses_lower_camel_case_sampling_controls():
    distribution = PumpAngularDistribution.uniformCone(
        np.pi / 6.0,
        polarSamples=2,
        azimuthalSamples=3,
    )
    assert distribution.weights.size == 6
    assert distribution.weights.sum() == pytest.approx(1.0)
    assert np.all(distribution.polarAngles < np.pi / 6.0)


def test_simulation_step_signature_matches_picmi_default():
    assert inspect.signature(PublicSimulation.step).parameters["numberOfSteps"].default == 1
