# Copyright 2026 Tim Hanel
# SPDX-License-Identifier: GPL-3.0-or-later

import inspect

import numpy as np
import pytest

from HASEonGPU import (
    GaussianPump,
    PlanarPumpRelay,
    Pump,
    PumpAngularDistribution,
    PumpSpectrum,
    Simulation,
    SuperGaussianPumpProfile,
    SurfacePumpInjector,
    UniformPumpProfile,
)


def test_core_class_and_method_names_follow_public_convention():
    assert Simulation.__name__[0].isupper()
    assert tuple(inspect.signature(Simulation).parameters)[:5] == (
        "opticalComponents",
        "gainMedium",
        "phiASE",
        "timeIntegrator",
        "timeStepSize",
    )


def test_gaussian_pump_owns_sampling_and_stays_separate_from_injection():
    pump = GaussianPump(
        total_power=12.5,
        spectrum=PumpSpectrum.monochromatic(940e-9),
        ray_count=1234,
        pump_steps=7,
        rng_seed=99,
        waist=(1.5, 1.25),
        exponent=40,
        angular_distribution=PumpAngularDistribution.collimated(),
        name="lower_pump",
    )
    injector = SurfacePumpInjector(surface_domains=("lower",))

    assert pump.total_power == 12.5
    assert pump.profile.radius_u == 1.5
    assert pump.profile.radius_v == 1.25
    assert pump.profile.weightAt([[0.0, 0.0, 0.0]])[0] == pytest.approx(1.0)
    np.testing.assert_array_equal(pump.spectrum.weights, [1.0])
    assert pump.ray_count == 1234
    assert pump.pump_steps == 7
    assert pump.rng_seed == 99
    assert injector.surface_domains == ("lower",)


def test_uniform_cone_uses_lower_camel_sampling_controls():
    distribution = PumpAngularDistribution.uniformCone(
        np.pi / 6.0,
        polarSamples=2,
        azimuthalSamples=3,
    )
    assert distribution.weights.size == 6
    assert distribution.weights.sum() == pytest.approx(1.0)
    assert np.all(distribution.polar_angles < np.pi / 6.0)


def test_simulation_step_uses_constructor_steps_by_default():
    assert inspect.signature(Simulation.step).parameters["nsteps"].default is None
