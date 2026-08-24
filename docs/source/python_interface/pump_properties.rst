Pump configuration
==================

The pump API separates a physical source from its placement:

* ``Pump`` owns power, wavelength distribution, profile, angular distribution,
  ray count, seed, and active steps;
* ``SurfacePumpInjector`` selects a typed surface ``Domain``;
* ``PlanarPumpRelay`` describes an optional finite return path.

.. code-block:: python

   profile = SuperGaussianPumpProfile(
       radius_u=0.015,
       radius_v=0.015,
       exponent=40,
   )
   pump = Pump(
       total_power=16_000.0,
       spectrum=PumpSpectrum.monochromatic(940e-9),
       ray_count=50_000,
       pump_steps=50,
       rng_seed=5489,
       angular_distribution=PumpAngularDistribution.collimated(),
       profile=profile,
   )
   simulation.addPump(
       pump,
       SurfacePumpInjector(pumpInput),
       relays=(PlanarPumpRelay.retroreflect(pumpOutput),),
   )

The pump's material interaction comes from the shared executable ``Material``;
it does not carry a second physical cross-section registry. ``PumpSpectrum``
only selects and weights pump wavelengths.

Pump wavelengths, profile radii and centres, and relay offsets use metres.
``Pump.total_power`` uses watts. Angles and profile axes are dimensionless.

``Pump.total_power`` is integrated aperture power. To convert a peak power
density, integrate the profile over its typed surface domain:

.. code-block:: python

   area = integrate_pump_profile(topology, pumpInput, profile)
   pump = Pump(
       total_power=peakPowerDensity * area,
       spectrum=PumpSpectrum.monochromatic(940e-9),
       ray_count=50_000,
       profile=profile,
   )

Here ``area`` is in square metres, so ``peakPowerDensity`` must be in watts per
square metre. For example, :math:`16\,\mathrm{kW\,cm^{-2}}` is
:math:`1.6\times10^8\,\mathrm{W\,m^{-2}}`.

``PlanarPumpRelay`` supports flips, in-plane rotation, offset, tilt,
magnification, transmission, and aperture vignetting. ``transmission`` is the
retained power fraction of that finite relay mapping; it is not transmission
through the component boundary.

The compiled solver samples launch position, wavelength, and direction, then
traverses neighboring Tet4 cells to a physical boundary. See
:ref:`general-monte-carlo-pump` for the transport and population-rate model.
