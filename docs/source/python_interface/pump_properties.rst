Pumps and pump solvers
======================

The frontend separates three different concerns:

* ``Pump`` describes physical incident light;
* ``SurfacePumpInjector`` and optional ``PlanarPumpRelay`` objects describe
  where that light enters and how it is re-imaged;
* ``MonteCarloPumpSolver`` controls the numerical sampling shared by registered
  pumps.

Physical pump light
-------------------

.. code-block:: python

   pump = Pump(
       total_power=100_000.0,
       spectrum=PumpSpectrum.monochromatic(940e-9),
       profile=SuperGaussianPumpProfile(
           radius_u=1.5, radius_v=1.5, exponent=40
       ),
       angular_distribution=PumpAngularDistribution.collimated(),
       name="lower_pump",
   )

Power is in watts and wavelengths are in metres. Spectrum and angular weights
are normalized by their constructors. ``UniformPumpProfile`` is the default;
``GaussianPump`` is a convenience constructor for an elliptical
``SuperGaussianPumpProfile``.

Cross sections are intentionally absent from ``Pump``. Absorption and emission
are properties of the material being traversed and come from its
``CrossSectionTable``.

Injection and relays
--------------------

Register a pump on one or more named or numeric exterior surface domains:

.. code-block:: python

   simulation.add_pump(
       pump,
       SurfacePumpInjector(("lower_left", "lower_right")),
       relays=(
           PlanarPumpRelay.retroreflect("upper", transmission=0.98),
       ),
   )

The injector's domains must identify exterior faces with positive surface tags.
They need not form a separate ``BoundaryLayout``: injection and exterior
optical behavior are independent registrations.

``PlanarPumpRelay`` maps an exit aperture to an entry aperture with optional
axis flips, rotation, offset, tilt, magnification, and scalar transmission. It
does not model a general curved optical system.

Numerical sampling
------------------

``MonteCarloPumpSolver(ray_count=..., seed=..., max_steps=...)`` sets the number
of pump rays per evaluation, the reproducible unsigned 32-bit seed, and an
optional pump-only step limit. ``Simulation.step(..., pump_steps=N)`` may also
limit pumping for one call.

``PumpSolver`` is an extension role. Other descriptors can be composed in the
frontend, but the current native adapter rejects them before launch.
