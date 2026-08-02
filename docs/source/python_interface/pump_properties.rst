Pumps and pump solvers
======================

A ``Pump`` describes physical incident light only: power, wavelength spectrum,
spatial profile, and angular distribution. Cross sections are material
properties and are not repeated on a pump.

.. code-block:: python

   pump = Pump(
       total_power=100_000.0,
       spectrum=PumpSpectrum.monochromatic(940e-9),
       profile=SuperGaussianPumpProfile(
           radius_u=1.5, radius_v=1.5, exponent=40
       ),
   )
   simulation.add_pump(pump, SurfacePumpInjector("pump_input"))

``PlanarPumpRelay`` can describe explicit affine re-imaging between boundary
domains. ``GaussianPump`` is a convenience constructor.

``PumpSolver`` is the extension role for pumping algorithms.
``MonteCarloPumpSolver(ray_count=..., seed=..., max_steps=...)`` is the only
solver currently wired to the native backend. Physical pump definitions and
solver selection remain separate so another pumping algorithm can be added
without changing material or mesh APIs.
