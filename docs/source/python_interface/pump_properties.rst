Physical pump light and numerical injection
===========================================

The pump API separates physical light, its injection geometry, optional relay
optics, and Monte Carlo sampling. This keeps optical quantities out of the
solver controls.

Total power and spectrum
------------------------

``Pump.totalPower`` is the aperture-integrated incident optical power, not a
power density and not a per-ray value. ``PumpSpectrum`` is a discrete set of
wavelengths whose non-negative weights are normalized to sum to one:

.. code-block:: python

   spectrum = PumpSpectrum(units.nm * [930, 940, 950], weights=[0.2, 0.6, 0.2])
   pump = Pump(totalPower=16 * units.kW, spectrum=spectrum)

For one wavelength, use:

.. code-block:: python

   spectrum = PumpSpectrum.monochromatic(940 * units.nm)

The sampled wavelength selects absorption and emission cross sections from the
traversed material. A ``Pump`` intentionally has no cross-section property.

Spatial profile
---------------

``UniformPumpProfile`` distributes power uniformly over the selected launch
aperture. ``SuperGaussianPumpProfile`` evaluates

.. math::

   w(u,v) = \exp\!\left[-\left(
   (u/r_u)^2 + (v/r_v)^2\right)^{m/2}\right],

where ``radiusU`` and ``radiusV`` are physical length scales, ``exponent`` is
:math:`m`, and ``axisU``/``axisV`` define the world-space ellipse:

.. code-block:: python

   profile = SuperGaussianPumpProfile(1.5 * units.cm,
                                      radiusV=1.0 * units.cm, exponent=8)

At one radius along either principal axis the profile is :math:`e^{-1}`.
``center`` is a three-component length quantity in mesh coordinates. The
profile controls relative sampling density; ``totalPower`` remains the
integrated power assigned to the pump.

``GaussianPump`` is a convenience constructor using ``exponent=2`` by default:

.. code-block:: python

   pump = GaussianPump(totalPower=16 * units.kW, spectrum=spectrum,
                       waist=1.5 * units.cm)

Angular distribution
--------------------

Directions are specified in the injector's inward-local frame. Polar angle is
measured away from the inward surface normal, azimuth is around that normal,
and both are in radians. Weights are normalized probabilities:

.. code-block:: python

   direction = PumpAngularDistribution.collimated()
   cone = PumpAngularDistribution.uniformCone(np.deg2rad(5.0))

The allowed polar range is ``[0, pi/2)``; the injector launches into the mesh.

Injection surfaces
------------------

Injection is numerical placement of a physical ``Pump``. Select one or more
tagged exterior faces and register them:

.. code-block:: python

   aperture = mesh.surface("lower_left", "lower_right")
   simulation.addPump(pump, SurfacePumpInjector(aperture))

The second argument is named ``injectionMethod`` because it couples two
otherwise independent descriptions:

* ``pump`` defines *what light exists*: integrated power, spectrum, transverse
  irradiance shape, and angular probabilities.
* ``injectionMethod`` defines *where that light enters*: the selected exterior
  triangles form the physical launch aperture.

For every selected triangle, the mesh's oriented exterior normal determines
the opposite inward direction. A tangent basis on the face provides the local
``u`` and ``v`` axes used by the profile and angular distribution. HASE
integrates the relative profile over all selected triangles and normalizes the
launch weights so their sum represents exactly ``pump.totalPower``. Therefore
``totalPower`` is neither multiplied by the face count nor interpreted as an
irradiance.

The same faces still need an exterior boundary model:

.. code-block:: python

   simulation.addBoundary(AbsorbingSurface(), domains=aperture)

Injection does not implicitly make a face absorbing, reflective, or
transparent. The boundary model governs a ray that later reaches the face;
the injector governs creation of the initial pump histories.

Planar relays
-------------

``PlanarPumpRelay`` maps rays leaving one coplanar aperture onto another. It
can flip local axes, rotate, translate, tilt, magnify, attenuate, and vignette
the relayed beam:

.. code-block:: python

   relay = PlanarPumpRelay.retroreflect(mesh.surface("upper"), transmission=0.98)
   simulation.addPump(pump, SurfacePumpInjector(aperture), relays=(relay,))

``rotation`` and ``tilt`` are in radians, ``offset`` is expressed in the mesh's
coordinate unit, and ``magnification`` and ``transmission`` are dimensionless.
This is an affine return path, not a general curved-optics or Fresnel model.

Numerical ray sampling
----------------------

``MonteCarloPumpSolver`` controls how the physical pump is sampled:

.. code-block:: python

   pumpSolver = MonteCarloPumpSolver(rayCount=100_000, seed=5489)

``rayCount`` changes Monte Carlo resolution, not total pump power. ``seed``
makes the unsigned 32-bit random stream reproducible. ``maxSteps`` and
``Simulation.step(..., pumpSteps=N)`` limit how many outer time steps include
pumping; they do not change the time-step duration.
