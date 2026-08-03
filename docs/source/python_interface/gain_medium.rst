Material and boundary physics
=============================

HASEonGPU separates the material database from the condition and placement
used in one simulation. Explicit units are required for dimensional
quantities; see :doc:`physical_quantities` and :doc:`material_library`.

Active material properties
--------------------------

``Material`` contains temperature-resolved optical properties. ``at`` selects
one state and supplies the active-ion density of one simulated part:

.. code-block:: python

   yag = Material("Yb:YAG").addState(
       temperature=300 * units.K,
       refractiveIndex=1.82,
       fluorescenceLifetime=0.941 * units.ms,
       crossSections=spectra,
   )
   crystal = yag.at(temperature=300 * units.K,
                    activeIonDensity=2.76e20 / units.cm**3)

The properties have distinct physical roles:

``refractiveIndex``
   The dimensionless interior optical index. The current boundary transport
   uses it with the exterior index to detect total internal reflection.

``fluorescenceLifetime``
   The upper-state lifetime :math:`\tau`. It controls spontaneous population
   decay :math:`-\beta/\tau` and the physical scaling of ASE flux.

``crossSections``
   The material's absorption and stimulated-emission probabilities per active
   ion as functions of wavelength. See :doc:`spectral_decomposition`.

``activeIonDensity``
   The total active-ion density :math:`N_{\mathrm{tot}}`, not only the excited
   population. The excited density is :math:`\beta N_{\mathrm{tot}}`.

For wavelength :math:`\lambda`, these values and the excitation fraction form
the local gain coefficient

.. math::

   g(\lambda) = N_{\mathrm{tot}}
   \left[\beta\left(\sigma_e(\lambda)+\sigma_a(\lambda)\right)
   - \sigma_a(\lambda)\right].

Positive :math:`g` amplifies a ray and negative :math:`g` attenuates it.

Passive and transparent materials
---------------------------------

A zero-density material does not have active-ion gain. A non-zero
``bulkAttenuation`` describes passive exponential loss per unit length:

.. code-block:: python

   glass = Material("glass").addState(
       temperature=300 * units.K,
       refractiveIndex=1.50,
       bulkAttenuation=0.02 / units.cm,
   )
   cap = glass.at(temperature=300 * units.K)

With zero density and zero attenuation, a condition is transparent. These
states are exposed as ``isActive``, ``isPassive``, and ``isTransparent``.
The frontend can compile passive attenuation, but the current native adapter
rejects that feature before launch. ``opticalAxis`` similarly records a
normalized material orientation that the current adapter does not execute.

Material placement
------------------

Attach a material to a selection owned by the mesh:

.. code-block:: python

   simulation.addMaterial(crystal, domains=mesh.volume("core"))
   simulation.addMaterial(cap, domains=mesh.volume("left_cap", "right_cap"))

Every cell must be selected exactly once. Reusing one ``MaterialCondition`` on
several selections reuses one compiled material ID; separately selected
conditions retain separate identity even when their numerical properties
match.

Exterior optical boundaries
----------------------------

``AbsorbingSurface`` terminates incident transport at the domain boundary.
``ConstantReflectivitySurface`` returns a constant fraction :math:`R` of the
incident ray weight and lets the remaining fraction leave the model:

.. code-block:: python

   simulation.addBoundary(AbsorbingSurface(), domains=mesh.surface("side"))
   simulation.addBoundary(ConstantReflectivitySurface(0.04, 1.0), domains=mesh.surface("ends"))

``reflectivity`` is dimensionless and lies in ``[0, 1]``.
``exterior_refractive_index`` is dimensionless and is used with the material
index for total internal reflection. The current ASE model does not calculate
angle- or polarization-dependent Fresnel coefficients and does not launch a
transmitted/refracted ray. ASE reflection also requires
``MonteCarloASESolver(useReflections=True)``.

Every exterior face must have exactly one boundary registration. Boundary
behavior and pump injection are independent even when they select the same
surface.

Internal interfaces
-------------------

Every neighboring pair of unlike material conditions needs an explicit model:

.. code-block:: python

   simulation.addInterface(PerfectTransmission(), between=(crystal, cap))

``PerfectTransmission`` and ``FresnelInterface`` express frontend intent. The
current native adapter rejects internal cross-material transport before launch.

Initial excitation
------------------

``InitialState`` supplies the dimensionless upper-state fraction :math:`\beta`
as one scalar, one value per Tet4 cell, or a complete selection map:

.. code-block:: python

   initial = InitialState({mesh.volume("core"): 0.10 * units.one,
                           mesh.volume("left_cap", "right_cap"): 0.0 * units.one})

Every value must be finite and within ``[0, 1]``. A selection map must cover
each cell exactly once without overlap.
