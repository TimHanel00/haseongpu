Materials and physical units
============================

A material library stores optical properties that belong to a material and a
temperature. A simulation uses a resolved ``Material``: one selected library
state combined with run-specific quantities such as active-ion density and
optical-axis orientation.

This distinction prevents a database entry such as Yb:YAG from fixing the
dopant concentration of every crystal made from it. It also keeps measured
spectra and their provenance together instead of distributing four unlabelled
arrays across the simulation configuration.

Resolve a bundled material
--------------------------

The bundled database contains the active room-temperature Yb:YAG record and
the passive absorbing glass used by the laser-pump example:

.. code-block:: python

   from HASEonGPU import units
   from material_library import loadBuiltinMaterials

   material = loadBuiltinMaterials().resolve(
       "YbYAG",
       temperature=293.15 * units.K,
       activeIonDensity=2.776e20 / units.cm**3,
   )

``"YbYAG"`` is the stable database key. ``material.materialName`` is the
physical record name, while the optional ``material.name`` is a run-specific
label. Neither name changes the physical coefficients.

The resolved object is mutable so a simulation can apply a measured
run-specific correction. Call ``validate()`` after direct edits:

.. code-block:: python

   material.refractiveIndex = 1.829
   material.validate()

Construct an active material directly
-------------------------------------

Small studies and material-data preparation can construct a condition without
an HDF5 library:

.. code-block:: python

   from HASEonGPU import CrossSectionTable, Material, units

   spectra = CrossSectionTable.monochromatic(
       wavelength=1030 * units.nm,
       absorption=1.0e-21 * units.cm**2,
       emission=2.0e-20 * units.cm**2,
   )
   material = Material(
       materialName="Yb:YAG test condition",
       temperature=293.15 * units.K,
       refractiveIndex=1.83,
       fluorescenceLifetime=0.941 * units.ms,
       crossSections=spectra,
       active=True,
       activeIonDensity=2.776e20 / units.cm**3,
   )

``active=True`` classifies the material as participating in active-ion
population dynamics. The classification belongs to the material record and is
independent of ``activeIonDensity``. The density quantifies the available ions
for one crystal condition; it does not switch the rate equations on or off.

An active material requires a fluorescence lifetime, cross sections, and
non-zero emission data. A library record may be resolved with zero density for
inspection or subsequent parameterization. Before that material is placed in a
``GainMedium``, the run condition must supply a positive density.
``bulkAttenuation`` is not required for a gain material. The current backend
does not apply a non-zero passive loss coefficient in active gain cells.

Passive materials and bulk attenuation
--------------------------------------

A passive material is constructed with ``active=False``. It has zero
``activeIonDensity`` and is excluded from population excitation. An absorbing
cladding can therefore be described without fluorescence or spectroscopic
cross sections:

.. code-block:: python

   claddingMaterial = Material(
       materialName="absorbing cladding",
       temperature=293.15 * units.K,
       refractiveIndex=1.45,
       fluorescenceLifetime=None,
       crossSections=None,
       active=False,
       bulkAttenuation=5.5 / units.cm,
   )

``bulkAttenuation`` is the wavelength-independent passive intensity-loss
coefficient :math:`\alpha_\mathrm{bulk}`. Along a homogeneous path of length
:math:`\ell`, the transported intensity follows

.. math::

   I(\ell) = I(0)\exp(-\alpha_\mathrm{bulk}\ell).

It is distinct from the active-ion absorption and stimulated-emission terms
formed from ``activeIonDensity`` and ``crossSections``. The coefficient has
inverse-length units; ``5.5 / units.cm`` is therefore :math:`5.5\,\mathrm{cm}^{-1}`.

An omitted coefficient is ``None`` and contributes no volumetric attenuation.
An explicit ``0.0 / units.cm`` has the same transport effect and may be useful
when a measured material record should state the zero coefficient.
``isActive`` and ``isPassive`` report the explicit material classification;
``isTransparent`` identifies a passive condition with no bulk-loss
contribution.
``bulkAttenuation`` is the canonical material property, constructor keyword,
and HDF5 field. It describes the wavelength-independent bulk intensity loss;
it is not the wavelength-dependent active-ion absorption cross section stored
in ``crossSections.absorption``.

Units
-----

Material inputs are ``Quantity`` values. A number without a unit is rejected
where a physical dimension is required. Units can be converted without
changing the represented quantity:

.. code-block:: python

   lifetime = 0.941 * units.ms
   assert lifetime.toValue(units.s) == 0.000941

   density = 2.776e20 / units.cm**3
   attenuation = 5.5 / units.cm

The predefined catalogue includes metre, centimetre, millimetre, micrometre,
nanometre, second, millisecond, microsecond, kelvin, watt, and kilowatt.
Multiplication, division, and powers form derived units. Each ``Unit`` also
carries the ``unitSI`` and seven-component ``unitDimension`` values used at the
openPMD boundary.

Temperature states
------------------

``MaterialLibrary`` can store several measured or modelled states under one
key. ``resolve(..., interpolation="exact")`` requires a stored temperature.
Linear interpolation is explicit, emits ``TemperatureInterpolationWarning``,
and is limited to temperatures bracketed by stored states. Extrapolation is
not performed.

A record whose reference temperature is unknown must document that fact in
its metadata. Such a record cannot be mixed with numeric-temperature states or
resolved as though it represented a requested temperature.

Spectral resolution
-------------------

``spectralResolution`` resamples the selected absorption and emission curves
onto one endpoint-inclusive linear grid before transport:

.. code-block:: python

   material = loadBuiltinMaterials().resolve(
       "YbYAG",
       temperature=293.15 * units.K,
       activeIonDensity=2.776e20 / units.cm**3,
       spectralResolution=1000,
   )

The requested count cannot be smaller than the stored table. Resampling
metadata records the method and source count. See :doc:`spectral_decomposition`
for construction, interpolation, and legacy text import details.

Persist a material library
--------------------------

Libraries use a versioned HDF5 representation with units and JSON-compatible
metadata on the material record, temperature state, and cross-section table.
Format 1.1 stores the boolean ``active`` attribute on each material record;
temperature states under the same key cannot disagree about it. Passive
attenuation is stored only when it is specified:

.. code-block:: python

   from HASEonGPU import MaterialLibrary

   library = MaterialLibrary()
   library.register("YbYAG", roomTemperatureMaterial)
   library.registerState("YbYAG", elevatedTemperatureMaterial)
   library.toHdf5("materials.h5")

   restored = MaterialLibrary.fromHdf5("materials.h5")

``Material.toHdf5`` is a convenience for a one-record, one-state library.
Existing files are not overwritten unless ``overwrite=True`` is explicit.
Version 1.0 files predate the activity attribute. They remain readable through
a warning-emitting compatibility path that infers the classification from
stored lifetime and cross-section data. Rewriting such a library records the
classification explicitly as version 1.1.

From material to simulation
---------------------------

A resolved material becomes physical when assigned to an
``OpticalComponent``. The following repository example constructs the complete
topology, component, gain medium, and ``Simulation`` without starting the
backend:

.. literalinclude:: ../../../example/materialApiExample.py
   :language: python
   :pyobject: buildSimulation

The example partitions one shared topology into disjoint gain and cladding
volumes. Both are ``OpticalComponent`` objects, but only the active crystal is
placed in ``GainMedium``. It constructs the optional exterior surface from the
domain union and assigns a reflective coating to one cladding face. Surface
reflectivity and volumetric attenuation remain independent properties. The
current backend accepts several gain components when they reference the same
resolved ``Material`` object and one common attenuation value across passive
cells. Component and domain composition is described in :doc:`gain_medium`.
