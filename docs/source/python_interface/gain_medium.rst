Materials, layouts, and interfaces
==================================

Material definition and instance
--------------------------------

``MaterialDefinition`` is reusable, concentration-independent physics:

.. code-block:: python

   spectra = CrossSectionTable(wavelengths, absorption, emission)
   yag = MaterialDefinition(
       name="Yb:YAG",
       refractive_index=1.82,
       fluorescence_lifetime=941e-6,
       cross_sections=spectra,
   )

``MaterialInstance`` supplies simulation-specific values such as active-ion
number density and optional optical-axis orientation:

.. code-block:: python

   gain = MaterialInstance(yag, active_ion_density=2.76e26, name="gain")
   undoped = MaterialInstance(yag, active_ion_density=0.0, name="cap")

Material objects are independent of meshes. All units are SI: density is
``m^-3``, lifetime is seconds, wavelength is metres, cross section is ``m^2``,
and bulk attenuation is ``m^-1``.

Multiple material layouts
-------------------------

Register any number of material instances. Layouts must cover every cell
exactly once.

.. code-block:: python

   simulation.add_material(gain, MaterialLayout("core"))
   simulation.add_material(undoped, MaterialLayout(("left_cap", "right_cap")))

Reusing the same ``MaterialInstance`` in multiple layouts reuses one entry in
the compiled material table. Distinct instances remain distinct even if their
values happen to match.

Internal interfaces
-------------------

Every adjacent pair of unlike material instances needs an explicit interface:

.. code-block:: python

   simulation.add_interface(
       MaterialInterface(PerfectTransmission(), name="core_to_cap"),
       MaterialInterfaceLayout(between=(gain, undoped)),
   )

``FresnelInterface`` is also available so models can be expressed without a
future API change. The compiled frontend retains a per-face interface table.
The native backend does **not** yet implement perfect transmission, Fresnel
reflection/refraction, or transport across unlike material domains.

Exterior boundaries
-------------------

Exterior faces are configured separately and must be covered exactly once:

.. code-block:: python

   simulation.add_boundary(
       ExteriorBoundary(AbsorbingSurface()), BoundaryLayout("all_exterior")
   )

``ConstantReflectivitySurface(reflectivity=..., exterior_refractive_index=...)``
is supported by the current single-material adapter.
