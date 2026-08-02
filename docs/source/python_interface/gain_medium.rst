Materials, layouts, boundaries, and interfaces
===============================================

Material definition and instance
--------------------------------

``MaterialDefinition`` stores reusable, concentration-independent physics:

.. code-block:: python

   spectra = CrossSectionTable(wavelengths, absorption, emission)
   yag = MaterialDefinition(
       name="Yb:YAG",
       refractive_index=1.82,
       fluorescence_lifetime=941e-6,
       cross_sections=spectra,
       bulk_attenuation=0.0,
   )

``MaterialInstance`` supplies values specific to one simulated part:

.. code-block:: python

   gain = MaterialInstance(yag, active_ion_density=2.76e26, name="gain")
   undoped = MaterialInstance(yag, active_ion_density=0.0, name="cap")

An active instance has a positive ``active_ion_density`` and requires a
fluorescence lifetime plus non-zero emission data. A passive instance has zero
active-ion density and positive bulk attenuation; a transparent instance has
neither. ``optical_axis`` is normalized by the frontend, but the current native
adapter does not yet support oriented materials.

All values use SI: density is ``m^-3``, lifetime is seconds, wavelength is
metres, cross section is ``m^2``, and attenuation is ``m^-1``.

Material placement
------------------

Register material instances on named or numeric volume domains:

.. code-block:: python

   simulation.add_material(gain, MaterialLayout("core"))
   simulation.add_material(undoped, MaterialLayout(("left_cap", "right_cap")))

Every cell must be selected exactly once. Reusing the same
``MaterialInstance`` in several layouts reuses one compiled material ID;
different instances remain different even when their values match.

Exterior boundaries
-------------------

Every exterior face must also be covered exactly once:

.. code-block:: python

   simulation.add_boundary(
       ExteriorBoundary(AbsorbingSurface(), name="housing"),
       BoundaryLayout("all_exterior"),
   )

``AbsorbingSurface`` and
``ConstantReflectivitySurface(reflectivity=...,
exterior_refractive_index=...)`` are supported by the current backend. A
boundary layout selects mesh surface domains; it is independent of pump
injection, even when both select the same domain.

Internal material interfaces
----------------------------

Every neighboring pair of unlike material instances needs one explicit model:

.. code-block:: python

   simulation.add_interface(
       MaterialInterface(PerfectTransmission(), name="core_to_cap"),
       MaterialInterfaceLayout(between=(gain, undoped)),
   )

``PerfectTransmission`` and ``FresnelInterface`` compile to per-face interface
tables. They express the intended model, but the current native adapter rejects
both before launch because cross-material transport is not implemented yet.

Initial excitation
------------------

``InitialState`` accepts one scalar, one value per Tet4 cell, or a domain map:

.. code-block:: python

   InitialState(0.0)
   InitialState(np.zeros(mesh.number_of_cells))
   InitialState({"core": 0.1, "left_cap": 0.0, "right_cap": 0.0})

Excitation fractions must be finite and within ``[0, 1]``. Domain-mapped
initial state must cover each cell exactly once.
