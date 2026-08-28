Simulation YAML
===============

Schema version 3 mirrors the five-object frontend graph:

.. code-block:: python

   simulation = Simulation.fromYaml("simulation.yaml")

All paths are resolved relative to the containing YAML file. Unknown keys and
references are errors.

.. code-block:: yaml

   schema_version: 3

   materials:
     yb_yag:
       from_hdf5:
         path: materials.h5
         key: YbYAG
       temperature: 293.15
       active_ion_density: 2.776e26
       interpolation: exact
       spectral_resolution: 1000
     absorbing_cladding:
       from_hdf5:
         path: materials.h5
         key: CladdingGlass
       temperature: 293.15
       absorption_coefficient: 550.0

   topologies:
     assembly:
       from_file:
         path: assembly.msh
         format: gmsh

   domains:
     crystal_volume:
       from_gmsh:
         topology: assembly
         physical_group: crystal
         entity_kind: volume
     cladding_volume:
       from_gmsh:
         topology: assembly
         physical_group: cladding
         entity_kind: volume
     occupied_volume:
       union: [crystal_volume, cladding_volume]
     exposed_surface:
       boundary: occupied_volume
     pump_face:
       from_gmsh:
         topology: assembly
         physical_group: pump_input
         entity_kind: surface

   optical_components:
     crystal:
       domain: crystal_volume
       material: yb_yag
       surface_optics:
         - domain: pump_face
           reflectivity: 0.0
           exterior_refractive_index: 1.0
     cladding:
       domain: cladding_volume
       material: absorbing_cladding

   gain_media:
     amplifier:
       components: [crystal]

   simulation:
     optical_components: [crystal, cladding]
     gain_medium: amplifier
     exterior_surface: exposed_surface
     initial_excitation:
       value: 0.0
     phi_ase:
       ase_steps: 150
     time_integrator:
       method: frozen_phi_ase_runge_kutta4
     time_step_size: 2.0e-5
     simulation_steps: 150

For a step that evaluates pump and ASE transport once and freezes both source
rates across all four RK4 stages, use
``method: frozen_sources_runge_kutta4``. This differs from
``frozen_phi_ase_runge_kutta4``, which freezes the ASE flux but reevaluates the
pump rate at the intermediate RK4 states.

``yb_yag`` is only a local reference name. ``path`` and ``key`` choose the
database record. YAML bare dimensional values use SI: kelvin, ``m^-3``,
seconds, metres, ``m^2``, and ``m^-1``.

Material resolution
-------------------

Every YAML material is resolved from a versioned HDF5 library. The stored
record supplies its active/passive classification and available
temperature-dependent properties. ``temperature``, ``active_ion_density``,
``interpolation``, and ``spectral_resolution`` select the condition used by
this simulation. ``active_ion_density`` does not reclassify the material.

Run-specific overrides ``refractive_index``, ``fluorescence_lifetime``, and
``bulk_attenuation`` are applied after resolution and revalidated. They are
appropriate for measured corrections, but do not update the source library.
Use the material-library API when the corrected data should become a durable,
provenance-bearing material state. See :doc:`materials`.

``bulk_attenuation`` uses ``m^-1``. A passive material that omits it contributes
no volumetric attenuation. ``absorption_coefficient`` is its exact YAML alias
and the preferred physical spelling in configuration examples;
a material entry cannot provide both names. This coefficient describes
wavelength-independent bulk intensity loss, not the active-ion absorption
cross section.

Domain forms
------------

A named domain selects exactly one form: ``from_gmsh``, ``where``,
``topology``, ``component``, ``exterior_cells``, ``union``, ``difference``, or
``boundary``. ``exterior_tets`` remains a compatibility alias for
``exterior_cells``. Union accepts a non-empty operand list; difference accepts
exactly two operands. ``boundary`` calculates the surface of a volume-domain
union after removing faces between selected neighboring cells. Operands may be
named domains or inline domain definitions.

Every optical component names a volume ``domain``; topology shortcuts are not
part of the component schema. ``simulation.exterior_surface`` is optional and
references a surface domain. If omitted, the simulation uses the boundary of
the union of every optical-component domain.

Per-primitive construction and mixing
-------------------------------------

Each major primitive reads only its own named section:

.. code-block:: python

   material = Material.fromYaml("simulation.yaml", "yb_yag")
   domain = Domain.fromYaml("simulation.yaml", "crystal_volume")
   component = OpticalComponent.fromYaml(
       "simulation.yaml", "crystal", materials={"yb_yag": material}
   )
   gainMedium = GainMedium.fromYaml(
       "simulation.yaml", "amplifier", materials={"yb_yag": material}
   )

``Simulation.fromYaml`` accepts the same registries: ``materials``,
``topologies``, ``domains``, ``opticalComponents``, and ``gainMedia``. Injected
objects win by name; unresolved references are created from YAML. One shared
context caches constructed objects and detects reference cycles.

Material spectra are authoritative for both ASE and pump interactions. Schema
version 3 therefore has no top-level cross-section registry and pumps do not
name a second cross-section object.

See ``config/laserPumpCladding.yaml`` for all pump, relay, estimator,
integrator, output, and execution controls.
