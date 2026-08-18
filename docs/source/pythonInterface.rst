Python interface guide
======================

The Python frontend separates material data, geometry, physical composition,
and numerical execution. The compiled C++/Alpaka runtime receives a lowered
representation of that graph; backend fields are not the public construction
API.

Object model
------------

Five types define the physical graph:

``Material``
   A mutable optical state resolved at one temperature. Materials carry
   an explicit active/passive classification, refractive index, optional
   passive bulk attenuation, fluorescence lifetime, active-ion density,
   optical axis, metadata, and any absorption/emission spectra used by ASE and
   pump transport.

``OpticalComponent``
   One physical object: a volume ``Domain`` combined with one ``Material``.
   Components own their boundary-optics assignments.

``GainMedium``
   The active subset of the simulation's optical components. Passive cladding
   remains an ``OpticalComponent`` but is not placed in this container.

``Domain``
   An immutable typed region of volume cells or faces. Its frontend operations
   are independent of the concrete cell type and support union (``+``),
   difference (``-``), and boundary extraction.

``Simulation``
   The complete graph plus excitation state, pumps, ASE controls, time
   integration, and execution controls.

A material-library record is resolved before it enters this graph:

.. code-block:: text

   MaterialLibrary --resolve--> Material --------+
                                                  +--> OpticalComponent --+--> Simulation
   mesh implementation -----------> Domain ------+          |             |
                                                            +--> GainMedium-+

A compact programmatic graph looks like this:

.. code-block:: python

   from HASEonGPU import (
       Domain, ExplicitEuler, GainMedium, OpticalComponent, PhiASE, Simulation,
       VolumeTopology, units,
   )
   from material_library import loadBuiltinMaterials

   material = loadBuiltinMaterials().resolve(
       "YbYAG",
       temperature=293.15 * units.K,
       activeIonDensity=2.776e20 / units.cm**3,
   )
   topology = VolumeTopology.fromFile("crystal.msh", format="gmsh")
   crystalDomain = Domain.fromTopology(topology)
   component = OpticalComponent(domain=crystalDomain, material=material)
   gainMedium = GainMedium([component])

   simulation = Simulation(
       opticalComponents=[component],
       gainMedium=gainMedium,
       initialExcitation=0.0,
       phiASE=PhiASE(ase_steps=0),
       timeIntegrator=ExplicitEuler(),
       timeStepSize=1e-6,
       simulationSteps=1,
   )

The simulation obtains executable geometry from its component domains; it does
not own a separate topology list or aggregate optical domain. Gain and passive
regions must be disjoint. Domains on one shared topology retain adjacency,
while independent mesh bindings remain disconnected. The public domain model
can represent other cell structures, but the current backend lowers Tet4
``VolumeTopology`` bindings only. Gain components must reference the same
resolved ``Material`` object, and passive components must share one attenuation
coefficient. Material activity, rather than a non-zero density used as a flag,
determines which components may participate in excitation dynamics.

``Simulation.fromYaml`` constructs the same graph from schema version 3. Named
Python objects can be injected into that construction, allowing measured or
generated objects to be combined with durable YAML run configuration.

Concept pages
-------------

.. toctree::
   :maxdepth: 2
   :caption: Python modeling concepts

   python_interface/materials
   python_interface/topology
   python_interface/gain_medium
   python_interface/spectral_decomposition
   python_interface/pump_properties
   python_interface/phi_ase
   forwardAseRse
   python_interface/simulation
   python_interface/configuration
   python_interface/utilities
