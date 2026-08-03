Python Interface Guide
======================

The public frontend is imported from ``HASEonGPU``. It separates geometry,
material physics, numerical algorithms, and evolving state so that each object
has one responsibility. Modules below ``pyInclude`` are implementation details;
new applications should not import them.

The object model
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Concern
     - Public objects
     - Responsibility
   * - Topology
     - ``UnstructuredMesh``
     - Tet4 points, connectivity, adjacency, and named domain tags.
   * - Material physics
     - ``Material``, ``MaterialLibrary``, ``MaterialCondition``
     - Temperature-resolved optical data, HDF5 persistence, and one selected run condition.
   * - Placement and optics
     - ``MeshSelection``, exterior boundary and material-interface models
     - Attach physical models to mesh domains without putting them in the mesh.
   * - Algorithms
     - ``MonteCarloASESolver``, ``MonteCarloPumpSolver``, time integrators
     - Choose numerical methods and their run controls.
   * - Evolution
     - ``Simulation``, ``InitialState``, ``TimeStepState``
     - Assemble, validate, execute, and expose completed-step snapshots.

The lifecycle
-------------

A frontend run has distinct assembly and execution phases:

#. create or load a Tet4 ``UnstructuredMesh``;
#. load or define temperature-resolved materials and select a condition;
#. construct ``Simulation`` with solver descriptors and initial state;
#. register materials, exterior boundaries, internal interfaces, and pumps on
   ``mesh.volume(...)`` or ``mesh.surface(...)`` selections;
#. call ``compile()`` to validate layouts and inspect backend-neutral tables;
#. call ``validateBackend()`` to check that the current native adapter can
   execute those tables;
#. call ``step()`` or ``runUntil()`` and consume ``TimeStepState`` callbacks.

``compile()`` performs no native launch. It resolves typed mesh selections,
requires every cell and exterior face to be covered exactly once, and requires
an explicit interface wherever adjacent cells contain different material
conditions. ``validateBackend()`` is deliberately separate: a problem may be a
valid frontend model even when the current native adapter cannot execute all of
its features yet.

.. important::

   Frontend compilation supports multiple materials and explicit
   ``PerfectTransmission`` or ``FresnelInterface`` models. The current
   C++/openPMD 0.1 adapter executes only one isotropic active material, no
   internal material interface, and the built-in Monte Carlo ASE and pump
   solvers. Unsupported configurations fail before transport is launched.

Minimal complete setup
----------------------

All public physical values carry explicit units; see
:doc:`python_interface/physical_quantities`. The following snippets are
included from ``example/minimalExampleNewInterface.py`` so the guide and
runnable example share one source.

Create topology with domain identity but no material data:

.. literalinclude:: ../../example/minimalExampleNewInterface.py
   :language: python
   :start-after: # docs:start: mesh
   :end-before: # docs:end: mesh
   :dedent: 4

Define cross sections, temperature-resolved material physics, and a selected
simulation condition:

.. literalinclude:: ../../example/minimalExampleNewInterface.py
   :language: python
   :start-after: # docs:start: material
   :end-before: # docs:end: material
   :dedent: 4

Compose the algorithms and attach material and exterior optics to mesh domains:

.. literalinclude:: ../../example/minimalExampleNewInterface.py
   :language: python
   :start-after: # docs:start: simulation
   :end-before: # docs:end: simulation
   :dedent: 4

Finally, register physical pump light separately from its injection surface:

.. literalinclude:: ../../example/minimalExampleNewInterface.py
   :language: python
   :start-after: # docs:start: pump
   :end-before: # docs:end: pump
   :dedent: 4

Configuration is mutable until the first execution. ``compile()`` may be
called repeatedly while assembling the model. After initialization, material,
boundary, interface, pump, and initialization-callback registrations are
frozen; step callbacks may still be added.

Where to continue
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Python interface concepts

   python_interface/migration
   python_interface/topology
   python_interface/physical_quantities
   python_interface/material_library
   python_interface/gain_medium
   python_interface/spectral_decomposition
   python_interface/pump_properties
   python_interface/ase_solver
   python_interface/uncertainty
   python_interface/simulation
   python_interface/utilities

Use ``example/minimalExampleNewInterface.py`` for a self-contained Tet4 run and
``example/gmshMinimalExample.py`` for named Gmsh physical groups. The larger
``example/laserPumpCladding.py`` uses the same public composition API and the
bundled HDF5 material database for its time-stepped Tet4 calculation. Generated
signatures and members are listed in :doc:`pythonAPI`.
