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
     - ``CrossSectionTable``, ``MaterialDefinition``, ``MaterialInstance``
     - Reusable optical data and run-specific active-ion density.
   * - Placement and optics
     - ``MaterialLayout``, ``BoundaryLayout``, ``MaterialInterfaceLayout``
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
#. create material definitions and instances;
#. construct ``Simulation`` with solver descriptors and initial state;
#. register materials, exterior boundaries, internal interfaces, and pumps;
#. call ``compile()`` to validate layouts and inspect backend-neutral tables;
#. call ``validate_backend()`` to check that the current native adapter can
   execute those tables;
#. call ``step()`` or ``run_until()`` and consume ``TimeStepState`` callbacks.

``compile()`` performs no native launch. It resolves named or numeric domains,
requires every cell and exterior face to be covered exactly once, and requires
an explicit interface wherever adjacent cells contain different material
instances. ``validate_backend()`` is deliberately separate: a problem may be a
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

All public physical values use SI units. The following snippets are included
from ``example/minimalExampleNewInterface.py`` so the guide and runnable
example share one source.

Create topology with domain identity but no material data:

.. literalinclude:: ../../example/minimalExampleNewInterface.py
   :language: python
   :start-after: # docs:start: mesh
   :end-before: # docs:end: mesh
   :dedent: 4

Define cross sections, reusable material physics, and a run-specific material
instance:

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
   python_interface/gain_medium
   python_interface/spectral_decomposition
   python_interface/pump_properties
   python_interface/phi_ase
   python_interface/simulation
   python_interface/utilities

Use ``example/minimalExampleNewInterface.py`` for a self-contained Tet4 run and
``example/gmshMinimalExample.py`` for named Gmsh physical groups. The
``example/laserPumpCladding.py`` driver intentionally remains a private legacy
compatibility regression; it is not a template for new user code. Generated
signatures and members are listed in :doc:`pythonAPI`.
