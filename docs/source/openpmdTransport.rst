openPMD Transport
=================

HASEonGPU uses openPMD to exchange a self-describing primitive graph between
the Python frontend and ``calcPhiASE``. The graph retains the frontend's
``Simulation``, ``Domain``, ``OpticalComponent``, ``Material``, pump, and
solver structure instead of defining a second flattened input model.

This page describes runtime selection, session behavior, and the stored graph.
Alpaka compute selection is documented separately in
:doc:`backendSelection`. Contributors extending the graph should use
:doc:`developer/transportFields`.

Storage backends
----------------

The openPMD storage backend is independent from the Alpaka compute backend.
``auto`` selects the first backend supported by both the Python and C++
openPMD providers in this order: ``adios``, ``adios-sst``, then ``hdf5``.

``adios``
   File-backed ADIOS2 ``.bp`` series. This is preferred by ``auto`` when it is
   available.

``adios-sst``
   ADIOS2 SST streaming series. Synchronized Python/backend stepping requires
   this streaming backend.

``hdf5``
   HDF5 ``.h5`` series. The selected openPMD provider must include HDF5
   support.

Select the backend on ``PhiASE`` or in schema-v3 YAML:

.. code-block:: python

   phi_ase = PhiASE(
       backend="Host_Cpu_CpuSerial",
       openpmdBackend="auto",
   )

.. code-block:: yaml

   schema_version: 3
   simulation:
     phi_ase:
       backend: Host_Cpu_CpuSerial
       openpmd_backend: auto

The frontend checks an explicit selection, or resolves ``auto``, against both
the provider linked into ``calcPhiASE`` and the active Python ``openpmd_api``
module before launching the backend.

Transported object graph
------------------------

Each frontend primitive privately describes the state it owns through
``_transportDescription()``. The hook is internal serialization metadata; the
primitive's public constructor, properties, and methods remain responsible for
the user-facing API and validation.

A description contains:

``transportField(name, ...)``
   A value owned directly by the primitive. Examples include
   ``Simulation.controlFields`` and ``Material.refractiveIndex``.

``reference(name, ...)``
   A relationship to another self-describing primitive. For example,
   ``OpticalComponent`` references its ``Domain`` and ``Material``, while
   ``Material`` references its ``CrossSectionTable``.

The composer follows references and deduplicates objects by identity. Two
components that share one material therefore store one material node and one
cross-section-table node. The components contain references to those nodes;
they do not copy the material fields.

The generic writer consumes the resulting ``TransportGraph`` without
inspecting concrete frontend types. Adding a field to ``Material`` or
``Simulation`` does not add a corresponding branch to the writer.

Domain-local SoA topology
-------------------------

A ``Domain`` stores a selection mask for each ``VolumeTopology`` shard it
references. It does not depend on one simulation-global mesh object. This
keeps topology ownership suitable for later partitioning by domain, rank, or
device.

Topology arrays use structure-of-arrays order:

* points: ``coordinate x point``;
* cell connectivity: ``localVertex x cell``;
* cell faces: ``localVertex x localFace x cell``;
* neighbors and boundaries: ``localFace x cell``;
* face centers and normals: ``coordinate x localFace x cell``; and
* cell centers and sample points: ``coordinate x cell``.

Stored representation
---------------------

Transport version 1.2 distinguishes ``full`` and ``dynamic`` input iterations
and permits explicit, resizable cross-section updates at synchronized step
boundaries.
The initial iteration contains the complete primitive graph. Later control
iterations retain the graph identity and references, but write only fields
declared with ``dynamic=True``. A dynamic field with a ``controlField`` is
written only when that name is selected by ``Simulation.controlFields``;
therefore material cross sections are not transferred again unless
``cross_sections`` is explicitly enabled. Iteration attributes identify the
transport version, update mode, root path, node paths, node types, and
references between nodes.

Numeric scalar and array fields are stored as openPMD mesh records. Each record
carries its logical graph path, owning primitive type, field name, axes, shape,
dynamic flag, encoding, and physical-unit metadata. ``Quantity`` values retain
their ``unitSI`` and ``unitDimension`` information. Strings use native string
attributes; non-empty string sequences and references use native string-vector
attributes. Empty sequences use a scalar ``[]`` marker because ADIOS2 cannot
round-trip an empty string-vector attribute. The stored scalar/vector datatype
keeps that marker distinct from a one-element sequence containing ``"[]"``.
Provider APIs perform Unicode handling without a second JSON decoder. JSON
metadata remains serialized in namespaced string attributes.

Logical paths are encoded into provider-safe record and attribute keys. A
consumer uses the stored logical path rather than assigning meaning to the
physical key. Primitive definitions consequently do not depend on ADIOS2 or
HDF5 naming restrictions.

The C++ reader reconstructs the root ``Simulation`` for a full iteration and
delegates each referenced namespace to the corresponding C++ primitive's
``fromTransport(reader, prefix)`` function. A dynamic iteration updates the
existing simulation and prepared excitation plan instead of reconstructing the
topology or material context. Numeric loading is prefetched by subtree;
following one primitive does not require loading unrelated numeric fields.
The Python ``frontendState`` projection exists only for callback topology and
diagnostics; it is not transported as executable state. Consequently
the C++ ``simulation preparation`` is the single physical lowering boundary.

Direct ASE sessions
-------------------

``PhiASE.run(...)`` normally opens, writes, reads, and closes one transport
session. Direct execution accepts the physical ``GainMedium`` graph; spectra
come from the materials referenced by its components:

.. code-block:: python

   phi_ase.run(gainMedium=medium, initialExcitation=0.25)
   result = phi_ase.getResults()

For repeated SST requests, keep one stream open:

.. code-block:: python

   session = phi_ase.openStream()
   try:
       for excitation in states:
           phi_ase.run(
               gainMedium=medium,
               initialExcitation=excitation,
               openpmdSession=session,
           )
           result = phi_ase.getResults()
   finally:
       phi_ase.closeStream()

``openpmdSession="persistent"`` lets ``PhiASE`` own a reusable stream.
``openpmdSession="interval"`` forces one-shot behavior.

Compiled simulation sessions
----------------------------

``Simulation.step(...)`` and ``Simulation.runUntil(...)`` launch
``calcPhiASE --cpp-control``. A ``Simulation`` root references its physical
components, gain medium, exterior surface, excitation state, ``PhiASE``, time
integrator, and pump registrations. Pump registrations reference the physical
pump, injection method, and relays.

In ``autonomous`` mode, Python sends the initial graph and the C++ backend owns
the complete time loop. Python receives the completed steps selected by
``outputSteps`` and ``outputFields``. With an SST backend, the result receiver
starts before the input writer and a bounded handoff applies backpressure when
a callback is slower than the backend. Caller-managed simulation sessions are
not supported.

``synchronized-debug`` mode requires ``adios-sst``. After output step *N*,
Python runs the registered ``onStep`` and ``beforeStep`` callbacks, refreshes
the frontend ``ExcitationState``, and writes dynamic iteration *N*. The backend
waits for that iteration before starting step *N+1*. Dynamic iterations contain
``Simulation.currentStep``, ``Simulation.currentTime``, and
``ExcitationState.values``; topology and other static graph fields remain in
the backend context created from iteration zero.

``controlFields`` declares which supported state Python may update between
steps. It is transport data owned by ``Simulation``; the generic writer stores
the string sequence without interpreting its simulation meaning.

Results and snapshots
---------------------

One-shot ASE output uses the ``phiAseResult`` namespace. It contains
cell-centered flux, standard error, relative standard error, total ray count,
ASE depletion, and reflection-termination metadata.

Compiled simulation output uses one ``simulationSnapshot`` iteration for each
selected completed step. A snapshot contains only the evolving fields selected
by ``outputFields`` and their result metadata. Output fields are cell-centered;
the compiled transport does not expose a separate point-centered excitation
state.

MPI uses the same primitive graph and result namespaces. MPI changes the
execution topology, not the transport model. Rank/device layout, launch
behavior, and shared-storage requirements are documented in :doc:`mpi`.

Provider compatibility
----------------------

The Python ``openpmd_api`` module and the C++ ``openPMD::openPMD`` provider
must be compatible and must both support the selected storage backend. The
guided setup checks the normal configuration:

.. code-block:: bash

   python3 utils/configure_hase.py

For a manual provider check:

.. code-block:: bash

   python3 utils/check_openpmd_compatibility.py \
     --backend adios-sst \
     --cmake-prefix-path /path/to/openpmd/prefix

The wheel does not vendor openPMD runtime libraries or generated
``openpmd_api`` bindings. If the matching Python package is outside the normal
Python path, use ``HASE_OPENPMD_PYTHON_PACKAGE_DIR`` at build time or
``HASE_OPENPMD_PYTHONPATH`` at runtime. Provider build options are listed in
:ref:`openpmd-provider-options`.

Artifact retention
------------------

Temporary transport artifacts are normally removed when a session exits.
These environment variables support debugging:

``HASE_OPENPMD_KEEP_ARTIFACTS=1``
   Keep artifacts below ``./hase-openpmd-artifacts``.

``HASE_OPENPMD_ARTIFACT_DIR=/path``
   Write artifacts to an explicit directory.

``HASE_OPENPMD_ARTIFACT_PREFIX=name``
   Prefix generated artifact names.

``HASE_OPENPMD_ARTIFACT_RUN_ID=id``
   Use a stable run identifier instead of a timestamped identifier.

``HASE_OPENPMD_WATCHDOG_INTERVAL=30``
   Set the interval while the streamed-result watchdog waits. ``0`` or
   ``none`` disables it.

``HASE_OPENPMD_THREAD_JOIN_TIMEOUT=10``
   Set the time allowed for streaming helper threads to stop.

``HASE_CALCPHIASE=/path/to/calcPhiASE``
   Force the Python transport to use a specific executable.
