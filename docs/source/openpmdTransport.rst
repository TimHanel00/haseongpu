openPMD Transport
=================

HASEonGPU uses openPMD as the transport boundary between the Python frontend and
the C++ ``calcPhiASE`` backend. Users work with ``UnstructuredMesh``,
``MaterialCondition``, typed selections, solver descriptors, and ``Simulation``.
The transport encoder converts the supported single-material problem into the
established openPMD 0.2 records and attributes consumed by the backend.
Multiple-material and internal-interface tables are not serialized yet.

This transport is separate from the HDF5 :doc:`material database
<python_interface/material_library>`. The latter is read with ``h5py`` and is
not an openPMD series.

Storage Backends
----------------

The openPMD storage backend is independent from the Alpaka compute backend.
``auto`` is the default: it chooses the first backend supported by both the
compiled and Python openPMD providers in this order: ``adios``, ``adios-sst``,
then ``hdf5``. Explicit runtime values are:

``adios-sst``
   ADIOS2 SST streaming series. Select it explicitly when a live stream is
   preferable to file-backed exchange.

``adios``
   ADIOS2 file-backed ``.bp`` series. This is the automatic default when
   supported because it is currently more robust and usually faster than SST
   for HASEonGPU's frontend/backend exchange.

``hdf5``
   HDF5 ``.h5`` series.  Requires HDF5 support in the selected openPMD-api
   provider.

Select it in Python or YAML:

.. code-block:: python

   aseSolver = MonteCarloASESolver(..., openPmdBackend="auto")

.. code-block:: yaml

   compute:
     backend: Host_Cpu_CpuSerial       # Alpaka compute backend
     openpmd_backend: auto              # choose a compatible backend

Set the ``MonteCarloASESolver(openPmdBackend=...)`` argument or YAML
``openpmd_backend`` to override automatic selection for a particular run.

Simulation transport lifetime
-----------------------------

``Simulation.step`` owns the complete openPMD lifetime. Python writes one
initial input iteration and the compiled C++ backend owns the complete time
loop, persistent device context, queues, meshes, spectra, and evolving cell
state. Caller-managed sessions and Python-controlled normal stepping are not
part of the public API.

Autonomous output is selected at initialization with ``outputSteps`` and
``outputFields``. With SST, an internal receiver drains those snapshots while
the backend continues; slow callbacks can eventually apply bounded transport
backpressure but never participate in numerical stepping. File-backed runs
read the selected snapshots after the backend exits. A final-only run is just
``outputSteps=autonomousFinal(numberOfSteps)``.

``synchronized-debug`` is the separate bidirectional contract. Every completed
step is streamed to Python, callbacks may return a configured ``beta_volume``
control field, and the backend waits for the matching control iteration before
continuing. It intentionally requires SST and trades throughput for observable,
deterministic step boundaries.

Provider Compatibility
----------------------

The Python ``openpmd_api`` module and the C++ ``openPMD::openPMD`` provider
must be compatible and must both support the selected runtime backend.  The
guided setup checks this for common installs:

.. code-block:: bash

   python3 utils/configure_hase.py

For manual checks against an existing provider:

.. code-block:: bash

   python3 utils/check_openpmd_compatibility.py \
     --backend adios-sst \
     --cmake-prefix-path /path/to/openpmd/prefix

Then point installation or CMake configuration at the same provider, for
example with ``CMAKE_PREFIX_PATH`` or ``openPMD_DIR``.  If the matching Python
package is not on the normal Python path, set ``HASE_OPENPMD_PYTHON_PACKAGE_DIR``
at build time or ``HASE_OPENPMD_PYTHONPATH`` before importing HASEonGPU.

The HASEonGPU wheel does not vendor openPMD runtime libraries or generated
``openpmd_api`` bindings.  The runtime environment must provide compatible
openPMD libraries and Python bindings.

openPMD Record Layout
---------------------

The public frontend objects (``UnstructuredMesh``, ``Material``,
``MaterialCondition``, typed selections, and solver descriptors) are not an
openPMD schema. The transport encoder emits the openPMD series below. This
wire contract remains single-material even though frontend compilation supports
multiple materials and per-face interface tables.

Input ``Quantity`` objects are converted to the wire schema's canonical units:
wavelengths to ``nm``, cross sections to ``cm^2``, active-ion density to
``cm^-3``, fluorescence lifetime to ``s``, and geometry to the declared mesh
unit. Mesh records carry ``unitSI`` and ``unitDimension`` metadata, which the
native parser validates before using the stored magnitudes. Scalar extension
attributes are first converted by their unit-aware schema specifications; their
stored units are fixed by the HASE 0.2 wire contract rather than repeated as
openPMD record metadata.

All array data at that boundary is written as openPMD ``Mesh`` records below
each ``Iteration``'s ``meshes`` group. Scalar arrays are named openPMD records
with the scalar ``SCALAR`` record component. Component records, currently
``core_points``, use named components such as ``x``, ``y``, and ``z``. The
record names are HASE-owned, which is allowed by openPMD, but the records carry
the normal openPMD mesh and component metadata: ``geometry``,
``geometryParameters``, ``dataOrder``, ``axisLabels``, ``gridSpacing``,
``gridGlobalOffset``, ``gridUnitSI``, ``unitDimension``, component ``unitSI``,
and component ``position``.

Scalar simulation and backend settings are not openPMD field records. Values
such as ``numberOfPoints``, ``thickness``, ``rngSeed``, ``backend``, and
``parallelMode`` are stored as attributes on the openPMD iteration.
These values configure the HASE backend and do
not represent sampled mesh data. They therefore are not part of ``/meshes``
and do not carry record metadata such as ``axisLabels`` or component
``position``.

Forward-reflection request attributes follow the same HASE openPMD extension
schema: ``useReflections``, ``reflectionMaxIterations``,
``reflectionTolerance``, and ``surfaceReservoirSize``. The parser rejects
the retired ``forward_ray_length`` attribute; forward rays now traverse to a
physical boundary. Runtime environment overrides ``HASE_SRM_MAX_ITERATIONS``
and ``HASE_SRM_DIVERGENCE_STREAK`` are deliberately not serialized because
they are local execution policy rather than portable request data.

The topology convention inside ``/meshes`` follows VTK's unstructured-grid
model and tetrahedral cell. openPMD provides the mesh-record model, but it
does not standardize VTK-style Tet4 connectivity itself. HASEonGPU therefore
stores a VTK-compatible unstructured-cell layout in openPMD records:

* ``core_points`` stores VTK ``POINTS`` as ``x``, ``y``, and ``z`` components.
* ``core_cells_connectivity`` stores the VTK cell connectivity point ids.
* ``core_cells_offsets`` stores offsets into the connectivity array.
* ``core_cells_types`` stores the VTK cell type id; Tet4 cells use type ``10``.

Main input field records are:

* ``core_beta_volume`` for dynamic excited-state data
* ``core_cladding_cell_type``, ``core_refractive_index``, and
  ``core_reflectivity`` for static material/surface data
* ``core_lambda_absorption``, ``core_lambda_emission``,
  ``core_sigma_absorption``, and ``core_sigma_emission`` for spectra

The C++ backend writes result records under ``core_result_``:
``phiAse``, ``standard_error``, ``relative_standard_error``, ``total_rays``,
and ``dndt_ase``. ``standard_error`` has the same flux unit as ``phiAse``;
``relative_standard_error`` is dimensionless. Result records use record-C
layout.

Result iterations also carry registered HASE extension attributes for SRM
termination: ``srm_status``, ``srm_passes``, ``srm_remaining_fraction``,
``srm_max_iterations``, and ``srm_divergence_streak``. They are scalar
iteration metadata, not mesh records. Python readers expose them as
``Result.srmStatus``, ``srmPasses``, ``srmRemainingFraction``,
``srmMaxIterations``, and ``srmDivergenceStreak``.

Compiled Simulation Run Control
--------------------------------

For compiled ``Simulation`` runs, iteration attributes also include run-control
metadata:

* ``time_step`` and ``number_of_steps``
* ``pump_steps``
* ``enable_ase`` and ``pre_pump``
* ``execution_mode``, ``output_steps``, ``output_fields``, and
  ``control_fields``
* ``time_integrator`` (``explicit-euler``, ``heun``, ``midpoint``,
  ``runge-kutta-4``, ``frozen-phi-ase-runge-kutta-4``,
  ``implicit-euler``, or ``exponential-euler``)
* ``implicit_iterations`` and ``implicit_tolerance`` for implicit Euler
* ``pump_schema_version`` (currently ``1``), ``pump_ray_count``, and ``pump_rng_seed``
* flattened source, spectrum, angular, profile, and planar-relay arrays

The C++ backend writes only the selected completed steps. Snapshot iterations
contain selected cell records from ``core_beta_volume`` and ``core_result_phi_ase``,
``core_result_standard_error``, ``core_result_relative_standard_error``,
``core_result_total_rays``, ``core_result_dndt_ase``, and
``core_result_dndt_pump``. Static canonical mesh/material/spectral records stay
in the initialization input and are not duplicated in output snapshots.


Iteration Updates
-----------------

The first and normally only Python-written iteration contains the full static context: topology,
material records, spectra, compute attributes, and the dynamic
``core_beta_volume`` field. Only synchronized-debug adds later input iterations;
those are dynamic-only controls and currently may contain ``core_beta_volume``.

Changing topology, spectra, material constants, or compute settings requires a
new input series whose first iteration carries a complete static update.  This
keeps repeated ASE evaluations and streaming runs small while preserving a
stable backend contract.

MPI Launching
-------------

The standalone binary reads the same transport layout under MPI:

.. code-block:: bash

   mpiexec -npernode 4 ./build/calcPhiASE \
       --input-path=input.sst \
       --output-path=output.sst

The high-level Python frontend launches the binary automatically when MPI mode
is selected:

.. code-block:: python

   import HASEonGPU

   aseSolver = HASEonGPU.MonteCarloASESolver(
       parallelMode="mpi",
       ranksPerNode=4,
       openPmdBackend="adios-sst",
   )
   simulation = HASEonGPU.Simulation(..., aseSolver=aseSolver)
   simulation.step()

The scheduler controls the node allocation, while ``ranksPerNode`` controls the
number of ranks launched on each allocated node. File-based transport data is
created below ``./IO/phiase_mpi`` so the launch directory must be shared for a
multi-node run.

Artifact Retention
------------------

Temporary transport artifacts are normally removed when a session exits.  These
environment variables help with debugging:

``HASE_OPENPMD_KEEP_ARTIFACTS=1``
   Keep artifacts below ``./hase-openpmd-artifacts``.

``HASE_OPENPMD_ARTIFACT_DIR=/path``
   Write artifacts to an explicit directory.

``HASE_OPENPMD_ARTIFACT_PREFIX=name``
   Prefix generated artifact names.

``HASE_OPENPMD_ARTIFACT_RUN_ID=id``
   Use a stable run id instead of a timestamped id.

``HASE_OPENPMD_WATCHDOG_INTERVAL=30``
   Watchdog interval while the result receiver waits.  Use ``0`` or ``none`` to
   disable the watchdog.

``HASE_OPENPMD_THREAD_JOIN_TIMEOUT=10``
   Time allowed for streaming helper threads to stop during session close.

``HASE_CALCPHIASE=/path/to/calcPhiASE``
   Force the Python transport to use a specific binary.
