openPMD Transport
=================

HASEonGPU uses an openPMD series as the transport boundary between the Python
frontend and the C++ ``calcPhiASE`` backend.  The high-level Python objects
remain the user-facing model; the transport serializes those objects into the
schema documented in ``docs/EXT_HASE.md``.

The transport is used by ``PhiASE.run(...)`` and by the standalone
``calcPhiASE`` binary.  Advanced users can also import
``pyInclude.openpmd.transport`` to write input series, run the backend, and read
result series explicitly.

Storage Backends
----------------

The openPMD storage backend controls the on-disk or streaming format. It is
separate from the alpaka compute backend selected by the ``PhiASE.backend``
setting.

Accepted transport backend names are:

``adios``
   ``.bp`` series with an explicit ADIOS2 openPMD configuration.

``hdf5``
   ``.h5`` series. Requires an openPMD-api build with HDF5 support.

``adios-sst``
   ``.sst`` ADIOS2 SST streaming series. Use this backend only when a producer
   and consumer are intended to run concurrently.

Select the default from Python with ``HASE_OPENPMD_BACKEND``:

.. code-block:: bash

   HASE_OPENPMD_BACKEND=adios python3 my_simulation.py

For direct helper calls, pass ``transport=``:

.. code-block:: python

   import pyInclude.openpmd.transport as transport

   result = transport.runPhiASE(
       phi_ase,
       medium,
       spectra,
       transport="adios",
   )

The CMake build also has ``HASE_OPENPMD_BACKEND``. That option selects which
openPMD dependencies and default file extension are built into the C++ test and
binary configuration. Runtime Python selection still requires the matching
openPMD-api Python module and backend support to be available.

Schema Summary
--------------

Current Python-written input uses canonical wedge topology records:

* ``core_points`` with ``x``, ``y``, and ``z`` components.
* ``core_cells_connectivity`` with six node indices per wedge prism.
* ``core_cells_offsets`` with contiguous six-node offsets.
* ``core_cells_types`` containing VTK wedge type ``13``.

Dynamic fields are written as flattened mesh records:

* ``core_beta_volume`` for prism-centered excited-state fraction.
* ``core_point_beta`` for point-level excited-state fraction.

Static material and spectral records include ``core_cladding_cell_type``,
``core_refractive_index``, ``core_reflectivity``,
``core_lambda_absorption``, ``core_lambda_emission``,
``core_sigma_absorption``, and ``core_sigma_emission``.

The C++ backend writes result records under ``core_result_``:
``phi_ase``, ``mse``, ``total_rays``, and ``dndt_ase``.

For the complete record list, units, shapes, metadata attributes, and
compatibility notes, see ``docs/EXT_HASE.md`` in the repository.

Iteration Updates
-----------------

An openPMD input series may contain multiple iterations.

The first Python-written iteration includes ``haseStaticUpdate = true`` and
contains topology, static material records, spectra, and dynamic fields.
Later iterations default to ``haseStaticUpdate = false`` and contain only
``core_beta_volume`` and ``core_point_beta``. The C++ parser caches the latest
static topology and applies each dynamic-only iteration to that cached context.

This split is important for streaming or repeated ASE evaluations because the
large static topology does not need to be rewritten for every dynamic state.

Unsupported Options
-------------------

The openPMD transport does not preserve explicit device lists and does not
write VTK output from the backend. ``PhiASE.devices`` and ``PhiASE.writeVtk``
are therefore rejected by the Python transport when they request unsupported
behavior. The C++ parser also rejects ``write_vtk = true`` and explicit
``devices`` metadata.

Use the Python-side VTK helpers documented in :doc:`python_interface/utilities`
for visualization output.

MPI Launching
-------------

The standalone ``calcPhiASE`` binary can be launched under MPI and reads the
same openPMD input contract:

.. code-block:: bash

   mpiexec -npernode 4 ./build/calcPhiASE \
       --input-path=input.bp \
       --output-path=output.bp

For Python-controlled runs that need an MPI launcher, pass a command prefix to
the transport helper:

.. code-block:: python

   result = transport.runPhiASE(
       phi_ase,
       medium,
       spectra,
       command_prefix=["mpiexec", "-npernode", "4"],
       workspace_dir="IO/phiase_mpi",
   )

The ``parallel_mode`` metadata is still written into the input series and
consumed by the backend compute configuration. The process topology itself is
controlled by how ``calcPhiASE`` is launched.

Artifact Retention
------------------

By default, temporary input and output series are removed when the Python
transport session exits. Use these environment variables when debugging the
transport files:

``HASE_OPENPMD_KEEP_ARTIFACTS=1``
   Keep artifacts below ``./hase-openpmd-artifacts``.

``HASE_OPENPMD_ARTIFACT_DIR=/path/to/dir``
   Write artifacts to an explicit directory.

``HASE_OPENPMD_ARTIFACT_PREFIX=name``
   Prefix generated artifact names.

``HASE_OPENPMD_ARTIFACT_RUN_ID=name``
   Use a stable run id instead of a timestamped id.

``HASE_CALCPHIASE=/path/to/calcPhiASE``
   Force the Python transport to use a specific openPMD ``calcPhiASE`` binary.
