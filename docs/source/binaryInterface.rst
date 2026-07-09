Binary Interface
================

This page documents the direct command-line usage of the generated
``hase-cpp`` binary.

For general setup and dependency information, see
:doc:`Getting Started <gettingStarted>`.

Compilation
-----------

In order to use haseongpu from the command line, manual compilation is required.

For compilation details see :doc:`compilation`.

After compilation, the executable is available as:

.. code-block:: text

   ./build/hase-cpp

Usage
-----

The binary consumes an openPMD input series and writes an openPMD result series. The input layout is the same one written by the Python ``PhiASE`` transport; see :doc:`openpmdTransport` for record names, storage backends, and iteration update rules.

.. code-block:: bash

   ./build/hase-cpp \
       --input-path=<openPMD-input-series> \
       --output-path=<openPMD-output-series>

Simulation parameters, mesh topology, dynamic gain fields, backend selection,
parallel mode, sample range, and optional RNG seed are read from the openPMD
transport metadata. The command line accepts only the openPMD transport
entrypoint options for those values. Unsupported metadata such as ``write_vtk = true`` or
explicit ``devices`` is rejected by the parser.

Example
-------

Single-process run:

.. code-block:: bash

   ./build/hase-cpp \
       --input-path=./input.bp \
       --output-path=./output.bp

MPI run:

.. code-block:: bash

   mpiexec -npernode 4 ./build/hase-cpp \
       --input-path=./input.bp \
       --output-path=./output.bp

Command-Line Arguments
----------------------

``--input-path``
^^^^^^^^^^^^^^^^

Path to the input openPMD series.

This must point to an openPMD series following the HASE openPMD transport
layout. Python-written series store arrays as openPMD ``Mesh`` records, but
the unstructured extruded-prism topology encoded by ``core_points``,
``core_cells_connectivity``, ``core_cells_offsets``, and ``core_cells_types`` is
HASE-defined. Static iterations include that topology; dynamic-only iterations
contain only ``core_beta_volume`` and reuse the cached static topology.

``--output-path``
^^^^^^^^^^^^^^^^^

Path where ``hase-cpp`` writes the result openPMD series. Results are written as ``core_result_phi_ase``, ``core_result_mse``, ``core_result_total_rays``, and ``core_result_dndt_ase`` mesh records.

Compiled Simulation Mode
------------------------

``hase-cpp`` can also own the full C++/Alpaka time loop when the input
iteration contains simulation run-control attributes:

.. code-block:: bash

   ./build/hase-cpp \
       --input-path=./simulation-input.bp \
       --output-path=./simulation-output.bp \
       --cpp-control

In this mode Python sends the initial mesh/material/spectra/beta state and the
binary writes one output iteration per completed time step. The output snapshot
iterations include ``core_point_beta``, ``core_beta_volume``,
``core_result_phi_ase``, ``core_result_mse``, ``core_result_total_rays``,
``core_result_dndt_ase``, and ``core_result_dndt_pump``. Iteration 0 also
carries the static mesh/material/spectral records so the snapshot series can be
read independently.

Run-control attributes include ``time_step``, ``number_of_steps``,
``pump_steps``, ``time_integrator``, optional implicit-Euler controls
``implicit_iterations`` and ``implicit_tolerance``, and pump attributes such as
``pump_routine``, ``pump_intensity``, ``pump_wavelength``, ``pump_radius_x``,
``pump_radius_y``, ``pump_duration``, and ``pump_substeps``. The supported pump
routine is ``one-dimensional-z-traversal``.
