Binary Interface
================

This page documents the direct command-line usage of the generated
``calcPhiASE`` binary.

For general setup and dependency information, see
:doc:`Getting Started <gettingStarted>`.

Compilation
-----------

In order to use haseongpu from the command line, manual compilation is required.

For compilation details see :doc:`compilation`.

After compilation, the executable is available as:

.. code-block:: text

   ./build/calcPhiASE

Usage
-----

The binary consumes an openPMD input series and writes an openPMD result series. The input schema is the same contract used by the Python ``PhiASE`` transport; see :doc:`openpmdTransport` for record names, storage backends, and iteration update rules.

.. code-block:: bash

   ./build/calcPhiASE \
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

   ./build/calcPhiASE \
       --input-path=./input.bp \
       --output-path=./output.bp

MPI run:

.. code-block:: bash

   mpiexec -npernode 4 ./build/calcPhiASE \
       --input-path=./input.bp \
       --output-path=./output.bp

Command-Line Arguments
----------------------

``--input-path``
^^^^^^^^^^^^^^^^

Path to the input openPMD series.

This must point to an openPMD series following the HASE openPMD transport
contract. Python-written series use canonical wedge topology records on static
iterations and only ``core_beta_volume`` plus ``core_point_beta`` on
dynamic-only iterations.

``--output-path``
^^^^^^^^^^^^^^^^^

Path where ``calcPhiASE`` writes the result openPMD series. Results are written as ``core_result_phi_ase``, ``core_result_mse``, ``core_result_total_rays``, and ``core_result_dndt_ase`` mesh records.
