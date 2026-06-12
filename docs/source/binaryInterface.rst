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

The binary consumes an openPMD input series and writes an openPMD result series:

.. code-block:: bash

   ./build/calcPhiASE \
       --input-path=<openPMD-input-series> \
       --output-path=<openPMD-output-series>

Simulation parameters, mesh topology, dynamic gain fields, backend selection,
parallel mode, sample range, and optional RNG seed are read from the openPMD
transport metadata. The command line no longer accepts the legacy parser
options for those values.

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

This must point to an openPMD series written by the HASE openPMD transport.
The series contains the static topology, dynamic fields, and compute metadata
required by the backend parser.

``--output-path``
^^^^^^^^^^^^^^^^^

Path where ``calcPhiASE`` writes the result openPMD series.
