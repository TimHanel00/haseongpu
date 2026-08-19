Binary Interface
================

``calcPhiASE`` is the standalone C++ executable behind the Python frontend.
Most users call it through ``PhiASE`` or ``Simulation``; use it directly when a
workflow already produces the HASE openPMD input series.

Build
-----

Manual CMake compilation produces ``./build/calcPhiASE``. See
:doc:`compilation` for build and provider options. A thin Python installation
also records the matching executable and normally launches it automatically.

One ASE evaluation
------------------

The default mode reads one input series and writes one result series:

.. code-block:: bash

   ./build/calcPhiASE \
       --input-path=./input.bp \
       --output-path=./output.bp

The input iteration contains a complete ``Simulation`` primitive graph. For a
direct ASE request, Python constructs that graph from ``PhiASE``, the physical
``GainMedium``, its components, materials, cross-section tables, domains, and
domain-local topology shards. The output ``phiAseResult`` namespace contains
cell-centered flux, uncertainty, history counts, depletion rate, and
reflection termination metadata. See :doc:`openpmdTransport` for the graph and
on-disk projection.

Compiled simulation mode
------------------------

``--cpp-control`` interprets the input iteration as a complete time-stepped run
request:

.. code-block:: bash

   ./build/calcPhiASE \
       --input-path=./simulation-input.bp \
       --output-path=./simulation-output.bp \
       --cpp-control

The ``Simulation`` root owns time-step and execution controls and references
the integrator, ``PhiASE``, excitation state, optical components, gain medium,
exterior surface, and pump registrations. Pumps in turn reference their
spectrum, profile, angular distribution, injection method, and relays. The
binary does not read a second flattened run-control schema.

C++/Alpaka owns the loop and writes one ``simulationSnapshot`` iteration for
each selected completed step. Without an output schedule, every completed step
is selected. Each snapshot contains only the requested evolving cell fields
and result metadata.

Python ``Simulation.step`` uses this mode automatically. Physical object
composition is documented in :doc:`pythonInterface`; serialization and
snapshot namespaces are documented in :doc:`openpmdTransport`.

Arguments
---------

``--input-path=<series>``
   Required HASE openPMD input series.

``--output-path=<series>``
   Required destination for result iterations.

``--cpp-control``
   Optional compiled simulation mode. Without it, the executable performs one
   PhiASE evaluation.

No other command-line options are accepted. Physics, sampling, compute, and
transport settings belong to the openPMD request rather than to a second binary
CLI configuration surface.

MPI
---

The same executable runs under MPI and consumes the same transport layout.
Build requirements, frontend launching, rank/device distribution, and scheduler
examples are centralized in :doc:`mpi`.
