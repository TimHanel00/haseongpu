Legacy Low-Level Python Interface
=================================

This page documents the compatibility status of the historical low-level
``calcPhiASE(...)`` Python entry point. Workflows should use the high-level
:doc:`Python Interface Guide <pythonInterface>` and configure ASE through
``PhiASE``.

Current Compatibility
---------------------

``from HASEonGPU import calcPhiASE`` is no longer exported. The HASE pybind
module and direct in-process backend adapters were removed with the forward-only
openPMD backend transition.

The removed legacy text-file backend parser is not restored. Use the openPMD
transport path for both single-rank and MPI execution.

Modern Execution Path
---------------------

``PhiASE.run(...)`` serializes ``GainMedium``, spectra, and compute settings to
the HASE openPMD transport, launches the compiled ``calcPhiASE`` backend, and
reads ``core_result_*`` records back into Python. The standalone binary accepts
only ``--input-path`` and ``--output-path`` and reads simulation settings from
the openPMD input series.

See :doc:`openpmdTransport` for the openPMD record layout, storage backend
options, dynamic-iteration behavior, MPI launch examples, and unsupported
metadata such as ``write_vtk`` and explicit ``devices``.

Migration Guide
---------------

Prefer replacing direct argument lists with domain objects:

* ``MeshTopology`` owns points, triangle connectivity, levels, and thickness.
* ``GainMedium`` owns ``betaCells``, ``betaVolume``, cladding data, refractive
  indices, reflectivities, ``nTot``, and ``crystalTFluo``.
* ``CrossSectionData`` or ``SpectralDecomposition`` owns wavelength and
  cross-section tables.
* ``PhiASE`` owns sampling, convergence, backend, sample-range, and
  optional RNG seed settings.

For workflow examples, start with :doc:`pythonInterface`. For generated
signatures, use :doc:`pythonAPI`.
