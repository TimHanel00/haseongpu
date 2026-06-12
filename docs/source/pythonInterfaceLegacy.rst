Retired Low-Level Python Interface
==================================

This page is kept only as a migration note. The old public
``from HASEonGPU import calcPhiASE`` direct call is not part of the documented
Python interface in the current 2.0 code path. New code should use
:doc:`Python Interface Guide <pythonInterface>` and configure ASE through
``PhiASE``.

Current Execution Path
----------------------

``PhiASE.run(...)`` serializes ``GainMedium``, spectra, and compute settings to
the HASE openPMD transport, launches the compiled ``calcPhiASE`` backend, and
reads ``core_result_*`` records back into Python. The standalone binary accepts
only ``--input-path`` and ``--output-path`` and reads all simulation settings
from the openPMD input series.

See :doc:`openpmdTransport` for the current transport schema, storage backend
options, dynamic-iteration behavior, MPI launch examples, and unsupported
metadata such as ``write_vtk`` and explicit ``devices``.

Migration Guide
---------------

Replace direct argument lists with domain objects:

* ``MeshTopology`` owns points, triangle connectivity, levels, and thickness.
* ``GainMedium`` owns ``betaCells``, ``betaVolume``, cladding data, refractive
  indices, reflectivities, ``nTot``, and ``crystalTFluo``.
* ``CrossSectionData`` or ``SpectralDecomposition`` owns wavelength and
  cross-section tables.
* ``PhiASE`` owns sampling, convergence, reflection, backend, sample-range, and
  optional RNG seed settings.

For workflow examples, start with :doc:`pythonInterface`. For generated
signatures, use :doc:`pythonAPI`.
