Material library, temperature states, and HDF5
==============================================

``materialLibrary`` is an independently importable package for unit-aware,
temperature-resolved optical data. It does not import the HASEonGPU mesh,
simulation, native runtime, or openPMD transport. The dependency direction is
one way: a simulation consumes a resolved ``MaterialCondition``. Physical unit
types come from the separate ``hase_units`` package shared with HASEonGPU.

The four material layers
------------------------

The library separates identity, measured state, run condition, and storage:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Object
     - Responsibility
   * - ``Material``
     - A named material and its ordered temperature states.
   * - ``MaterialState``
     - Refractive index, lifetime, cross sections, attenuation, and provenance
       at one documented temperature.
   * - ``MaterialCondition``
     - An immutable state resolved for one simulation, augmented with the
       run-specific active-ion density, name, and optional optical axis.
   * - ``MaterialLibrary``
     - A keyed collection of materials with versioned HDF5 persistence.

Use the bundled database
------------------------

Importing ``HASEonGPU`` loads the repository database as
``HASEonGPU.materialLibrary``. Its historical Yb:YAG entry is selected in two
lines:

.. code-block:: python

   from HASEonGPU import materialLibrary, units
   crystal = materialLibrary["YbYAG"].at(
       temperature=293.15 * units.K,
       activeIonDensity=2.776e20 / units.cm**3,
   )

The cross sections in the historical source tables were recorded at room
temperature. The database stores this explicitly as ``293.15 K`` and records
the room-temperature context in the state metadata.

For standalone tools that should not import HASEonGPU, load the same database
directly:

.. code-block:: python

   from materialLibrary import loadBuiltinMaterials
   database = loadBuiltinMaterials()

Define a temperature-resolved material
--------------------------------------

Create a ``Material`` and add measured or modelled states with provenance:

.. code-block:: python

   from materialLibrary import Material
   from hase_units import units

   yag = Material("Yb:YAG", metadata={"doi": "10.example/source"})
   yag.addState(temperature=280 * units.K, refractiveIndex=1.82,
                 fluorescenceLifetime=0.96 * units.ms,
                 crossSections=spectra_280, metadata={"sample": "cold"})
   yag.addState(temperature=320 * units.K, refractiveIndex=1.84,
                 fluorescenceLifetime=0.90 * units.ms,
                 crossSections=spectra_320, metadata={"sample": "warm"})

Temperatures must be positive quantities. A state whose source lacks a
temperature uses ``temperature=None`` and must explain why:

.. code-block:: python

   reference = Material("reference")
   reference.addState(temperature=None, refractiveIndex=1.82,
       metadata={"temperature_status": "not documented by source"})

An unknown-temperature reference cannot be mixed with numeric temperature
states in the same ``Material``.

Select and interpolate a run condition
--------------------------------------

Exact selection is the default:

.. code-block:: python

   crystal = yag.at(temperature=280 * units.K,
                    activeIonDensity=2.776e20 / units.cm**3,
                    spectralResolution=1000)

Request linear interpolation explicitly when no exact state exists:

.. code-block:: python

   crystal = yag.at(temperature=300 * units.K, interpolation="linear",
                    activeIonDensity=2.776e20 / units.cm**3)

Interpolation emits ``TemperatureInterpolationWarning`` with the requested and
supporting temperatures. Refractive index, lifetime, bulk attenuation, and
both cross-section curves are interpolated together. Different wavelength
grids are first combined into one grid. Extrapolation and interpolation across
a property missing from either bounding state are rejected.

The returned ``MaterialCondition`` is the object passed to
``Simulation.addMaterial``. Material metadata is merged with the selected
state metadata, with the state taking precedence, and the interpolation mode
is recorded in the condition metadata.

``spectralResolution`` linearly resamples the selected cross-section table
before it is passed to transport. The material library uses the former native
transport semantics: an endpoint-inclusive equidistant wavelength grid,
piecewise-linear cross sections, and the exact final input sample as the final
output sample. The implementation is vectorized with NumPy. The requested
resolution must be at least the tabulated sample count; downsampling is
rejected because it would discard material data. The returned table records
the source and output sample counts in ``spectral_resampling`` metadata.

HDF5 persistence
----------------

``MaterialLibrary`` stores several materials under Python-identifier keys:

.. code-block:: python

   from materialLibrary import MaterialLibrary
   database = MaterialLibrary()
   database.register("YbYAG", yag)
   database.toHdf5("optical-materials.h5")

   restored = MaterialLibrary.fromHdf5("optical-materials.h5")
   cold = restored["YbYAG"].at(temperature=280 * units.K)

Material lookup is deliberately mapping-only: use ``restored["YbYAG"]`` so a
misspelled key raises ``KeyError``. Keys must be valid Python identifiers and
duplicates are rejected. Writing is non-destructive by default and fails if
the path exists; pass ``overwrite=True`` only when replacement is intended.

For a file containing one material, use the convenience methods:

.. code-block:: python

   yag.toHdf5("yb-yag.h5", key="YbYAG")
   yag = Material.fromHdf5("yb-yag.h5")

If a file contains several materials, ``Material.fromHdf5(..., key="YbYAG")``
requires an explicit key.

The root carries the format name and version. Every physical dataset carries
``unit``, ``unitSI``, and the seven-component ``unitDimension``; material,
state, and spectrum provenance is stored as JSON metadata. This is a HASE
material database, not an openPMD series, although it uses the same unit
metadata convention. HDF5 persistence requires the Python ``h5py`` package.

Import legacy text tables
-------------------------

HDF5 is canonical. ``CrossSectionTable.fromTextDirectory(path)`` imports the
historical four-file convention as ``nm`` and ``cm^2`` but emits
``LegacyMaterialTextWarning`` because the files carry no self-describing units,
temperature, or provenance.

Import the spectra explicitly, attach the missing physical state, and write the
library through the regular Python API:

.. code-block:: python

   spectra = CrossSectionTable.fromTextDirectory("legacy-data", metadata={"source": "citation or DOI"})
   material = Material("Yb:YAG").addState(temperature=None, refractiveIndex=1.82, fluorescenceLifetime=0.941 * units.ms, crossSections=spectra, metadata={"temperature_status": "not documented by source"})
   library = MaterialLibrary()
   library.register("YbYAG", material)
   library.toHdf5("material.h5")

Use a temperature quantity such as ``300 * units.K`` only when the source
actually documents that temperature. Pass ``overwrite=True`` to ``toHdf5``
only when replacing an existing target file intentionally.
