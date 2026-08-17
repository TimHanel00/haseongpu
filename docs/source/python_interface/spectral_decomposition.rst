Material spectra
================

Absorption and emission spectra are physical ``Material`` properties.
``CrossSectionTable`` stores unit-bearing wavelength, absorption, and emission
arrays in the HDF5 material database:

.. code-block:: python

   from HASEonGPU import CrossSectionTable, units

   table = CrossSectionTable(
       wavelengths=[930, 940, 950] * units.nm,
       absorption=[6.7e-21, 7.8e-21, 8.1e-21] * units.cm**2,
       emission=[1.5e-21, 1.9e-21, 2.4e-21] * units.cm**2,
   )

``Material.fromHdf5`` resolves the table along with the material's other
temperature-dependent properties. ``spectralResolution`` optionally creates an
endpoint-inclusive linear grid without downsampling the source data.

Both curves share one wavelength grid. Wavelengths must be positive and
strictly increasing, except for the repeated constant grid produced when
explicitly resampling monochromatic data. Absorption and emission cross
sections must be finite, non-negative area quantities. ``absorptionAt`` and
``emissionAt`` perform linear interpolation at a unit-bearing wavelength.

Legacy four-file data
---------------------

``CrossSectionTable.fromTextDirectory`` imports ``lambda_a.txt``,
``lambda_e.txt``, ``sigma_a.txt``, and ``sigma_e.txt``. It assumes the
historical units of nanometres and square centimetres, forms the union
wavelength grid, and interpolates each curve to that grid.

The importer always emits ``LegacyMaterialTextWarning`` because the text files
do not encode units, temperature, or provenance. Convert retained datasets to
the versioned material-library HDF5 format and record their source in metadata.

``Simulation`` converts the executable gain material to the current backend's
``CrossSectionData`` representation once. The same converted spectrum is used
by ASE and every pump, so schema version 3 does not contain a separate
cross-section registry.

``CrossSectionData`` and ``LaserProperties`` remain supporting low-level APIs
for direct backend transport and result analysis. They are not separate
physical objects in the five-type frontend graph.
