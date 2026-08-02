Cross-section tables
====================

``CrossSectionTable`` belongs to ``MaterialDefinition`` rather than to the mesh,
pump, or solver. It contains one strictly increasing wavelength grid and
non-negative absorption and emission arrays.

.. code-block:: python

   spectra = CrossSectionTable(
       wavelengths=[900e-9, 1030e-9],
       absorption=[1.1e-25, 1.2e-25],
       emission=[2.0e-24, 2.48e-24],
   )

All values use SI units (metres and square metres). ``monochromatic`` creates a
one-sample table. Historical four-file tables can be loaded with:

.. code-block:: python

   spectra = CrossSectionTable.from_directory("example/input")

This reads ``lambda_a.txt`` and ``lambda_e.txt`` as nanometres and
``sigma_a.txt`` and ``sigma_e.txt`` as ``cm^2``. It creates the union of both
wavelength grids and interpolates missing absorption or emission values.

Arrays are copied, validated, and made read-only. The private openPMD 0.1
adapter converts cross sections back to the native wire units; user code should
never perform that conversion.
