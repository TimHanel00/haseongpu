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
one-sample table. ``from_directory`` reads the historical ``lambda_a.txt``,
``sigma_a.txt``, ``lambda_e.txt``, and ``sigma_e.txt`` files and converts their
nm/cm² values to SI.

The private openPMD 0.1 adapter converts cross sections back to the legacy
backend units; user code should never perform that conversion.
