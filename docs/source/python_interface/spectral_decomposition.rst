Absorption and emission spectra
===============================

``CrossSectionTable`` stores the wavelength-dependent material cross sections
used by both ASE and pump transport. It belongs to a temperature-specific
``MaterialState``;
cross sections are properties of the material a photon traverses, not of the
mesh, pump, or Monte Carlo solver.

Physical meaning
----------------

``absorption`` is :math:`\sigma_a(\lambda)`, the effective absorption area per
active ion. ``emission`` is :math:`\sigma_e(\lambda)`, the stimulated-emission
area per active ion. Their values combine with active-ion density and
excitation fraction to produce the local gain coefficient described in
:doc:`gain_medium`.

The table requires one strictly increasing wavelength grid and two equally
sized, finite, non-negative area arrays:

.. code-block:: python

   spectra = CrossSectionTable(units.nm * [900, 940, 1030],
                               units.cm**2 * [1.1e-21, 1.6e-21, 1.2e-21],
                               units.cm**2 * [2.0e-20, 2.2e-20, 2.48e-20])

The selected units are retained in the Python object. The transport adapter
converts them to its native representation and writes consistent openPMD unit
metadata; application code should not insert manual ``nm`` or ``cm^2``
conversion factors.

Interpolation and inspection
----------------------------

Query either curve with a compatible wavelength quantity:

.. code-block:: python

   sigma_abs = spectra.absorptionAt(940 * units.nm)
   print(sigma_abs.toValue(units.cm**2))

``monochromatic`` creates a one-wavelength table for deliberately
single-frequency studies:

.. code-block:: python

   spectra = CrossSectionTable.monochromatic(wavelength=1030 * units.nm,
       absorption=1.2e-21 * units.cm**2, emission=2.48e-20 * units.cm**2)

Material files
--------------

Historical four-file tables can still be imported:

.. code-block:: python

   spectra = CrossSectionTable.fromTextDirectory("legacy-material-data")

This reads ``lambda_a.txt`` and ``lambda_e.txt`` as nanometres and
``sigma_a.txt`` and ``sigma_e.txt`` as ``cm^2``. It forms the union of the two
wavelength grids and interpolates each curve onto that grid. The resulting
object still exposes its declared ``nm`` and ``cm^2`` units. This import emits
``LegacyMaterialTextWarning`` because the text representation has no place for
temperature, units, or provenance. Prefer the HDF5 format described in
:doc:`material_library`.

Sampling versus material resolution
-----------------------------------

The number of tabulated wavelengths describes the material data. A resolved
condition can request a denser grid before transport:

.. code-block:: python

   condition = material.at(temperature=300 * units.K,
                           spectralResolution=1000)

This frontend interpolation is separate from Monte Carlo sampling resolution.
Increasing numerical spectral resolution cannot recover detail absent from the
measured cross-section data.
