Physical quantities and units
=============================

Physical input values carry explicit units. The independent ``hase_units``
package owns ``Unit``, ``Quantity``, physical dimensions, and the unit
catalogue. Both ``materialLibrary`` and the HASEonGPU simulation frontend use
those same types directly; neither package owns a second unit system.

HASEonGPU re-exports the shared catalogue for convenient simulation scripts:

.. code-block:: python

   from HASEonGPU import units
   lifetime, density = 0.941 * units.ms, 2.776e20 / units.cm**3

Standalone material and data tools can import the common package without
loading HASEonGPU or ``materialLibrary``:

.. code-block:: python

   from hase_units import Quantity, Unit, units

HASEonGPU checks dimensions at assembly boundaries and retains the selected
unit on the Python ``Quantity``. At the transport boundary it converts values
to the native representation and writes explicit openPMD unit metadata.

The catalogue provides ``m``, ``cm``, ``mm``, ``um``, ``nm``, ``s``, ``ms``,
``us``, ``kg``, ``g``, ``A``, ``K``, ``mK``, ``mol``, ``cd``, ``W``, ``kW``,
and dimensionless ``one``. Multiplication, division, and integer powers create
derived units:

.. code-block:: python

   attenuation = 0.02 / units.cm
   print(attenuation.toValue(units.m**-1))

Dimensionless physical state can still require an explicit ``Quantity``.
``InitialState`` uses ``units.one`` so accidental dimensional input is caught.
Pure ratios and normalized controls such as refractive index, reflectivity,
spectrum weights, and RSE remain ordinary numbers.

Geometry scale
--------------

Mesh coordinates are stored as numbers, so ``coordinateUnit`` supplies their
physical scale. The same point array can therefore describe metres or
centimetres without rewriting it:

.. code-block:: python

   mesh = UnstructuredMesh.fromFile("crystal.msh", coordinateUnit=units.mm)

The coordinate unit controls volumes, ray segment lengths, and spatial pump
profiles. It must describe length.

Excited-state fraction
----------------------

The excitation fraction :math:`\beta` is the fraction of active ions in the
upper laser level. It is dimensionless and must lie in ``[0, 1]``:

.. code-block:: python

   initialState = InitialState(0.10 * units.one)

For cell-varying initialization, pass one value per Tet4 or map mesh selections
to values:

.. code-block:: python

   initialState = InitialState({mesh.volume("gain"): 0.10 * units.one,
                                 mesh.volume("cap"): 0.0 * units.one})

Temperature
-----------

``MaterialState.temperature`` is an absolute thermodynamic temperature and
therefore uses ``units.K`` (or a compatible unit such as ``units.mK``):

.. code-block:: python

   condition = yag.at(temperature=300 * units.K)

Temperature is part of material-data selection, not a global simulation
control. See :doc:`material_library` for exact selection, interpolation, and
the explicit unknown-temperature state.

Active-ion density
------------------

``activeIonDensity`` is the total density :math:`N_{\mathrm{tot}}` of active
ions, not the excited-state population. Their excited density is
:math:`\beta N_{\mathrm{tot}}`. The value belongs to a run-specific
``MaterialCondition``:

.. code-block:: python

   crystal = yag.at(temperature=300 * units.K,
                    activeIonDensity=2.776e20 / units.cm**3)

Fluorescence lifetime
---------------------

``fluorescenceLifetime`` is the upper-state lifetime :math:`\tau`. It sets the
spontaneous decay term :math:`-\beta/\tau` and participates in the physical
scaling of the ASE flux:

.. code-block:: python

   yag = Material("Yb:YAG").addState(
       temperature=300 * units.K, refractiveIndex=1.82,
       fluorescenceLifetime=0.941 * units.ms, crossSections=spectra)

Absorption and emission cross sections
--------------------------------------

The wavelength-dependent cross sections :math:`\sigma_a(\lambda)` and
:math:`\sigma_e(\lambda)` describe a photon's absorption and stimulated
emission probability per active ion. Together with :math:`\beta` and
:math:`N_{\mathrm{tot}}`, they form the local gain coefficient

.. math::

   g(\lambda) = N_{\mathrm{tot}}
   \left[\beta\left(\sigma_e(\lambda)+\sigma_a(\lambda)\right)
   - \sigma_a(\lambda)\right].

Provide one increasing wavelength grid and equally sized area-valued arrays:

.. code-block:: python

   spectra = CrossSectionTable(units.nm * [900, 1030],
                               units.cm**2 * [1.1e-21, 1.2e-21], units.cm**2 * [2.0e-20, 2.48e-20])

Interpolate a property without manually converting units:

.. code-block:: python

   sigma_pump = spectra.absorptionAt(940 * units.nm)

Refractive index and attenuation
--------------------------------

``refractiveIndex`` is dimensionless. The current boundary model uses the
interior and exterior indices to identify total internal reflection;
``ConstantReflectivitySurface.reflectivity`` supplies the otherwise constant
reflected fraction:

.. code-block:: python

   boundary = ConstantReflectivitySurface(reflectivity=0.04,
                                           exterior_refractive_index=1.0)

``bulkAttenuation`` is a non-negative inverse length for passive loss. It can
be expressed by the frontend, although the current native adapter rejects this
per-material feature before launch:

.. code-block:: python

   glass = Material("glass").addState(
       temperature=300 * units.K, refractiveIndex=1.50,
       bulkAttenuation=0.02 / units.cm)

Pump quantities
---------------

``Pump.totalPower`` is the aperture-integrated optical power. Pump spectral
wavelengths and profile radii are length quantities; spectrum and angular
weights are normalized and dimensionless:

.. code-block:: python

   pump = Pump(16 * units.kW, PumpSpectrum.monochromatic(940 * units.nm),
               profile=SuperGaussianPumpProfile(1.5 * units.cm, exponent=40))

Cross sections do not belong to a pump. The solver evaluates the material's
cross-section table at the sampled pump wavelength.

Time
----

Simulation time controls require time quantities. The frontend retains the
chosen units while the backend receives seconds:

.. code-block:: python

   simulation = Simulation(..., timeStepSize=20 * units.us,
                           maxTime=3 * units.ms)

See :doc:`uncertainty` for the dimensionless Monte Carlo RSE and
:doc:`../theoryAndModel` for the complete transport and population equations.
