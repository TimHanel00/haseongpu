# JuliaASE laserPumpCladding five-step reference

This fixture is the JuliaASE side of an end-to-end regression for
`example/laserPumpCladding.py`. CI runs the HASE example twice, with ASE
reflections disabled and enabled, and compares **every one of five simulation
steps** with the committed JuliaASE time series at 5% relative tolerance.
JuliaASE is not required in CI.

The compared observables are the volume-weighted mean inversion fraction
(`beta`) and the
volume integral of PhiASE. Integrated observables are used because HASE and
JuliaASE intentionally use independent tetrahedral meshes and RNG
implementations. The first PhiASE value is zero in both implementations due to
the shared `prePump=true` initial step; the test compares the four subsequent
ASE evaluations rather than checking only the final snapshot.

## Shared case

- 60 mm diameter, 7 mm thick, 2 at.% Yb:YAG disk at 300 K
- 20 us time step; five pumped simulation steps
- 940 nm, 16 kW/cm2 double-pass pump
- 15 mm-radius Super-Gaussian pump profile with exponent 40
- 50,000 pump rays and 30,000 ASE rays
- 191-point 905--1095 nm ASE spectrum
- frozen-PhiASE RK4 and a cold initial inversion
- ideal-AR end surfaces, either with TIR enabled against air or with all ASE
  reflections disabled; the cylindrical edge is absorbing

HASE uses its `example/data/ptTet4.vtk` mesh in legacy centimetres. JuliaASE
uses its pinned `disk_debug.msh` mesh in SI units. The regression converts the
HASE PhiASE integral from photons*cm/s to photons*m/s before comparison.

This is an outcome regression between two independent implementations, not a
claim that their discretizations are identical. In particular, JuliaASE uses
per-wavelength ASE depletion and freezes its pump rate during an RK4 step;
HASE freezes the integrated PhiASE field and reevaluates its pump rate at the
RK4 stages. JuliaASE's relay source also represents the nominal 940 nm,
collimated pump with a narrow 1 nm spectral interval and 1 mrad cone, whereas
HASE uses an exactly monochromatic and collimated source. JuliaASE samples the
Super-Gaussian inside its 15 mm source aperture; HASE integrates the same
profile over the full bottom face, whose omitted tail contributes about 1%.
Those implementation
differences, the independent meshes, and independent RNG streams are why the
test compares physical aggregate outcomes at 5%, rather than expecting
bitwise or cellwise agreement.

## Provenance and regeneration

The reference was generated with Julia 1.10.9 from the private JuliaASE
repository at commit
`f6e19290ac06f7fd6f9492e6bb14a86973a50166`. The exact JuliaASE full-example
source, debug mesh, material database, Julia executable, project, and manifest
checksums are recorded or pinned by `reference.json` and the regeneration
wrapper.

The Python regeneration fixture
`scripts/regenerate_juliaase_laser_pump_cladding_fixture.py` embeds the small
Julia driver that loads JuliaASE's own `test/simulation_run/run_full_simulation.jl`,
selects the two HASE boundary modes, applies HASE's pre-pump convention, and
emits the five-step observables. No standalone Julia fixture script is stored
in this repository.

After checking out the pinned private revision and placing the exact Julia
1.10.9 distribution on `PATH`, regenerate with:

```console
$ JULIAASE_ROOT=/path/to/juliaASE \
    JULIA_DEPOT_PATH=/path/to/isolated/depot \
    python3 scripts/regenerate_juliaase_laser_pump_cladding_fixture.py
```

Validate committed provenance without executing JuliaASE:

```console
$ python3 scripts/regenerate_juliaase_laser_pump_cladding_fixture.py --validate-only
JuliaASE laserPumpCladding fixture metadata is valid
```

The JuliaASE repository is private. Contact Daniel Albach if source access is
required.
