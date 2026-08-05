# JULIA_gain_1D small-signal-gain reference

`julia1D.csv` is an independently generated one-dimensional small-signal-gain
(SSG) curve used by
`tests/python/simulation/test_laserPumpCladdingSsgRegression.py`. The test runs
HASEonGPU's `laserPumpCladding.py` example with ASE disabled and compares its
pump-only gain evolution with the committed Julia result.

The reference was regenerated on 2026-08-04 and is byte-identical to the
historical CSV. `julia1D.metadata.json` records the exact source revision,
driver, parameters, Julia executable, resolved manifest, material checksum,
and output checksum.

## Private source repository

The generator repository is **private** and is deliberately not a submodule or
CI dependency:

- Repository:
  <https://codebase.helmholtz.cloud/penelope-julia/julia_gain_1d.git>
- Required commit: `7e01494a4984ab02dd14fec6f4b7f055e01f49b9`
- Access: contact **Dr. Daniel Albach**

CI consumes the committed CSV without cloning or executing JULIA_gain_1D.
Possession of the metadata, manifest, and HASE-side driver does not grant or
replace access to the private source and its material data.

## Physical and numerical case

The reference models a 7 mm, 2 at.% Yb:YAG crystal at 300 K. It is pumped at
940 nm with 16 kW/cm² for 1 ms in a forward/backward double-pass geometry,
then allowed to decay for 1 ms. SSG is evaluated at 1030 nm with two gain
passes. The run uses 101 time points, 10 crystal points, and pump-consistent
RK4. The CSV therefore spans `t = 0` through `2 ms` in `20 µs` increments.

HASE produces snapshots only after completed steps, so row zero is checked as
the Julia initial condition and rows 1 through 100 are compared with HASE.
The comparison parameters and tolerances are recorded in
`julia1D.metadata.json` and consumed directly by the test.

## Regeneration

Check out the private repository at the recorded commit and keep it clean:

```console
$ git -C /path/to/julia_gain_1d checkout --detach \
    7e01494a4984ab02dd14fec6f4b7f055e01f49b9
$ JULIA_GAIN_1D_ROOT=/path/to/julia_gain_1d \
    JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python3 scripts/regenerate_julia_gain_1d_fixture.py
```

The wrapper refuses a different revision, a dirty checkout, changed project or
material data, a different Julia executable, or a changed committed manifest.
It executes the pinned Julia entrypoint embedded in
`scripts/regenerate_julia_gain_1d_fixture.py`, rewrites the CSV and metadata,
and validates the output grid. The entrypoint source hash is recorded in the
metadata, and the pinned dependency resolution is stored in
`juliaGain1D-manifest.toml`. No Julia scripts are stored in this repository.

Committed metadata and checksums can be verified without Julia or the private
checkout:

```console
$ python3 scripts/regenerate_julia_gain_1d_fixture.py --validate-only
JULIA_gain_1D fixture metadata is valid
```
