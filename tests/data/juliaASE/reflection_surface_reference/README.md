# JuliaASE reflection surface reference

This fixture is a single-Tet4 surface-reflection regression case. HASE CI
consumes `reference.json` without importing or executing JuliaASE. HASE and
JuliaASE use independent RNG implementations, so comparisons use the
statistical tolerances stored in the fixture.

## Reference provenance

The reference was regenerated on 2026-08-04 from an exact, clean JuliaASE
checkout:

- JuliaASE repository:
  <https://codebase.helmholtz.cloud/penelope-julia/julia_ase.git>
- JuliaASE commit: `f6e19290ac06f7fd6f9492e6bb14a86973a50166`
- Julia version: `1.10.9`
- Julia distribution SHA-256:
  `5a2d2c5224594b683c97e7304cb72407fbcf0be4a0187789cba1a2f73f0cbf09`
- JuliaASE `Project.toml` SHA-256:
  `8e79c420e591978d179233554597cbaa896debd18baf119c4de325b71695f7d7`
- Resolved manifest: `juliaase-manifest.toml`, SHA-256
  `6bc0d15a228289b5c6f6c2b50e7ed6e1b862729a2ac23657dccecfab8251347a`
- Embedded driver source in `scripts/regenerate_juliaase_reflection_fixture.py`,
  SHA-256
  `35f6a67b006446f0f244f6e5df75b9c692fda39486dd99b4e38b5585625f8961`
- Python regeneration script: `scripts/regenerate_juliaase_reflection_fixture.py`

The JuliaASE repository is **private**. External reviewers and newer
maintainers who need the source checkout should contact **Daniel Albach** to
request access. The committed manifest fixes dependency resolution but does
not replace access to the private JuliaASE source.

`reference.json` records the exact commit, clean-checkout status, timestamp,
tool versions, platform, checksums, command, parameters, driver result, and
driver warnings under `referenceGeneration`. The wrapper rejects a different
JuliaASE commit, a dirty checkout, changed embedded driver source or
`Project.toml`, or a Julia executable other than the recorded 1.10.9 binary.
It uses the committed manifest in an isolated temporary Julia project, so
regeneration does not create or update a manifest inside the private checkout.
No Julia scripts are stored in this repository.

## JuliaASE execution parameters

The driver was run with the following settings:

| Setting | Value |
| --- | --- |
| Ray histories | `1_000_000` |
| RNG | `Random.MersenneTwister(12345)` |
| Source cell mask | `trues(1)` |
| Radiative lifetime (`tau_rad`) | `1.0` |
| Wavelength | `1030.0` nm |
| SRM convergence tolerance (`epsilon`) | `1.0e-5` |
| Maximum reflection passes | `4` |
| Divergence streak | `3` |
| Per-ray step limit | `10_000` |
| Execution | CPU, `nthreads=1`, `n_chunks=1`, `use_gpu=false` |
| JuliaASE memory tier | `TIER3` |
| SRM reservoir size | `64` per boundary face |
| Polarization tracking | disabled |
| Julia/OMP/OpenBLAS threads | `1` / `1` / `1` |
| Julia load path | `@:@stdlib` |
| Julia startup file | disabled |

The one-cell state used the Float32 literals `beta=0.18f0`, absorption
`0.01f0`, and emission `0.02f0` to derive the net gain passed to JuliaASE. The
runtime gain was `-0.004599999636411667` (Float32), and
`Float64(beta) / 6.0` gave a source rate of `0.030000001192092896`. The driver
did not install the absorption and emission values as JuliaASE cross-section
tables.

The refractive index was `1.5` inside and `1.0` outside. JuliaASE faces 1--3
were absorbing. Face 4 used a constant Float32 coating with s- and
p-reflectivity `0.65` and transmissivity `0.35` over 0--90 degrees and
1030.0--1030.1 nm.

## Regeneration

After obtaining the private checkout, select the recorded commit and put Julia
1.10.9 on `PATH`:

```console
$ git -C /path/to/juliaASE checkout --detach \
    f6e19290ac06f7fd6f9492e6bb14a86973a50166
$ JULIAASE_ROOT=/path/to/juliaASE \
    JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python3 scripts/regenerate_juliaase_reflection_fixture.py \
    --ray-count 1000000
```

Use `--dry-run` to print the complete regenerated JSON without overwriting the
fixture. The wrapper instantiates `juliaase-manifest.toml` and records the
actual runtime metadata on every run.

The committed metadata and file checksums can be checked without JuliaASE or a
native HASE build:

```console
$ python3 scripts/regenerate_juliaase_reflection_fixture.py --validate-only
JuliaASE reflection fixture metadata is valid
```

## Stored values

JuliaASE directly produced `phiAse`, `reflectedPassWeightFractions`,
`referenceRayCount`, and `referenceInitialReflectedWeight`. The new run
converged after one pass. The Python wrapper converted JuliaASE's rate sign to
the HASE/openPMD convention and derived `dndtAse` as

```text
(beta * (sigmaEmission + sigmaAbsorption) - sigmaAbsorption) * phiAse
```

It then calculated `finalBetaVolume` as
`beta - timeStep * dndtAse`, with `timeStep=0.05`. Thus, `timeStep`, `dndtAse`,
and `finalBetaVolume` are fixture post-processing rather than JuliaASE
transport inputs.

The regenerated direct values are:

```text
phiAse                         0.052214350551366806
referenceInitialReflectedWeight 185603.515625
reflectedPassWeightFractions   [0.0]
status                         converged
passes                         1
```

## Driver warnings

The exact run emitted two JuliaASE warnings, preserved with filesystem paths
normalized to `$JULIAASE_ROOT` and `$HASE_ROOT` in
`referenceGeneration.result.driverStderr`:

1. The mesh extent is 1.0 m, above JuliaASE's 0.05 m kernel-epsilon envelope.
2. With reservoir capacity `K=64`, 100% of reflected energy is on a saturated,
   angularly broad face, so the reflected source may be undersampled.

These warnings are part of the recorded run and must not be silently removed
from the provenance. This fixture is a cross-implementation regression with a
statistical tolerance; it is not an analytic validation of JuliaASE at this
mesh scale or a reservoir-size convergence study.

## Face mapping and mesh

HASE local face 0 and JuliaASE local face 4 are both the face opposite the
fourth vertex, i.e. the `z=0` face. Physical surface 11 therefore selects the
same face in both implementations. HASE's explicit 0.65 surface reflectivity is
represented by the constant JuliaASE coating described above; total internal
reflection remains enabled.

`single_reflective_tet4.msh` contains the one Tet4 volume and physical surface
11 used by the surface-domain optics arrays in `reference.json`. Both
`rayCount` (the corresponding HASE comparison setting) and `referenceRayCount`
(the JuliaASE generation setting) are `1_000_000`.
