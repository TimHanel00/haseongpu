# EXT_HASE: HASE openPMD Extension and Transport Schema

This document describes the HASE-owned openPMD schema currently written by the
Python frontend and consumed by the C++ `calcPhiASE` openPMD backend.

The schema source of truth is the Python implementation in
`pyInclude/openpmd/schema/` and `pyInclude/openpmd/transport.py`, together with
the C++ parser in `src/openpmd/OpenPmdParser.cpp`. This document is a compact
contract reference; it is not parsed by the frontend or backend.

## Extension Markers

HASE writes these root attributes on the openPMD series:

| attribute             | value                   |
|-----------------------|-------------------------|
| haseOpenPMDextension  | HASE                    |
| haseVersion           | 0.1                     |
| haseSchemaVersion     | 0.1                     |
| simulationType        | laserCrystalASE         |
| geometryType          | extrudedTriangularPrism |

HASE does not write the openPMD-reserved `openPMDextension` attribute. If HASE
later registers a formal openPMD extension identifier, the writer can add that
reserved marker without changing the HASE-owned record inventory below.

## Layer Boundaries

The Python interface is primitive-oriented. Users describe `points`,
`triangles`, and extruded triangular `prisms` through high-level objects such
as `MeshTopology`, `GainMedium`, and `PhiASE`. Users may also add primitive
fields through `GainMedium.defineField(...)` or schema inheritance from
`BaseSchema`.

The openPMD transport maps those domain objects to mesh records and iteration
attributes. Transport details such as the `core_` record prefix,
`backendFlat`, `recordC`, and `haseStaticUpdate` are serialization details.
They are not Python modeling names.

The C++ parser consumes the canonical openPMD records listed below. It derives
backend-only triangle centers, normals, forbidden edges, neighbors, and
surfaces from canonical wedge topology instead of requiring those derived
records in the current writer output.

## Naming Conventions

- Python user-facing field names use `lowerCamelCase`, for example
  `betaVolume`, `pointBeta`, `sigmaAbsorption`, and `dndtAse`.
- Transport record names use `snake_case`.
- Backend input records use the current mesh group prefix `core_`.
- Backend result records use the prefix `core_result_`.
- User-defined primitive fields use their declared record name. Fields created
  by `GainMedium.defineField(...)` default to `custom_<snake_case_name>` and do
  not use the `core_` prefix.
- Iteration attributes use `snake_case`, for example `number_of_points`,
  `max_sigma_emission`, and `min_sample_range`.
- HASE metadata attributes use `hase` plus `PascalCase`, for example
  `haseSchemaVersion`, `hasePrimitiveShape`, and `haseBackendRequired`.

## HASE Mesh Metadata

Every HASE mesh record carries normal openPMD mesh metadata and these HASE
attributes:

| attribute            | meaning                                           |
|----------------------|---------------------------------------------------|
| haseSchemaVersion    | HASE schema version, currently `0.1`              |
| haseEntity           | semantic entity name derived from `haseAxes`      |
| haseAxes             | primitive axes before backend flattening          |
| haseLayoutOrder      | `backendFlat` for input records, `recordC` for results |
| hasePrimitiveShape   | primitive shape before backend flattening         |
| haseStatic           | true when the record is static topology/model data |
| haseDynamic          | true when the record may change between iterations |
| haseBackendRequired  | true when the current C++ backend requires it      |
| haseUnit             | human-readable HASE unit string                   |
| haseUserDefined      | true for user-defined primitive fields            |
| haseUserFieldName    | original user field name for user-defined fields  |

`axisLabels` remains the openPMD axis label list. Flattened input records use
`axisLabels = ["flatIndex"]`; the semantic axes are stored in `haseAxes`.
Result records use `axisLabels = ["point", "level"]` and `haseLayoutOrder =
"recordC"`.

## Iteration Attributes

These values are serialized as iteration attributes for the current backend
contract.

| Python field        | attribute            | dtype | unit   | unitSI | unitDimension |
|---------------------|----------------------|-------|--------|--------|---------------|
| numberOfPoints      | number_of_points     | int   | count  | 1.0    | DIMENSIONLESS |
| numberOfTriangles   | number_of_cells      | int   | count  | 1.0    | DIMENSIONLESS |
| numberOfLevels      | number_of_levels     | int   | count  | 1.0    | DIMENSIONLESS |
| thickness           | thickness            | float | m      | 1.0    | LENGTH        |
| nTot                | n_tot                | float | cm^-3  | 1.0e6  | INV_VOLUME    |
| crystalTFluo        | crystal_t_fluo       | float | s      | 1.0    | TIME          |
| claddingNumber      | cladding_number      | int   | count  | 1.0    | DIMENSIONLESS |
| claddingAbsorption  | cladding_absorption  | float | cm^-1  | 100.0  | INV_LENGTH    |
| minRaysPerSample    | min_rays_per_sample  | int   | count  | 1.0    | DIMENSIONLESS |
| maxRaysPerSample    | max_rays_per_sample  | int   | count  | 1.0    | DIMENSIONLESS |
| mseThreshold        | mse_threshold        | float | 1      | 1.0    | DIMENSIONLESS |
| repetitions         | repetitions          | int   | count  | 1.0    | DIMENSIONLESS |
| adaptiveSteps       | adaptive_steps       | int   | count  | 1.0    | DIMENSIONLESS |
| useReflections      | use_reflections      | bool  | 1      | 1.0    | DIMENSIONLESS |
| spectralResolution  | spectral_resolution  | int   | count  | 1.0    | DIMENSIONLESS |
| monochromatic       | monochromatic        | bool  | 1      | 1.0    | DIMENSIONLESS |
| maxSigmaAbsorption  | max_sigma_absorption | float | cm^2   | 1.0e-4 | CROSS_SECTION |
| maxSigmaEmission    | max_sigma_emission   | float | cm^2   | 1.0e-4 | CROSS_SECTION |
| backend             | backend              | str   | 1      | 1.0    | DIMENSIONLESS |
| maxGpus             | max_gpus             | int   | count  | 1.0    | DIMENSIONLESS |
| parallelMode        | parallel_mode        | str   | 1      | 1.0    | DIMENSIONLESS |
| minSampleRange      | min_sample_range     | int   | index  | 1.0    | DIMENSIONLESS |
| maxSampleRange      | max_sample_range     | int   | index  | 1.0    | DIMENSIONLESS |
| rngSeed             | rng_seed             | int   | 1      | 1.0    | DIMENSIONLESS |

`rng_seed` is optional. If absent, the C++ backend uses its unspecified-seed
path. The Python `PhiASE` wrapper normally supplies a seed for each run unless
the lower-level transport is used directly.

`write_vtk` and `devices` are not supported by the openPMD transport. The
Python writer rejects them before writing. The C++ parser also rejects
`write_vtk = true` and explicit `devices` metadata so unsupported requests fail
clearly.

## Iteration Cadence

Input series can contain multiple iterations.

- Iterations with `haseStaticUpdate = true` contain topology, static material
  records, spectra, and dynamic fields. The first Python-written iteration is a
  static update.
- Iterations with `haseStaticUpdate = false` contain only dynamic fields:
  `core_beta_volume` and `core_point_beta`.
- The C++ parser caches topology and static records from the latest static
  update and applies later dynamic-only iterations to that cached context.
- A dynamic-only iteration before any static update is invalid.

## Primitive Schemas

| primitive | class          | axes         | description                                          |
|-----------|----------------|--------------|------------------------------------------------------|
| point     | PointSchema    | point        | topology points and point-level fields               |
| triangle  | TriangleSchema | cell         | triangular face/cell fields in the base topology     |
| prism     | PrismSchema    | cell, layer  | extruded triangular prism volume fields              |

## Canonical Static Topology Records

The current Python writer uses canonical wedge topology records. These records
are written under the iteration `meshes` group when `haseStaticUpdate = true`.

| record                   | components | dtype   | haseAxes               | hasePrimitiveShape       | axisLabels | dynamic | backendRequired | unit |
|--------------------------|------------|---------|------------------------|--------------------------|------------|---------|-----------------|------|
| core_points              | x, y, z    | float64 | coordinate, mesh_point | 3, numberOfPoints*levels | mesh_point | false   | false           | m    |
| core_cells_connectivity  | SCALAR     | uint32  | cell, local_vertex     | numberOfPrisms, 6        | flatIndex  | false   | false           | 1    |
| core_cells_offsets       | SCALAR     | uint32  | cell_offset            | numberOfPrisms+1         | flatIndex  | false   | false           | 1    |
| core_cells_types         | SCALAR     | uint32  | cell                   | numberOfPrisms           | flatIndex  | false   | false           | 1    |

`core_cells_types` currently contains VTK cell type `13` (`VTK_WEDGE`) for
every prism. `core_cells_offsets` must be contiguous six-node wedge offsets:
`0, 6, 12, ...`.

The parser also keeps a compatibility reader for older static topology inputs
that provide `core_vertices` with components `x` and `y` plus
`core_connectivity`. That path is compatibility-only; new Python-written input
uses the canonical records above.

## Required Input Records

These records are consumed by the current C++ backend. With the exception of
canonical topology, scalar input records are flattened in backend order and use
`axisLabels = ["flatIndex"]`.

| Python field        | record                  | dtype   | haseAxes        | hasePrimitiveShape          | dynamic | backendRequired | unit |
|---------------------|-------------------------|---------|-----------------|-----------------------------|---------|-----------------|------|
| betaVolume          | core_beta_volume        | float64 | cell, layer     | numberOfTriangles, levels-1 | true    | true            | 1    |
| pointBeta           | core_point_beta         | float64 | point, level    | numberOfPoints, levels      | true    | true            | 1    |
| claddingCellType    | core_cladding_cell_type | uint32  | cell            | numberOfTriangles           | false   | true            | 1    |
| refractiveIndex     | core_refractive_index   | float32 | interface       | 4                           | false   | true            | 1    |
| reflectivity        | core_reflectivity       | float32 | cell, interface | numberOfTriangles, 2        | false   | true            | 1    |
| lambdaAbsorption    | core_lambda_absorption  | float64 | wavelength      | spectralResolution          | false   | false           | m    |
| lambdaEmission      | core_lambda_emission    | float64 | wavelength      | spectralResolution          | false   | false           | m    |
| sigmaAbsorption     | core_sigma_absorption   | float64 | wavelength      | spectralResolution          | false   | false           | cm^2 |
| sigmaEmission       | core_sigma_emission     | float64 | wavelength      | spectralResolution          | false   | false           | cm^2 |

`refractiveIndex` has layout `[bottomInside, bottomOutside, topInside,
topOutside]`. `reflectivity` has shape `(numberOfTriangles, 2)` with bottom and
top entries.

## Result Records

The C++ backend writes result records under the `core_result_` prefix. Results
use record-C layout with primitive shape `(numberOfPoints, numberOfLevels)`.

| Python field | record                 | dtype   | haseAxes     | axisLabels   | haseLayoutOrder | unit       | unitSI |
|--------------|------------------------|---------|--------------|--------------|-----------------|------------|--------|
| phiAse       | core_result_phi_ase    | float32 | point, level | point, level | recordC         | cm^-2 s^-1 | 1.0e4  |
| mse          | core_result_mse        | float64 | point, level | point, level | recordC         | 1          | 1.0    |
| totalRays    | core_result_total_rays | uint32  | point, level | point, level | recordC         | count      | 1.0    |
| dndtAse      | core_result_dndt_ase   | float64 | point, level | point, level | recordC         | s^-1       | 1.0    |

## User-Defined Primitive Fields

User-defined primitive fields are valid openPMD mesh records with the same HASE
metadata attributes as built-in records. `haseUserDefined = true` and
`haseUserFieldName` preserve the Python field name. These records are persisted
by the Python writer for downstream tools, but the current C++ ASE backend does
not use them unless a future backend explicitly opts in.

## Backend Selection and Storage Backends

The `backend` iteration attribute selects the alpaka compute backend used by
HASEonGPU, for example `Host_Cpu_CpuSerial`. It is separate from the openPMD
storage backend.

The Python transport can write/read these openPMD storage backends:

| transport name | suffix | openPMD configuration |
|----------------|--------|-----------------------|
| adios          | .bp    | ADIOS2 backend         |
| adios-sst      | .sst   | ADIOS2 SST streaming   |
| hdf5           | .h5    | HDF5 backend           |


## Ray Records

Ray records are not part of the current required HASE backend contract. If rays
are stored by future tools, they should use particle-like openPMD records for
position, direction, weight, wavelength, and path length with normal openPMD
unit metadata.
