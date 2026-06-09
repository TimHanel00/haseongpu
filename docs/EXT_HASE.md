# EXT_HASE: HASE openPMD Extension and Transport Schema

haseOpenPMDextension = HASE
haseVersion = 0.1
simulationType = laserCrystalASE
geometryType = extrudedTriangularPrism

The built-in HASE schema is defined in Python in `pyInclude/openpmd/schema/`.
This document describes that built-in design, its units, and how Python frontend
primitive fields map onto the current openPMD transport consumed by the C++
backend. It is documentation of the Python schema source, not an input parsed by
the Python frontend.

HASE does not write the openPMD-reserved `openPMDextension` attribute. The
current HASE-owned marker is `haseOpenPMDextension = HASE`. If HASE later
registers an openPMD extension identifier, the writer can add that reserved
extension marker without changing the record inventory below.

## Layer Boundaries

### Python Interface

The Python interface is primitive-oriented. User-facing HASE primitives are
`points`, `triangles`, and `prisms`; each primitive schema is explicitly
extensible through normal Python inheritance from `BaseSchema`-derived classes
such as `PointSchema`, `TriangleSchema`, and `PrismSchema`. Users add scalar or
vector fields to primitive schemas and address those fields by domain names in
Python.

The built-in point schema is shaped by its multidimensional `position` field.
The backend still serializes this field through the historical `vertices/{x,y}`
record components; the legacy field lookup name `points` is kept as a
compatibility alias for the same schema field.

The Python layer owns primitive classes, domains, gain-medium fields, spectra,
and solver settings. It must not expose backend adapter objects or require users
to construct the flattened transport contract.

### Transport Layer

The transport layer maps the Python domain model to openPMD records consumed by
the current C++ backend. Each additional primitive field maps to an additional
openPMD mesh record unless it is represented as a component of an existing
record, such as `vertices/{x,y}`.

Transport details such as `core_`, `backendFlat`, and the current iteration
attributes are serialization details. They exist because the backend consumes
them; they are not user-facing Python schema names.

### openPMD Standard

openPMD provides the storage convention. Records live in standard locations such
as an iteration's `meshes` group, and mesh records carry openPMD metadata such
as `unitSI`, `unitDimension`, `axisLabels`, `gridSpacing`, `gridGlobalOffset`,
record components, and data order.

HASE adds domain metadata on top of openPMD mesh records. That metadata explains
what a record means for this simulation, for example which primitive axes it
came from and whether the backend requires it. HASE metadata must not replace
required openPMD metadata.

All rows under `Mesh Records` and `Component Fields` are openPMD mesh records
written below the iteration's `meshes` group. The `unit` column is the
human-readable HASE unit string. `unitSI` and `unitDimension` are the
corresponding openPMD record metadata. A unit of `1` is an explicit
dimensionless unit, not a missing unit.

## Naming Conventions

- Extension document filename: `EXT_HASE.md`, following the openPMD extension
  document style.
- Built-in primitive schema classes: `PointSchema`, `TriangleSchema`, and
  `PrismSchema` in `PascalCase`.
- Python user-facing field names: `lowerCamelCase`, for example `betaVolume`,
  `pointBeta`, `sigmaAbsorption`, and `dndtAse`.
- Python primitive names: plural nouns for user collections (`points`,
  `triangles`, `prisms`) and singular names in schema tables (`point`,
  `triangle`, `prism`).
- Transport record names: `snake_case`, for example `beta_volume`,
  `sigma_absorption`, and `dndt_ase`.
- Current backend record prefix: `core_`; result records use `core_result_`.
- Current backend iteration attributes: `snake_case`, for example
  `number_of_points`, `max_sigma_emission`, and `min_sample_range`.
- HASE metadata attributes: `hase` prefix plus `PascalCase` suffix, for example
  `haseSchemaVersion`, `hasePrimitiveShape`, and `haseBackendRequired`.
- HASE-owned root attributes: `hase` prefix, for example
  `haseOpenPMDextension`, `haseVersion`, `simulationType`, and `geometryType`.
- Reserved openPMD names are not reused for HASE-owned strings.

## HASE Record Attributes

HASE mesh records use these extension attributes in addition to normal openPMD
mesh metadata:

- `haseSchemaVersion`: HASE transport schema version.
- `haseEntity`: semantic entity binding, such as `cell_layer`.
- `haseAxes`: semantic HASE axes, such as `["cell", "layer"]`.
- `haseLayoutOrder`: transport layout convention, such as `backendFlat` or
  `recordC`.
- `hasePrimitiveShape`: semantic primitive shape before backend flattening.
- `haseStatic`: true for static model/topology fields.
- `haseDynamic`: true for fields that may vary between iterations.
- `haseBackendRequired`: true if the C++ backend currently requires the field.
- `haseUnit`: human-readable HASE unit string.
- `haseUserDefined`: true for user-defined primitive fields.
- `haseUserFieldName`: original user field name, present for user-defined
  fields.

`axisLabels` remains the openPMD mesh-record axis label list. Flattened backend
records therefore use `axisLabels = ["flatIndex"]`; their HASE primitive axes
are stored separately in `haseAxes`.

## Root Attributes

| field                | attribute            | value                    |
|----------------------|----------------------|--------------------------|
| haseOpenPMDextension | haseOpenPMDextension | HASE                     |
| haseVersion          | haseVersion          | 0.1                      |
| simulationType       | simulationType       | laserCrystalASE          |
| geometryType         | geometryType         | extrudedTriangularPrism  |

## Simulation Attributes

These values are serialized as iteration attributes for the current backend contract. Physical attributes still declare units here; a later backend revision can promote them to scalar mesh records without changing their domain names.

| field              | attribute             | dtype | unit   | unitSI | unitDimension |
|--------------------|-----------------------|-------|--------|--------|---------------|
| numberOfPoints     | number_of_points      | int   | count  | 1.0    | DIMENSIONLESS |
| numberOfTriangles  | number_of_cells       | int   | count  | 1.0    | DIMENSIONLESS |
| numberOfLevels     | number_of_levels      | int   | count  | 1.0    | DIMENSIONLESS |
| thickness          | thickness             | float | m      | 1.0    | LENGTH        |
| nTot               | n_tot                 | float | cm^-3  | 1.0e6  | INV_VOLUME    |
| crystalTFluo       | crystal_t_fluo        | float | s      | 1.0    | TIME          |
| claddingNumber     | cladding_number       | int   | count  | 1.0    | DIMENSIONLESS |
| claddingAbsorption | cladding_absorption   | float | cm^-1  | 100.0  | INV_LENGTH    |
| minRaysPerSample   | min_rays_per_sample   | int   | count  | 1.0    | DIMENSIONLESS |
| maxRaysPerSample   | max_rays_per_sample   | int   | count  | 1.0    | DIMENSIONLESS |
| mseThreshold       | mse_threshold         | float | 1      | 1.0    | DIMENSIONLESS |
| repetitions        | repetitions           | int   | count  | 1.0    | DIMENSIONLESS |
| adaptiveSteps      | adaptive_steps        | int   | count  | 1.0    | DIMENSIONLESS |
| useReflections     | use_reflections       | bool  | 1      | 1.0    | DIMENSIONLESS |
| spectralResolution | spectral_resolution   | int   | count  | 1.0    | DIMENSIONLESS |
| monochromatic      | monochromatic         | bool  | 1      | 1.0    | DIMENSIONLESS |
| maxSigmaAbsorption | max_sigma_absorption  | float | cm^2   | 1.0e-4 | CROSS_SECTION |
| maxSigmaEmission   | max_sigma_emission    | float | cm^2   | 1.0e-4 | CROSS_SECTION |
| backend            | backend               | str   | 1      | 1.0    | DIMENSIONLESS |
| maxGpus            | max_gpus              | int   | count  | 1.0    | DIMENSIONLESS |
| parallelMode       | parallel_mode         | str   | 1      | 1.0    | DIMENSIONLESS |
| minSampleRange     | min_sample_range      | int   | index  | 1.0    | DIMENSIONLESS |
| maxSampleRange     | max_sample_range      | int   | index  | 1.0    | DIMENSIONLESS |
| rngSeed            | rng_seed              | int   | 1      | 1.0    | DIMENSIONLESS |

## Primitive Schemas

| primitive | class          | axes       | description                                          |
|-----------|----------------|------------|------------------------------------------------------|
| point     | PointSchema    | point      | 2D topology point and point-level fields             |
| triangle  | TriangleSchema | cell       | Triangular face/cell fields in the extruded topology |
| prism     | PrismSchema    | cell,layer | Extruded triangular prism volume fields              |

## Mesh Records

| primitive | field            | record             | dtype   | axes              | shape            | unit | unitSI | unitDimension | dynamic | backendRequired | userDefined | schemaRole |
|-----------|------------------|--------------------|---------|-------------------|------------------|------|--------|---------------|---------|-----------------|-------------|------------|
| point     | position         | vertices           | float64 | coordinate,point  | coordinate_point | m    | 1.0    | LENGTH        | false   | true            | false       | input      |
| point     | pointBeta        | point_beta         | float64 | point,level       |                  | 1    | 1.0    | DIMENSIONLESS | true    | true            | false       | input      |
| point     | phiAse           | phi_ase            | float32 | point,level       |                  | cm^-2 s^-1 | 1.0e4 | PHOTON_FLUX   | true    | false           | false       | result     |
| point     | mse              | mse                | float64 | point,level       |                  | 1    | 1.0    | DIMENSIONLESS | true    | false           | false       | result     |
| point     | totalRays        | total_rays         | uint32  | point,level       |                  | count | 1.0    | DIMENSIONLESS | true    | false           | false       | result     |
| point     | dndtAse          | dndt_ase           | float64 | point,level       |                  | s^-1 | 1.0    | RATE          | true    | false           | false       | result     |
| triangle  | connectivity     | connectivity       | uint32  | cell,local_vertex |                  | 1    | 1.0    | DIMENSIONLESS | false   | true            | false       | input      |
| triangle  | neighbors        | neighbors          | int32   | cell,local_side   |                  | 1    | 1.0    | DIMENSIONLESS | false   | true            | false       | input      |
| triangle  | forbiddenEdges   | forbidden_edges    | int32   | cell,local_side   |                  | 1    | 1.0    | DIMENSIONLESS | false   | true            | false       | input      |
| triangle  | normalPoints     | normal_points      | uint32  | cell,local_side   |                  | 1    | 1.0    | DIMENSIONLESS | false   | true            | false       | input      |
| triangle  | cellCenterX      | cell_center_x      | float64 | cell              |                  | m    | 1.0    | LENGTH        | false   | true            | false       | input      |
| triangle  | cellCenterY      | cell_center_y      | float64 | cell              |                  | m    | 1.0    | LENGTH        | false   | true            | false       | input      |
| triangle  | cellNormalX      | cell_normal_x      | float64 | cell,local_side   |                  | 1    | 1.0    | DIMENSIONLESS | false   | true            | false       | input      |
| triangle  | cellNormalY      | cell_normal_y      | float64 | cell,local_side   |                  | 1    | 1.0    | DIMENSIONLESS | false   | true            | false       | input      |
| triangle  | surface          | surface            | float32 | cell              |                  | m^2  | 1.0    | AREA          | false   | true            | false       | input      |
| triangle  | claddingCellType | cladding_cell_type | uint32  | cell              |                  | 1    | 1.0    | DIMENSIONLESS | false   | true            | false       | input      |
| triangle  | refractiveIndex  | refractive_index   | float32 | interface         |                  | 1    | 1.0    | DIMENSIONLESS | false   | true            | false       | input      |
| triangle  | reflectivity     | reflectivity       | float32 | cell,interface    |                  | 1    | 1.0    | DIMENSIONLESS | false   | true            | false       | input      |
| prism     | betaVolume       | beta_volume        | float64 | cell,layer        |                  | 1    | 1.0    | DIMENSIONLESS | true    | true            | false       | input      |

## Component Fields

| component     | field            | record            | dtype   | axes       | shape | unit | unitSI | unitDimension | dynamic | backendRequired | userDefined | schemaRole |
|---------------|------------------|-------------------|---------|------------|-------|------|--------|---------------|---------|-----------------|-------------|------------|
| crossSections | lambdaAbsorption | lambda_absorption | float64 | wavelength |       | m    | 1.0    | LENGTH        | false   | false           | false       | input      |
| crossSections | lambdaEmission   | lambda_emission   | float64 | wavelength |       | m    | 1.0    | LENGTH        | false   | false           | false       | input      |
| crossSections | sigmaAbsorption  | sigma_absorption  | float64 | wavelength |       | cm^2 | 1.0e-4 | CROSS_SECTION | false   | false           | false       | input      |
| crossSections | sigmaEmission    | sigma_emission    | float64 | wavelength |       | cm^2 | 1.0e-4 | CROSS_SECTION | false   | false           | false       | input      |

## Optional Mesh Records

Optional domain records include `gain`, `pumpIntensity`, `dopingDensity`,
`tauRadiative`, and user-defined primitive fields. Existing backend-required
records such as `refractiveIndex` and `reflectivity` are listed in `Mesh Records` because the current ray backend consumes them.

## Ray Records

If rays are stored, use particle-like records position, direction, weight,
wavelength, and pathLength with normal openPMD unit metadata.
