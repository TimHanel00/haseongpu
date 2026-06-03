# openPMD Design Contract And Run Plan

## Short Description

This work is about making the HASE openPMD transport feel like a domain-level
interface instead of a storage-format interface. Users should work with prisms,
points, triangular faces, domains, spectra, and additional custom fields. The
implementation should quietly keep those values in the canonical layouts needed
by the backend and by openPMD serialization.

The current openPMD wiring already separates Python frontend objects from the C++
simulation binary, but it still exposes signs of an older layout design. In
particular, reshape and transpose decisions are embedded directly in the writer,
and the tests mostly check that records exist instead of proving that exact
array values, shapes, dtypes, and ordering survive the transport.

Static, non-changing data should still be transported through openPMD. Static
model data and dynamic time-series data may use different schema conventions,
but that distinction should not require user involvement at first. Later, the
API can allow advanced users to mark fields as static or dynamic explicitly.

## Design Contract

- Users define and inspect HASE primitives: prisms, points, triangular faces,
  domains, wavelengths, spectra, and custom fields.
- Users may define primitives and primitive fields beyond what the topology or
  gain medium requires.
- Required physical fields that are not explicitly defined, such as
  `betaCells`, should be inferred from the gain medium when the backend can
  safely determine their values and shape.
- Topology and gain-medium fields remain inspectable, including advanced static
  fields such as connectivity.
- Users should be able to access primitive data as array-of-struct style views,
  for example `getPrisms`, `getPoints`, and `getTriangles`, while the internal
  data remains stored in struct-of-arrays form.
- Users should be able to find primitives by index or coordinates. Coordinate
  lookup should map efficiently to the pitched flat backend layout.
- Domains are user-facing primitive groups. Internally, assigning a primitive to
  a domain means assigning a domain identifier field to that primitive.
- Internal arrays are canonicalized once into the layout expected by the backend
  and by the openPMD transport.
- openPMD is the transport/storage layer, not the user-facing modeling API.
- Backend-required fields and user-defined fields share the same validation and
  metadata path.
- Static and dynamic data are both transported through openPMD, but their schema
  placement can differ.
- Unknown user-defined fields should be written to openPMD, with a Python-side
  warning if the field is not consumed by the backend.
- Fields are static by default unless marked dynamic. Dynamic fields are written
  for every parse/time-series step.
- Field definitions should include metadata such as unit, name, dtype, entity,
  backend usage, and dynamic/static behavior.

## Run Plan

### 1. Lock The Contract

Step 1 is closed by the following contract:

- The primary user-facing primitives are prisms, points, and triangular faces.
- Domains are first-class user-facing groupings of primitives, represented
  internally as domain identifier fields on prisms, points, or triangles.
- The user can define additional fields for these primitives, including fields
  that are not described by the initial topology or gain medium.
- The gain medium assigns required physical fields to primitives. If a required
  backend field is missing and can be inferred from the gain medium, the backend
  path should assume the value and shape instead of forcing the user to provide
  low-level arrays.
- Topology can come from multiple input formats. If an input format such as VTK
  already contains custom fields with one value per matching primitive, those
  fields should be parsed and become usable by the same field system.
- Topology fields are not hidden. Connectivity and other advanced static fields
  are inspectable on both topology and gain-medium objects.
- User access should feel like array-of-struct primitive access, for example
  `medium.getPrisms`, `medium.getPoints`, `medium.getTriangles`, and direct
  lookup by index or coordinates. Internally these views reflect struct-of-arrays
  storage with backend-compatible pitches.
- A primitive view should support reflecting a field definition onto existing
  primitives, so a user can extend prism/point/triangle data while preserving
  original coordinates, indices, and required physical fields.
- Unknown custom fields are written to openPMD. The Python frontend should warn
  when such a field is unused by the backend.
- Fields are static unless marked dynamic. Dynamic fields are emitted at every
  parse/time-series step.
- Every field definition carries at least name, entity, dtype, unit metadata,
  backend usage metadata, and static/dynamic behavior.

### 2. Define Canonical Entities

Introduce explicit entity names and shapes:

- `point`
- `cell`
- `prism`
- `layer`
- `point_level`
- `cell_layer`
- `interface`
- `wavelength`
- `local_vertex`
- `local_side`

Each field should have metadata: name, entity binding, dtype, unit metadata,
static/dynamic behavior, and whether the backend requires it.

### 3. Centralize Layout Conversion

Move all ordering, reshape, and transpose decisions out of the writer and into
one layout layer.

The layout layer should:

- validate expected shape and dtype,
- convert domain-shaped arrays to canonical backend/openPMD order,
- provide inverse conversion for tests and readers,
- reject ambiguous flat arrays unless they carry explicit entity metadata.

### 4. Fix The Test Contract

Replace record-existence-only checks with exact assertions.

Tests should write a small asymmetric mesh and verify:

- exact values,
- exact shape,
- exact dtype,
- axis labels,
- backend-compatible flattened order.

The test suite must fail on accidental transpose or ordering regressions.

### 5. Split Static And Dynamic Schema

Keep static values in openPMD, but separate their role from time-series data.

Static examples:

- topology,
- connectivity,
- geometry,
- material constants,
- spectral tables,
- backend-independent simulation setup.

Dynamic examples:

- `betaCells`,
- `betaVolume`,
- `phiAse`,
- `mse`,
- `totalRays`,
- `dndtAse`,
- future pump or time-step fields.

Initial behavior: fields are static unless the schema marks them dynamic.

### 6. Choose Storage Conventions

Use standard openPMD records where the data maps cleanly to openPMD concepts.
Use a HASE-specific schema namespace for topology and control data that do not
fit the openPMD standard cleanly.

Every stored field should carry enough metadata to explain:

- entity,
- axis labels,
- layout order,
- static/dynamic role,
- unit metadata,
- HASE schema version,
- backend-required status.

### 7. Add User-Defined Fields

Expose a domain-level API for custom fields, for example:

```python
gainMedium.defineField(
    "temperature",
    entity=("cell", "layer"),
    values=values,
    unit="K",
    dynamic=False,
)
```

Rules:

- field names are validated,
- entity determines expected shape,
- static/dynamic determines schema placement,
- backend ignores unknown fields unless explicitly configured to consume them,
- openPMD metadata preserves meaning for inspection and external tooling.

### 8. Fix Transport Flow

Concrete bugfixes:

- fix array ordering through the canonical layout layer,
- validate C++ parser extents before constructing `HostMesh`,
- preserve backend defaults instead of serializing Python `None` as `"None"`,
- preserve or explicitly reject unsupported compute fields such as VTK output,
- add SST timeout, child-process cleanup, and stderr reporting,
- fix MPI collectivity around openPMD output writes.

### 9. Migrate Incrementally

Recommended sequence:

1. Add schema and field specifications without changing behavior.
2. Rewrite tests to assert exact arrays.
3. Move reshape and transpose logic into layout helpers.
4. Update Python writer and C++ parser to use the canonical layout.
5. Add static/dynamic schema separation.
6. Add user-defined fields.
7. Add multi-iteration dynamic streaming later.

### 10. Acceptance Criteria

The design is ready when:

- users can define fields by entity, not by flat array order,
- backend-required arrays are stored in exactly the expected order,
- static and dynamic data are both carried via openPMD without user involvement,
- tests catch transpose and layout regressions,
- unknown user fields survive transport and remain inspectable,
- parser errors identify the broken field, expected shape, actual shape, and
  entity.
