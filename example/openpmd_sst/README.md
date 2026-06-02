# openPMD/ADIOS2-SST Prototype

This example is a first standalone producer/consumer for the intended
HASE-on-openPMD data path. It does not call the HASE simulation yet.

The Python producer writes random mesh-associated data to:

```text
/data/0/meshes/core/vertices/x
/data/0/meshes/core/vertices/y
/data/0/meshes/core/connectivity
/data/0/meshes/core/neighbors
/data/0/meshes/core/forbidden_edges
/data/0/meshes/core/normal_points
/data/0/meshes/core/cell_center/x
/data/0/meshes/core/cell_center/y
/data/0/meshes/core/cell_normal_x
/data/0/meshes/core/cell_normal_y
/data/0/meshes/core/surface
/data/0/meshes/core/beta_volume
/data/0/meshes/core/cladding_cell_type
/data/0/meshes/core/reflectivity
/data/0/meshes/core/point_beta
```

The current prototype serializes these logical groups with flat openPMD mesh
record names such as `core_vertices/x`, `core_connectivity`, and
`core_point_beta`.

Build the C++ consumer with:

```bash
cmake -S . -B build/openpmd-sst \
  -DHASE_BUILD_PhiAse=OFF
cmake --build build/openpmd-sst --target hase_openpmd_sst_consumer
```

Run the consumer and producer in separate terminals. SST is a live stream, so
both sides must overlap:

```bash
./build/openpmd-sst/example/openpmd_sst/hase_openpmd_sst_consumer hase_input.sst
```

```bash
python3 example/openpmd_sst/producer.py hase_input.sst
```

The Python environment must provide `openpmd_api` with ADIOS2 SST support.
