# TERRA diamond sphere fixture

`terraDiamondSphere.vtk` is the fixed 40,745-cell Tet4 sphere used to test ray
traversal across multiple domains. Its model was inspired by
[TERRA-NG](https://github.com/mantleconvection/TERRA-NG), a mantle convection
code whose spherical grid uses an icosahedron and pairs its twenty
faces into ten radial diamond domains. The sphere has a radius of 0.1 m and
reproduces that diamond topology for HASEonGPU's analytical ASE regression.

The legacy VTK file contains two identical unsigned cell scalars:

- `diamondId` paints the ten domains with IDs 0 through 9 for inspection in
  ParaView.
- `cellDomains` lets `VolumeTopology.fromFile` reconstruct the same domain
  assignment for the regression.

The regression reads the committed VTK directly. It does not import this
directory's generator and does not require Gmsh at test time.

## Reproduction

The fixture was generated with:

```bash
python tests/data/analyticalSphere/generateTerraDiamondSphere.py
```

Generation provenance:

- HASEonGPU source revision:
  `6738129f8c2ac9e7002dcb39c8f5f8766b467c28`
- TERRA-NG topology source revision:
  `b8e71f2b323da241649eaa93ec31d215b36bb1f7`
- Gmsh Python package: 4.15.2
- radius: 0.1 m
- mesh-size divisor: 12.5
- points: 7,796
- Tet4 cells: 40,745
- cells per diamond: 4,005, 4,037, 4,205, 4,108, 4,151, 4,072,
  4,067, 4,064, 4,061, 3,975
- generator SHA-256:
  `0d53791da5c21f6e949bfcd1ca23be342e7cae5797b0414f4571a2ad55df78da`
- VTK SHA-256:
  `d593e61096d4ecdcdad598f08346b3ba2bd409674e84c4a04324184ad3bf3dc5`

The generator reproduces TERRA-NG's twelve unit-icosahedron vertices and exact
`d_node[10][4]` pairing. It builds the twenty outward face normals, assigns each
noncentral Tet4 by the largest face-normal dot product of its normalized cell
center, maps the selected face to its diamond, and assigns the exact central
tetrahedron to diamond zero.

To inspect the partition, open `terraDiamondSphere.vtk` in ParaView, apply the
dataset, and color the cells by `diamondId`.
