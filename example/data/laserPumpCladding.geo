SetFactory("OpenCASCADE");

// The gain volume is the 0.03 m radius, 0.007 m long laser-pump cylinder.
// Keep the ten z planes used by the SSG reference: one input disk plus nine
// extrusion layers. Gmsh tetrahedralizes every layer while preserving these
// planes and the physical boundary names below.
Disk(1) = {0, 0, 0, 0.03, 0.03};
extrusion[] = Extrude {0, 0, 0.007} {
  Surface{1};
  Layers{9};
};

Physical Volume("gain_medium", 1) = {extrusion[1]};
Physical Surface("ase_bottom", 1) = {1};
Physical Surface("ase_top", 2) = {extrusion[0]};
Physical Surface("cladding", 3) = {extrusion[2]};

Mesh.CharacteristicLengthMin = 0.0025;
Mesh.CharacteristicLengthMax = 0.0025;
