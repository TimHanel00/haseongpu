from __future__ import annotations

import numpy as np

from .core import (
    AREA,
    CROSS_SECTION,
    DIMENSIONLESS,
    INV_LENGTH,
    INV_VOLUME,
    LENGTH,
    PHOTON_FLUX,
    RATE,
    TIME,
    ExtensionAttributeSpec,
    PrimitiveFieldSpec,
    PrimitiveSchemaDefinition,
)


HASE_EXTENSION_ATTRIBUTES = {
    "haseOpenPMDextension": "HASE",
    "haseVersion": "0.1",
    "simulationType": "laserCrystalASE",
    "geometryType": "unstructuredTet4",
}


SIMULATION_ATTRIBUTE_SPECS = (
    ExtensionAttributeSpec("numberOfPoints", "number_of_points", "int", unit="count"),
    ExtensionAttributeSpec("numberOfTriangles", "number_of_cells", "int", unit="count"),
    ExtensionAttributeSpec("numberOfLevels", "number_of_levels", "int", unit="count"),
    ExtensionAttributeSpec("thickness", "thickness", "float", unit="m", unitDimension=LENGTH),
    ExtensionAttributeSpec("nTot", "n_tot", "float", unit="cm^-3", unitSI=1.0e6, unitDimension=INV_VOLUME),
    ExtensionAttributeSpec("crystalTFluo", "crystal_t_fluo", "float", unit="s", unitDimension=TIME),
    ExtensionAttributeSpec("claddingNumber", "cladding_number", "int", unit="count"),
    ExtensionAttributeSpec(
        "claddingAbsorption", "cladding_absorption", "float", unit="cm^-1", unitSI=100.0, unitDimension=INV_LENGTH
    ),
    ExtensionAttributeSpec("minRaysPerSample", "min_rays_per_sample", "int", unit="count"),
    ExtensionAttributeSpec("maxRaysPerSample", "max_rays_per_sample", "int", unit="count"),
    ExtensionAttributeSpec("mseThreshold", "mse_threshold", "float"),
    ExtensionAttributeSpec("repetitions", "repetitions", "int", unit="count"),
    ExtensionAttributeSpec("adaptiveSteps", "adaptive_steps", "int", unit="count"),
    ExtensionAttributeSpec("useReflections", "use_reflections", "bool"),
    ExtensionAttributeSpec("spectralResolution", "spectral_resolution", "int", unit="count"),
    ExtensionAttributeSpec("monochromatic", "monochromatic", "bool"),
    ExtensionAttributeSpec(
        "maxSigmaAbsorption", "max_sigma_absorption", "float", unit="cm^2", unitSI=1.0e-4, unitDimension=CROSS_SECTION
    ),
    ExtensionAttributeSpec(
        "maxSigmaEmission", "max_sigma_emission", "float", unit="cm^2", unitSI=1.0e-4, unitDimension=CROSS_SECTION
    ),
    ExtensionAttributeSpec("backend", "backend", "str"),
    ExtensionAttributeSpec("maxGpus", "max_gpus", "int", unit="count"),
    ExtensionAttributeSpec("parallelMode", "parallel_mode", "str"),
    ExtensionAttributeSpec("minSampleRange", "min_sample_range", "int", unit="index"),
    ExtensionAttributeSpec("maxSampleRange", "max_sample_range", "int", unit="index"),
    ExtensionAttributeSpec("rngSeed", "rng_seed", "int"),
)


class PointSchema(PrimitiveSchemaDefinition):
    primitiveName = "point"
    axes = ("point",)
    shapeField = "position"

    position = PrimitiveFieldSpec(
        "position",
        "vertices",
        np.float64,
        axes=("coordinate", "point"),
        shape=lambda context: (2, context.numberOfPoints),
        unit="m",
        unitDimension=LENGTH,
    )
    pointBeta = PrimitiveFieldSpec(
        "pointBeta", np.float64, axes=("point", "level"), dynamic=True
    )
    phiAse = PrimitiveFieldSpec(
        "phiAse",
        np.float32,
        axes=("point", "level"),
        unit="cm^-2 s^-1",
        unitSI=1.0e4,
        unitDimension=PHOTON_FLUX,
        dynamic=True,
        backendRequired=False,
        schemaRole="result",
    )
    mse = PrimitiveFieldSpec(
        "mse", np.float64, axes=("point", "level"), dynamic=True, backendRequired=False, schemaRole="result"
    )
    totalRays = PrimitiveFieldSpec(
        "totalRays",
        np.uint32,
        axes=("point", "level"),
        unit="count",
        dynamic=True,
        backendRequired=False,
        schemaRole="result",
    )
    dndtAse = PrimitiveFieldSpec(
        "dndtAse",
        np.float64,
        axes=("point", "level"),
        unit="s^-1",
        unitDimension=RATE,
        dynamic=True,
        backendRequired=False,
        schemaRole="result",
    )


class TriangleSchema(PrimitiveSchemaDefinition):
    primitiveName = "triangle"
    axes = ("cell",)
    shapeField = "connectivity"
    connectivity = PrimitiveFieldSpec("connectivity", np.uint32, axes=("cell", "local_vertex"))
    neighbors = PrimitiveFieldSpec("neighbors", np.int32, axes=("cell", "local_side"))
    forbiddenEdges = PrimitiveFieldSpec("forbiddenEdges", np.int32, axes=("cell", "local_side"))
    normalPoints = PrimitiveFieldSpec("normalPoints", np.uint32, axes=("cell", "local_side"))
    center = PrimitiveFieldSpec("center", "cell_center", np.float64, axes=("coordinate", "cell"), unit="m", unitDimension=LENGTH)
    normal = PrimitiveFieldSpec("normal", np.float64, axes=("cell", "local_side", "coordinate"))
    surface = PrimitiveFieldSpec("surface", np.float32, axes=("cell",), unit="m^2", unitDimension=AREA)
    claddingGroup = PrimitiveFieldSpec("claddingGroup", np.uint32, axes=("cell",))
    refractiveIndex = PrimitiveFieldSpec("refractiveIndex", np.float32, axes=("interface",))
    reflectivity = PrimitiveFieldSpec("reflectivity", np.float32, axes=("cell", "interface"))


class PrismSchema(PrimitiveSchemaDefinition):
    primitiveName = "prism"
    axes = ("cell", "layer")
    shapeField = "betaVolume"

    betaVolume = PrimitiveFieldSpec("betaVolume", np.float64, axes=("cell", "layer"), dynamic=True)


BACKEND_FIELD_SPECS = (
    PrimitiveFieldSpec("cellCenterX", "cell_center_x", np.float64, axes=("cell",), unit="m", unitDimension=LENGTH),
    PrimitiveFieldSpec("cellCenterY", "cell_center_y", np.float64, axes=("cell",), unit="m", unitDimension=LENGTH),
    PrimitiveFieldSpec("cellNormalX", "cell_normal_x", np.float64, axes=("cell", "local_side")),
    PrimitiveFieldSpec("cellNormalY", "cell_normal_y", np.float64, axes=("cell", "local_side")),
    PrimitiveFieldSpec("claddingCellType", "cladding_cell_type", np.uint32, axes=("cell",)),
)


COMPONENT_FIELD_SPECS = (
    PrimitiveFieldSpec(
        "lambdaAbsorption", "lambda_absorption", np.float64, axes=("wavelength",), unit="m", unitDimension=LENGTH, backendRequired=False
    ),
    PrimitiveFieldSpec(
        "lambdaEmission", "lambda_emission", np.float64, axes=("wavelength",), unit="m", unitDimension=LENGTH, backendRequired=False
    ),
    PrimitiveFieldSpec(
        "sigmaAbsorption",
        "sigma_absorption",
        np.float64,
        axes=("wavelength",),
        unit="cm^2",
        unitSI=1.0e-4,
        unitDimension=CROSS_SECTION,
        backendRequired=False,
    ),
    PrimitiveFieldSpec(
        "sigmaEmission",
        "sigma_emission",
        np.float64,
        axes=("wavelength",),
        unit="cm^2",
        unitSI=1.0e-4,
        unitDimension=CROSS_SECTION,
        backendRequired=False,
    ),
)


PRIMITIVE_SCHEMA_CLASSES = {
    "point": PointSchema,
    "triangle": TriangleSchema,
    "prism": PrismSchema,
}


FIELD_ALIASES = {
    "points": "position",
}
