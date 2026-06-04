from types import SimpleNamespace

import numpy as np
import pytest

from pyInclude.openpmd import (
    HASE_SCHEMA_VERSION,
    PointSchema,
    PrimitiveFieldSpec,
    PrismSchema,
    TriangleSchema,
    backendFlat,
    backendFlatArray,
    fieldSpec,
    haseExtensionAttributes,
    primitiveArray,
    primitiveFieldSpecs,
    primitiveSchema,
    primitiveSchemas,
    schemaFields,
    simulationAttributeSpec,
)


def _context():
    return SimpleNamespace(numberOfTriangles=2, numberOfLevels=4, numberOfPoints=5, spectral=3)


def testPrimitiveShapeConvertsToBackendFlatFirstAxisFastest():
    spec = fieldSpec("betaVolume")
    primitive = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]])

    flat = backendFlatArray(primitive, spec, _context())

    np.testing.assert_array_equal(flat, np.array([1.0, 2.0, 10.0, 20.0, 100.0, 200.0]))


def testBackendFlatConvertsBackToPrimitiveShape():
    spec = fieldSpec("reflectivity")
    flat = backendFlat(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))

    primitive = primitiveArray(flat, spec, _context())

    np.testing.assert_array_equal(primitive, np.array([[0.1, 0.3], [0.2, 0.4]], dtype=np.float32))


def testAmbiguousFlatArrayIsRejectedUnlessMarkedBackendFlat():
    spec = fieldSpec("pointBeta")
    values = np.arange(20, dtype=np.float64)

    with pytest.raises(ValueError, match="ambiguous flat array"):
        backendFlatArray(values, spec, _context())

    np.testing.assert_array_equal(backendFlatArray(backendFlat(values), spec, _context()), values)



def testExtensionDocumentDefinesPrimitiveClassesAndBackendAttributeNames():
    assert HASE_SCHEMA_VERSION == haseExtensionAttributes["haseVersion"]
    assert haseExtensionAttributes["haseOpenPMDextension"] == "HASE"
    assert set(primitiveSchemas) == {"point", "triangle", "prism"}
    assert PointSchema.primitiveSchema().name == "point"
    assert TriangleSchema.primitiveSchema().name == "triangle"
    assert PrismSchema.primitiveSchema().name == "prism"
    assert simulationAttributeSpec("maxSigmaEmission").attribute == "max_sigma_emission"
    assert simulationAttributeSpec("rngSeed").attribute == "rng_seed"
    assert fieldSpec("surface").unit == "m^2"
    assert fieldSpec("surface").unitDimension == (2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert fieldSpec("sigmaAbsorption").unit == "cm^2"
    assert fieldSpec("sigmaAbsorption").unitSI == 1.0e-4
    assert fieldSpec("phiAse").unit == "cm^-2 s^-1"
    assert fieldSpec("dndtAse").unit == "s^-1"
    assert simulationAttributeSpec("nTot").unit == "cm^-3"
    assert simulationAttributeSpec("nTot").unitDimension == (-3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert simulationAttributeSpec("claddingAbsorption").unit == "cm^-1"
    assert simulationAttributeSpec("claddingAbsorption").unitSI == 100.0


def testSchemaFieldsAreDerivedFromPrimitiveSchemas():
    triangle_names = {spec.name for spec in primitiveFieldSpecs("triangle")}
    prism_names = {spec.name for spec in primitiveFieldSpecs("prism")}
    point_names = {spec.name for spec in primitiveFieldSpecs("point")}

    assert {"connectivity", "surface", "claddingCellType"} <= triangle_names
    assert "betaVolume" in prism_names
    assert {"pointBeta", "phiAse", "mse", "totalRays", "dndtAse"} <= point_names
    assert schemaFields["betaVolume"] in primitiveFieldSpecs("prism")


def testPrimitiveSchemaCanBeExtendedWithUserFieldSpec():
    extended = primitiveSchema("prism").extend(
        PrimitiveFieldSpec("temperature", "custom_temperature", float, unit="K", backendRequired=False, userDefined=True)
    )
    spec = extended.fieldSpecs()[-1]

    assert spec.name == "temperature"
    assert spec.entity == "cell_layer"
    assert spec.expectedShape(_context()) == (2, 3)
    assert spec.unit == "K"
    assert spec.userDefined is True



def testPrimitiveSchemaCanBeDeclaredByInheritance():
    class ThermalPrism(PrismSchema):
        temperature = PrimitiveFieldSpec("temperature", "custom_temperature", np.float64, unit="K", backendRequired=False)

    fields = {spec.name: spec for spec in ThermalPrism.fieldSpecs()}

    assert {"betaVolume", "temperature"} <= set(fields)
    assert fields["temperature"].entity == "cell_layer"
    assert fields["temperature"].expectedShape(_context()) == (2, 3)
    assert fields["temperature"].unit == "K"
