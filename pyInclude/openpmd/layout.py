from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .FieldSpec import FieldSpec


DIMENSIONLESS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
LENGTH = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
AREA = (2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
CROSS_SECTION = AREA
TIME = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
INV_LENGTH = (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
INV_VOLUME = (-3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
PHOTON_FLUX = (-2.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0)
RATE = (0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0)
BACKEND_FLAT = "backendFlat"


@dataclass(frozen=True)
class BackendFlatArray:
    values: object


def backendFlat(values):
    return BackendFlatArray(values)



def _shapePoint(context):
    return (context.numberOfPoints,)


def _shapeCell(context):
    return (context.numberOfTriangles,)


def _shapeCellSide(context):
    return (context.numberOfTriangles, 3)


def _shapeCellLayer(context):
    return (context.numberOfTriangles, context.numberOfLevels - 1)


def _shapePointLevel(context):
    return (context.numberOfPoints, context.numberOfLevels)


def _shapeInterface(context):
    return (4,)


def _shapeSurface(context):
    return (context.numberOfTriangles, 2)


def _shapeWavelength(context):
    return (context.spectral,)



_SHAPES_BY_AXES = {
    ("point",): _shapePoint,
    ("cell",): _shapeCell,
    ("cell", "local_vertex"): _shapeCellSide,
    ("cell", "local_side"): _shapeCellSide,
    ("cell", "layer"): _shapeCellLayer,
    ("point", "level"): _shapePointLevel,
    ("interface",): _shapeInterface,
    ("cell", "interface"): _shapeSurface,
    ("wavelength",): _shapeWavelength,
}


@dataclass(frozen=True)
class PrimitiveFieldSpec:
    name: str
    recordName: str
    dtype: object
    axes: tuple[str, ...] | None = None
    shape: object | None = None
    unit: str = "1"
    unitSI: float = 1.0
    unitDimension: tuple[float, float, float, float, float, float, float] = DIMENSIONLESS
    dynamic: bool = False
    backendRequired: bool = True
    userDefined: bool = False
    schemaRole: str = "input"

    def toFieldSpec(self, primitiveAxes: tuple[str, ...]) -> FieldSpec:
        axes = primitiveAxes if self.axes is None else tuple(self.axes)
        shape = self.shape if self.shape is not None else _SHAPES_BY_AXES[axes]
        return FieldSpec(
            self.name,
            self.recordName,
            axes,
            self.dtype,
            shape,
            unit=self.unit,
            unitSI=self.unitSI,
            unitDimension=self.unitDimension,
            dynamic=self.dynamic,
            backendRequired=self.backendRequired,
            userDefined=self.userDefined,
            schemaRole=self.schemaRole,
        )


@dataclass(frozen=True)
class PrimitiveSchema:
    name: str
    axes: tuple[str, ...]
    fields: tuple[PrimitiveFieldSpec, ...]

    def fieldSpecs(self) -> tuple[FieldSpec, ...]:
        return tuple(field.toFieldSpec(self.axes) for field in self.fields)

    def extend(self, *fields: PrimitiveFieldSpec) -> "PrimitiveSchema":
        return PrimitiveSchema(self.name, self.axes, self.fields + tuple(fields))


@dataclass(frozen=True)
class ExtensionAttributeSpec:
    name: str
    attribute: str
    dtype: str
    unit: str = "1"
    unitSI: float = 1.0
    unitDimension: tuple[float, float, float, float, float, float, float] = DIMENSIONLESS

    def cast(self, value):
        if self.dtype == "bool":
            return bool(value)
        if self.dtype == "int":
            return int(value)
        if self.dtype == "float":
            return float(value)
        if self.dtype == "str":
            return str(value)
        raise ValueError(f"unknown EXT_HASE attribute dtype '{self.dtype}'")


class PrimitiveSchemaDefinition:
    primitiveName: str | None = None
    axes: tuple[str, ...] = ()

    @classmethod
    def declaredFields(cls) -> tuple[PrimitiveFieldSpec, ...]:
        fields = []
        seen = set()
        for schema_cls in reversed(cls.mro()):
            for value in schema_cls.__dict__.values():
                if isinstance(value, PrimitiveFieldSpec) and value.name not in seen:
                    fields.append(value)
                    seen.add(value.name)
        return tuple(fields)

    @classmethod
    def primitiveSchema(cls) -> PrimitiveSchema:
        name = cls.primitiveName or cls.__name__
        return PrimitiveSchema(name, tuple(cls.axes), cls.declaredFields())

    @classmethod
    def fieldSpecs(cls) -> tuple[FieldSpec, ...]:
        return cls.primitiveSchema().fieldSpecs()


_DTYPE_BY_NAME = {
    "float64": np.float64,
    "float32": np.float32,
    "uint32": np.uint32,
    "int32": np.int32,
    "float": float,
    "int": int,
    "str": str,
    "bool": bool,
}
_UNIT_DIMENSION_BY_NAME = {
    "DIMENSIONLESS": DIMENSIONLESS,
    "LENGTH": LENGTH,
    "AREA": AREA,
    "CROSS_SECTION": CROSS_SECTION,
    "TIME": TIME,
    "INV_LENGTH": INV_LENGTH,
    "INV_VOLUME": INV_VOLUME,
    "PHOTON_FLUX": PHOTON_FLUX,
    "RATE": RATE,
}


def _schemaDocPath():
    return Path(__file__).resolve().parents[2] / "docs" / "EXT_HASE.md"


def _parseBool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def _parseAxes(value):
    value = str(value).strip()
    if not value:
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _shapeFromName(name):
    name = str(name).strip()
    if not name:
        return None
    if name == "coordinate_point":
        return lambda context: (2, context.numberOfPoints)
    raise ValueError(f"unknown EXT_HASE shape alias '{name}'")


def _parseMarkdownTables(path):
    rows_by_heading = {}
    current = None
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith("## "):
            current = line[3:].strip()
            index += 1
            continue
        if current and line.startswith("|"):
            table = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table.append(lines[index].strip())
                index += 1
            if len(table) >= 2:
                headers = [cell.strip() for cell in table[0].strip("|").split("|")]
                for row_line in table[2:]:
                    cells = [cell.strip() for cell in row_line.strip("|").split("|")]
                    if len(cells) == len(headers):
                        rows_by_heading.setdefault(current, []).append(dict(zip(headers, cells)))
            continue
        index += 1
    return rows_by_heading


def _fieldFromRow(row):
    return PrimitiveFieldSpec(
        name=row["field"],
        recordName=row["record"],
        dtype=_DTYPE_BY_NAME[row["dtype"]],
        axes=_parseAxes(row.get("axes", "")),
        shape=_shapeFromName(row.get("shape", "")),
        unit=row.get("unit", "1") or "1",
        unitSI=float(row.get("unitSI", "1.0") or 1.0),
        unitDimension=_UNIT_DIMENSION_BY_NAME[row.get("unitDimension", "DIMENSIONLESS") or "DIMENSIONLESS"],
        dynamic=_parseBool(row.get("dynamic", "false")),
        backendRequired=_parseBool(row.get("backendRequired", "true")),
        userDefined=_parseBool(row.get("userDefined", "false")),
        schemaRole=row.get("schemaRole", "input") or "input",
    )


def _loadHaseExtension(path=None):
    tables = _parseMarkdownTables(_schemaDocPath() if path is None else Path(path))
    root_rows = tables.get("Root Attributes", [])
    attribute_rows = tables.get("Simulation Attributes", [])
    primitive_rows = tables.get("Primitive Schemas", [])
    field_rows = tables.get("Mesh Records", tables.get("Primitive Fields", []))
    component_rows = tables.get("Component Fields", [])

    root_attributes = {row["attribute"]: row["value"] for row in root_rows}
    attribute_specs = tuple(
        ExtensionAttributeSpec(
            row["field"],
            row["attribute"],
            row["dtype"],
            unit=row.get("unit", "1") or "1",
            unitSI=float(row.get("unitSI", "1.0") or 1.0),
            unitDimension=_UNIT_DIMENSION_BY_NAME[row.get("unitDimension", "DIMENSIONLESS") or "DIMENSIONLESS"],
        )
        for row in attribute_rows
    )

    fields_by_primitive = {row["primitive"]: [] for row in primitive_rows}
    for row in field_rows:
        fields_by_primitive.setdefault(row["primitive"], []).append(_fieldFromRow(row))

    schema_classes = {}
    schemas = {}
    for row in primitive_rows:
        primitive = row["primitive"]
        class_name = row["class"]
        fields = tuple(fields_by_primitive.get(primitive, ()))
        attrs = {
            "primitiveName": primitive,
            "axes": _parseAxes(row["axes"]),
        }
        attrs.update({field.name: field for field in fields})
        schema_class = type(class_name, (PrimitiveSchemaDefinition,), attrs)
        schema_classes[primitive] = schema_class
        schemas[primitive] = schema_class.primitiveSchema()

    component_specs = tuple(_fieldFromRow(row).toFieldSpec(_parseAxes(row["axes"])) for row in component_rows)
    return root_attributes, attribute_specs, schema_classes, schemas, component_specs


haseExtensionAttributes, simulationAttributeSpecs, primitiveSchemaClasses, primitiveSchemas, componentFieldSpecs = _loadHaseExtension()
HASE_SCHEMA_VERSION = haseExtensionAttributes.get("haseVersion", "0.1")
globals().update({schema_class.__name__: schema_class for schema_class in primitiveSchemaClasses.values()})
PointSchema = primitiveSchemaClasses["point"]
TriangleSchema = primitiveSchemaClasses["triangle"]
PrismSchema = primitiveSchemaClasses["prism"]


def _fieldSpecsFromPrimitiveSchemas(schemas=primitiveSchemas):
    fields = {
        spec.name: spec
        for schema in schemas.values()
        for spec in schema.fieldSpecs()
    }
    fields.update({spec.name: spec for spec in componentFieldSpecs})
    return fields


schemaFields = _fieldSpecsFromPrimitiveSchemas()


def primitiveSchema(name: str) -> PrimitiveSchema:
    return primitiveSchemas[name]


def primitiveFieldSpecs(name: str) -> tuple[FieldSpec, ...]:
    return primitiveSchema(name).fieldSpecs()


def simulationAttributeSpec(name: str) -> ExtensionAttributeSpec:
    for spec in simulationAttributeSpecs:
        if spec.name == name:
            return spec
    raise KeyError(name)


def fieldSpec(name: str) -> FieldSpec:
    return schemaFields[name]


def resultFieldSpecs():
    return tuple(spec for spec in schemaFields.values() if spec.schemaRole == "result")


def spectralContext(values):
    return SimpleNamespace(spectral=np.asarray(values).size)


def flatEntityLabel(spec: FieldSpec) -> str:
    return "flatIndex"


def _isBackendFlat(values, layoutOrder):
    return isinstance(values, BackendFlatArray) or layoutOrder == BACKEND_FLAT


def _unwrap(values):
    if isinstance(values, BackendFlatArray):
        return values.values
    return values


def backendFlatArray(values, spec: FieldSpec, context, *, layoutOrder=None):
    expectedShape = spec.expectedShape(context)
    expectedSize = int(np.prod(expectedShape, dtype=np.int64))
    arr = np.asarray(_unwrap(values), dtype=spec.dtypeObject)
    if arr.size != expectedSize:
        raise ValueError(
            f"{spec.name} expects {expectedSize} values for entity {spec.entity} "
            f"with primitive shape {expectedShape}, got shape {arr.shape}"
        )

    if _isBackendFlat(values, layoutOrder):
        if arr.ndim != 1:
            raise ValueError(
                f"{spec.name} marked backend-flat must be a 1-D array with "
                f"{expectedSize} values for entity {spec.entity}, got shape {arr.shape}"
            )
        return arr

    if arr.shape == expectedShape:
        return arr.reshape(-1, order="F")

    if arr.ndim == 1:
        raise ValueError(
            f"{spec.name} got ambiguous flat array with {arr.size} values for entity "
            f"{spec.entity}; pass backendFlat(values) or layoutOrder='backendFlat' "
            f"to declare canonical backend-flat order, or pass primitive shape "
            f"{expectedShape}"
        )

    raise ValueError(
        f"{spec.name} expects primitive shape {expectedShape} for entity "
        f"{spec.entity}, got shape {arr.shape}"
    )


def primitiveArray(values, spec: FieldSpec, context, *, layoutOrder=None):
    arr = backendFlatArray(values, spec, context, layoutOrder=layoutOrder)
    return arr.reshape(spec.expectedShape(context), order="F")


def primitiveView(values, spec: FieldSpec, context, *, layoutOrder=None):
    return primitiveArray(values, spec, context, layoutOrder=layoutOrder)
