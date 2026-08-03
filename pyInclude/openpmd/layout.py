from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .FieldSpec import FieldSpec
from .schema import (
    BACKEND_FIELD_SPECS,
    COMPONENT_FIELD_SPECS,
    FIELD_ALIASES,
    HASE_TRANSPORT_ATTRIBUTES,
    PRIMITIVE_SCHEMA_CLASSES,
    RESULT_ATTRIBUTE_SPECS,
    SIMULATION_ATTRIBUTE_SPECS,
    BaseGroup,
    BaseSchema,
    ExtensionAttributeSpec,
    GroupFieldSpec,
    PointSchema,
    PrimitiveFieldSpec,
    PrimitiveSchema,
    PrimitiveSchemaDefinition,
    PrismSchema,
    TriangleSchema,
    unitDimension,
)


BACKEND_FLAT = "backendFlat"


@dataclass(frozen=True)
class BackendFlatArray:
    """Explicit marker that values already use HASE's flat backend ordering."""

    values: object
    """One-dimensional values ordered according to the field's backend layout."""


def backendFlat(values):
    """Mark ``values`` as already flattened in native backend order.

    Parameters
    ----------
    values
        One-dimensional scalar sequence. Its length is validated later against
        the selected :class:`FieldSpec` and topology context.
    """
    return BackendFlatArray(values)


haseTransportAttributes = HASE_TRANSPORT_ATTRIBUTES
simulationAttributeSpecs = SIMULATION_ATTRIBUTE_SPECS
resultAttributeSpecs = RESULT_ATTRIBUTE_SPECS
primitiveSchemaClasses = PRIMITIVE_SCHEMA_CLASSES
primitiveSchemas = {name: schema_class.primitiveSchema() for name, schema_class in primitiveSchemaClasses.items()}
componentFieldSpecs = tuple(spec.toFieldSpec(tuple(spec.axes)) for spec in COMPONENT_FIELD_SPECS)
backendFieldSpecs = tuple(spec.toFieldSpec(tuple(spec.axes)) for spec in BACKEND_FIELD_SPECS)
HASE_TRANSPORT_VERSION = "0.2"
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
    fields.update({spec.name: spec for spec in backendFieldSpecs})
    for alias, canonical in FIELD_ALIASES.items():
        fields[alias] = fields[canonical]
    return fields


schemaFields = _fieldSpecsFromPrimitiveSchemas()


def primitiveSchema(name: str) -> PrimitiveSchema:
    """Return a registered primitive schema.

    Parameters
    ----------
    name
        Schema-contract key such as ``"point"`` or ``"triangle"``. Unknown
        keys raise ``KeyError``.
    """
    return primitiveSchemas[name]


def primitiveFieldSpecs(name: str) -> tuple[FieldSpec, ...]:
    """Return resolved fields for one registered primitive.

    Parameters
    ----------
    name
        Primitive schema-contract key accepted by :func:`primitiveSchema`.
    """
    return primitiveSchema(name).fieldSpecs()


def simulationAttributeSpec(name: str) -> ExtensionAttributeSpec:
    """Return one simulation run-control attribute specification.

    Parameters
    ----------
    name
        Lower-camel Python attribute name. The returned specification exposes
        the separate snake-case wire name through ``attribute``.
    """
    for spec in simulationAttributeSpecs:
        if spec.name == name:
            return spec
    raise KeyError(name)


def fieldSpec(name: str) -> FieldSpec:
    """Return one unique resolved field specification.

    Parameters
    ----------
    name
        Python field name from :data:`schemaFields`; unknown names raise
        ``KeyError``.
    """
    return schemaFields[name]


def resultFieldSpecs():
    """Return every field whose schema role is ``"result"``."""
    return tuple(spec for spec in schemaFields.values() if spec.schemaRole == "result")


def spectralContext(values):
    """Build a minimal shape context from a one-dimensional spectral grid.

    Parameters
    ----------
    values
        Any sized array-like object. Only its element count is retained as
        ``spectralSampleCount``.
    """
    return SimpleNamespace(spectralSampleCount=np.asarray(values).size)


def flatEntityLabel(spec: FieldSpec) -> str:
    """Return the fixed openPMD axis label for flattened data.

    Parameters
    ----------
    spec
        Field being labelled. HASE currently uses ``"flatIndex"`` for every
        backend-flat field irrespective of its logical primitive axes.
    """
    return "flatIndex"


def _isBackendFlat(values, layoutOrder):
    return isinstance(values, BackendFlatArray) or layoutOrder == BACKEND_FLAT


def _unwrap(values):
    if isinstance(values, BackendFlatArray):
        return values.values
    return values


def backendFlatArray(values, spec: FieldSpec, context, *, layoutOrder=None):
    """Validate and flatten one primitive array into native Fortran ordering.

    Parameters
    ----------
    values
        Primitive-shaped values, or one-dimensional values marked by
        :func:`backendFlat`.
    spec
        Field declaration defining dtype, logical axes, and expected shape.
    context
        Topology or spectral context supplying axis sizes.
    layoutOrder
        ``"backendFlat"`` explicitly declares pre-flattened native order;
        otherwise primitive-shaped input is flattened in Fortran order.
    """
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
    """Return values in their declared primitive shape.

    Parameters
    ----------
    values, spec, context, layoutOrder
        Same validation and layout controls as :func:`backendFlatArray`; the
        result is reshaped to ``spec.expectedShape(context)``.
    """
    arr = backendFlatArray(values, spec, context, layoutOrder=layoutOrder)
    return arr.reshape(spec.expectedShape(context), order="F")


def primitiveView(values, spec: FieldSpec, context, *, layoutOrder=None):
    """Return the validated primitive-shaped view used by schema consumers.

    Parameters
    ----------
    values, spec, context, layoutOrder
        Same values and layout contract as :func:`primitiveArray`.
    """
    return primitiveArray(values, spec, context, layoutOrder=layoutOrder)
