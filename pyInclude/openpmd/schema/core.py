from __future__ import annotations

from dataclasses import dataclass

from ..FieldSpec import FieldSpec
from .unit_dimension import DIMENSIONLESS
from hase_units import Quantity, Unit


BACKEND_FLAT = "backendFlat"


def _camel_to_snake(name: str) -> str:
    out = []
    for index, char in enumerate(name):
        if char.isupper() and index > 0 and (not name[index - 1].isupper()):
            out.append("_")
        out.append(char.lower())
    return "".join(out)


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
    return (getattr(context, "numberOfSurfaceDomains", 0),)


def _shapeCellInterface(context):
    return (context.numberOfTriangles, 2)


def _shapeWavelength(context):
    return (context.spectralSampleCount,)



_SHAPES_BY_AXES = {
    ("point",): _shapePoint,
    ("cell",): _shapeCell,
    ("coordinate", "point"): lambda context: (2, context.numberOfPoints),
    ("point", "coordinate"): lambda context: (context.numberOfPoints, 2),
    ("coordinate", "cell"): lambda context: (2, context.numberOfTriangles),
    ("cell", "coordinate"): lambda context: (context.numberOfTriangles, 2),
    ("cell", "local_vertex"): _shapeCellSide,
    ("cell", "local_side"): _shapeCellSide,
    ("cell", "local_side", "coordinate"): lambda context: (context.numberOfTriangles, 3, 2),
    ("cell", "layer"): _shapeCellLayer,
    ("point", "level"): _shapePointLevel,
    ("interface",): _shapeInterface,
    ("cell", "interface"): _shapeCellInterface,
    ("surface",): _shapeSurface,
    ("wavelength",): _shapeWavelength,
}


@dataclass(frozen=True, init=False)
class PrimitiveFieldSpec:
    """Field declaration whose axes and shape may inherit from a primitive.

    Parameters
    ----------
    name
        Python-side field name.
    recordNameOrDtype
        Either the storage dtype, which derives a snake-case record name, or an
        explicit record name when ``dtype`` is also supplied.
    dtype
        Optional NumPy-compatible dtype for the explicit-record-name form.
    axes
        Optional logical axes; ``None`` inherits primitive axes.
    shape
        Optional callable from context to shape; ``None`` uses the registered
        shape rule for the resolved axes.
    unit
        Unit symbol or physical :class:`Unit`.
    unitSI, unitDimension
        Explicit openPMD scale and seven SI dimension exponents when ``unit``
        is a string.
    dynamic
        Whether values may change after iteration zero.
    backendRequired
        Whether native parsing requires the record.
    userDefined
        Whether the application owns this field.
    schemaRole
        Logical role such as ``"input"`` or ``"result"``.
    recordName
        Keyword-only override for the derived or positional record name.
    """
    name: str
    """Python-side attribute name used by schema-aware application code."""
    recordName: str
    """Exact openPMD record suffix; this is a serialized wire name."""
    dtype: object
    """NumPy-compatible scalar storage type used for the record component."""
    axes: tuple[str, ...] | None = None
    """Logical array axes, or ``None`` to inherit the primitive's axes."""
    shape: object | None = None
    """Context-to-shape callable, or ``None`` to use the registered axis rule."""
    unit: str = "1"
    """Human-readable physical unit symbol stored in schema metadata."""
    unitSI: float = 1.0
    """Multiplier converting stored magnitudes to SI values."""
    unitDimension: tuple[float, float, float, float, float, float, float] = DIMENSIONLESS
    """Seven openPMD SI base-dimension exponents."""
    dynamic: bool = False
    """Whether the record may change after transport iteration zero."""
    backendRequired: bool = True
    """Whether omission makes the native backend request invalid."""
    userDefined: bool = False
    """Whether application code, rather than HASE, owns the record values."""
    schemaRole: str = "input"
    """Logical transport role, normally ``"input"`` or ``"result"``."""

    def __init__(
        self,
        name: str,
        recordNameOrDtype: object,
        dtype: object | None = None,
        axes: tuple[str, ...] | None = None,
        shape: object | None = None,
        unit: str = "1",
        unitSI: float = 1.0,
        unitDimension: tuple[float, float, float, float, float, float, float] = DIMENSIONLESS,
        dynamic: bool = False,
        backendRequired: bool = True,
        userDefined: bool = False,
        schemaRole: str = "input",
        *,
        recordName: str | None = None,
    ):
        if dtype is None:
            dtype = recordNameOrDtype
            resolved_record_name = recordName or _camel_to_snake(name)
        else:
            resolved_record_name = str(recordName or recordNameOrDtype)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "recordName", resolved_record_name)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "unitSI", unitSI)
        object.__setattr__(self, "unitDimension", unitDimension)
        object.__setattr__(self, "dynamic", dynamic)
        object.__setattr__(self, "backendRequired", backendRequired)
        object.__setattr__(self, "userDefined", userDefined)
        object.__setattr__(self, "schemaRole", schemaRole)

    def toFieldSpec(self, primitiveAxes: tuple[str, ...]) -> FieldSpec:
        """Resolve inherited axes/shape into a concrete :class:`FieldSpec`.

        Parameters
        ----------
        primitiveAxes
            Logical axes inherited when this declaration's ``axes`` is None.
        """
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
    """Runtime collection of related field declarations.

    Parameters
    ----------
    name
        Primitive name used by the HASE schema.
    axes
        Logical axes inherited by fields that omit their own axes.
    fields
        Ordered :class:`PrimitiveFieldSpec` tuple.
    shapeField
        Optional field whose shape defines the primitive extent.
    """
    name: str
    """Primitive name used in HASE's openPMD schema contract."""
    axes: tuple[str, ...]
    """Logical axes inherited by fields that do not declare their own."""
    fields: tuple[PrimitiveFieldSpec, ...]
    """Ordered field declarations belonging to this primitive."""
    shapeField: str | None = None
    """Optional field whose resolved shape defines the primitive extent."""

    def fieldSpecs(self) -> tuple[FieldSpec, ...]:
        """Return all fields with inherited axes and shapes resolved."""
        return tuple(field.toFieldSpec(self.axes) for field in self.fields)

    def extend(self, *fields: PrimitiveFieldSpec) -> "PrimitiveSchema":
        """Return a new schema with fields appended in order."""
        return PrimitiveSchema(self.name, self.axes, self.fields + tuple(fields), self.shapeField)

    def fieldSpec(self, name: str) -> FieldSpec:
        """Return the resolved field named ``name`` or raise ``KeyError``."""
        for spec in self.fieldSpecs():
            if spec.name == name:
                return spec
        raise KeyError(name)

    def expectedShape(self, context, parentShape: tuple[int, ...] | None = None) -> tuple[int, ...]:
        """Resolve the primitive extent from its shape field or context."""
        if self.shapeField is not None:
            spec = self.fieldSpec(self.shapeField)
            field_shape = spec.expectedShape(context)
            return tuple(size for size, axis in zip(field_shape, spec.axes) if axis in self.axes)
        if parentShape is not None:
            return tuple(parentShape)
        return tuple(_SHAPES_BY_AXES[self.axes](context))


@dataclass(frozen=True)
class ExtensionAttributeSpec:
    """Typed scalar attribute in the HASE openPMD extension contract.

    Parameters
    ----------
    name
        Python/internal name.
    attribute
        Snake-case name stored on an openPMD iteration.
    dtype
        One of ``"bool"``, ``"int"``, ``"float"``, or ``"str"``.
    unit, unitSI, unitDimension
        Physical wire unit used to convert :class:`Quantity` input before the
        scalar attribute is stored.
    """
    name: str
    """Lower-camel Python name used by frontend and adapter code."""
    attribute: str
    """Exact snake-case attribute name serialized on the openPMD iteration."""
    dtype: str
    """Wire scalar category: bool, int, float, or string."""
    unit: str = "1"
    """Human-readable physical unit symbol for the serialized value."""
    unitSI: float = 1.0
    """Multiplier converting the serialized magnitude to SI."""
    unitDimension: tuple[float, float, float, float, float, float, float] = DIMENSIONLESS
    """Seven openPMD SI base-dimension exponents."""

    def __post_init__(self):
        if isinstance(self.unit, Unit):
            object.__setattr__(self, "unitSI", self.unit.unitSI)
            object.__setattr__(self, "unitDimension", self.unit.unitDimension)
            object.__setattr__(self, "unit", self.unit.symbol)

    @property
    def physicalUnit(self):
        """Resolved physical wire unit."""
        return Unit.fromOpenPmd(self.unit, self.unitSI, self.unitDimension)

    def cast(self, value):
        """Convert an optional quantity and cast it to the declared wire type."""
        if isinstance(value, Quantity):
            value = value.toValue(self.physicalUnit)
        if self.dtype == "bool":
            return bool(value)
        if self.dtype == "int":
            return int(value)
        if self.dtype == "float":
            return float(value)
        if self.dtype == "str":
            return str(value)
        raise ValueError(f"unknown openPMD transport attribute dtype '{self.dtype}'")


_GROUP_FIELD_MISSING = object()


@dataclass(frozen=True)
class GroupFieldSpec:
    """Declarative field copied from a :class:`BaseGroup` to a primitive.

    Parameters
    ----------
    name
        Constructor keyword and default destination name.
    dtype
        Expected application-side data type.
    default
        Optional default; omitting it makes the group constructor value
        required.
    targetName
        Optional destination attribute override.
    unit, unitSI, unitDimension
        Physical unit metadata associated with the grouped field.
    """
    name: str
    """Constructor keyword and default primitive destination name."""
    dtype: object
    """Expected application-side value type."""
    default: object = _GROUP_FIELD_MISSING
    """Optional default value; the sentinel means the field is required."""
    targetName: str | None = None
    """Optional destination-attribute override on the target primitive."""
    unit: str = "1"
    """Human-readable physical unit symbol associated with the field."""
    unitSI: float = 1.0
    """Multiplier converting a stored field value to SI."""
    unitDimension: tuple[float, float, float, float, float, float, float] = DIMENSIONLESS
    """Seven openPMD SI base-dimension exponents."""

    @property
    def target(self) -> str:
        """Destination attribute name on the primitive."""
        return self.targetName or self.name


class BaseGroup:
    """Base for declarative groups of values applied to schema primitives.

    Subclasses declare :class:`GroupFieldSpec` objects as class attributes.
    Constructor keywords provide their values; unknown and missing required
    fields are rejected.

    Parameters
    ----------
    **values
        One value per declared :class:`GroupFieldSpec` without a default.
    """
    groupName: str | None = None
    """Optional stable group label; defaults to the subclass name."""

    def __init__(self, **values):
        specs = {spec.name: spec for spec in self.fieldSpecs()}
        unknown = set(values) - set(specs)
        if unknown:
            raise TypeError(f"unknown group field '{sorted(unknown)[0]}'")
        resolved = {}
        for name, spec in specs.items():
            if name in values:
                resolved[name] = values[name]
            elif spec.default is not _GROUP_FIELD_MISSING:
                resolved[name] = spec.default
            else:
                raise TypeError(f"missing value for group field '{name}'")
        self._values = resolved
        self._groupToken = object()

    @classmethod
    def declaredFields(cls) -> tuple[GroupFieldSpec, ...]:
        """Collect unique group fields across the inheritance hierarchy."""
        fields = []
        seen = set()
        for group_cls in reversed(cls.mro()):
            for value in group_cls.__dict__.values():
                if isinstance(value, GroupFieldSpec) and value.name not in seen:
                    fields.append(value)
                    seen.add(value.name)
        return tuple(fields)

    @classmethod
    def fieldSpecs(cls) -> tuple[GroupFieldSpec, ...]:
        """Return the ordered group-field declarations."""
        return cls.declaredFields()

    @property
    def name(self) -> str:
        """Explicit ``groupName`` or the subclass name."""
        return self.groupName or self.__class__.__name__

    def fieldItems(self):
        """Iterate over ``(specification, value)`` pairs."""
        for spec in self.fieldSpecs():
            yield spec, self._values[spec.name]

    def add(self, primitive):
        """Apply every group value to ``primitive`` and return it.

        Parameters
        ----------
        primitive
            Schema primitive receiving the declared destination attributes.

        Existing destination values are not overwritten.
        """
        if hasattr(primitive, "_applyGroup"):
            primitive._applyGroup(self)
            return primitive
        for spec, value in self.fieldItems():
            current = getattr(primitive, spec.target, _GROUP_FIELD_MISSING)
            if current is not _GROUP_FIELD_MISSING:
                raise ValueError(f"field '{spec.target}' is already assigned on this primitive")
            setattr(primitive, spec.target, value)
        return primitive


class BaseSchema:
    """Base for class-declared openPMD primitive schemas.

    Subclasses set ``primitiveName``, ``axes``, and optional ``shapeField``,
    then declare :class:`PrimitiveFieldSpec` values as class attributes.
    """
    primitiveName: str | None = None
    """Serialized primitive name, or the schema subclass name when omitted."""
    axes: tuple[str, ...] = ()
    """Logical axes inherited by declared fields that omit their own axes."""
    shapeField: str | None = None
    """Optional declared field whose resolved shape defines the primitive."""

    @classmethod
    def declaredFields(cls) -> tuple[PrimitiveFieldSpec, ...]:
        """Collect unique field declarations across the class hierarchy."""
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
        """Materialize this declaration as a :class:`PrimitiveSchema`."""
        name = cls.primitiveName or cls.__name__
        return PrimitiveSchema(name, tuple(cls.axes), cls.declaredFields(), cls.shapeField)

    @classmethod
    def fieldSpecs(cls) -> tuple[FieldSpec, ...]:
        """Return all concrete fields in declaration order."""
        return cls.primitiveSchema().fieldSpecs()


class PrimitiveSchemaDefinition(BaseSchema):
    """Semantic base class for schema definitions representing primitives."""
    pass
