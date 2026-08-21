"""Semantic object descriptions shared by transport implementations."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Callable, Iterable


Getter = str | Callable[[object], object]


def _read(owner, getter: Getter):
    return getattr(owner, getter, None) if isinstance(getter, str) else getter(owner)


@dataclass(frozen=True)
class FieldDescription:
    """One canonical value owned by a frontend primitive.

    ``controlField`` optionally gates a dynamic value behind an explicitly
    selected synchronized control field. Static graph composition remains
    independent of runtime control flow.
    """

    name: str
    getter: Getter
    axes: tuple[str, ...] = ()
    dynamic: bool = False
    controlField: str | None = None
    optional: bool = False
    encoding: str = "native"

    def value(self, owner):
        return _read(owner, self.getter)


@dataclass(frozen=True)
class ReferenceDescription:
    """One canonical relationship to another frontend primitive."""

    name: str
    getter: Getter
    many: bool = False
    optional: bool = False

    def value(self, owner):
        return _read(owner, self.getter)


@dataclass(frozen=True)
class PrimitiveDescription:
    """Transport-neutral description supplied by one frontend type."""

    typeName: str
    fields: tuple[FieldDescription, ...] = ()
    references: tuple[ReferenceDescription, ...] = ()


def field(
    name: str,
    getter: Getter | None = None,
    *,
    axes: Iterable[str] = (),
    dynamic: bool = False,
    controlField: str | None = None,
    optional: bool = False,
    encoding: str = "native",
) -> FieldDescription:
    if controlField is not None and not dynamic:
        raise ValueError("a control-gated transport field must be dynamic")
    return FieldDescription(
        name=name,
        getter=name if getter is None else getter,
        axes=tuple(axes),
        dynamic=dynamic,
        controlField=controlField,
        optional=optional,
        encoding=encoding,
    )


def reference(
    name: str,
    getter: Getter | None = None,
    *,
    many: bool = False,
    optional: bool = False,
) -> ReferenceDescription:
    return ReferenceDescription(
        name=name,
        getter=name if getter is None else getter,
        many=many,
        optional=optional,
    )


@dataclass(frozen=True)
class TransportNode:
    """One resolved object in a composed transport graph."""

    owner: object = dataclass_field(compare=False, repr=False)
    path: str
    typeName: str
    fields: dict[str, tuple[FieldDescription, object]]
    references: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class TransportGraph:
    """Deduplicated, path-addressable frontend object graph."""

    root: str
    nodes: tuple[TransportNode, ...]

    def node(self, path: str) -> TransportNode:
        for node in self.nodes:
            if node.path == path:
                return node
        raise KeyError(path)


class TransportComposer:
    """Follow primitive-owned references and deduplicate shared objects."""

    def __init__(self):
        self._paths = {}
        self._counts = {}
        self._nodes = []

    def compose(self, root) -> TransportGraph:
        self._paths.clear()
        self._counts.clear()
        self._nodes.clear()
        root_path = self._include(root, preferredPath="")
        return TransportGraph(root=root_path, nodes=tuple(self._nodes))

    def _include(self, owner, *, preferredPath=None) -> str:
        if owner is None:
            raise TypeError("a required transport reference resolved to None")
        identity = id(owner)
        existing = self._paths.get(identity)
        if existing is not None:
            return existing

        description_method = getattr(owner, "_transportDescription", None)
        if description_method is None:
            raise TypeError(
                f"{type(owner).__name__} does not provide an internal transport description"
            )
        description = description_method()
        if not isinstance(description, PrimitiveDescription):
            raise TypeError(
                f"{type(owner).__name__}._transportDescription() must return "
                "PrimitiveDescription"
            )

        if preferredPath is None:
            index = self._counts.get(description.typeName, 0)
            self._counts[description.typeName] = index + 1
            path = f"objects/{description.typeName}/{index}"
        elif not preferredPath:
            path = description.typeName
        else:
            path = preferredPath
        self._paths[identity] = path

        field_values = {}
        for spec in description.fields:
            value = spec.value(owner)
            if value is None and not spec.optional:
                raise ValueError(
                    f"required transport field {description.typeName}.{spec.name} is None"
                )
            field_values[spec.name] = (spec, value)

        node_index = len(self._nodes)
        self._nodes.append(None)
        resolved_references = {}
        for spec in description.references:
            value = spec.value(owner)
            if value is None:
                if not spec.optional:
                    raise ValueError(
                        f"required transport reference {description.typeName}.{spec.name} is None"
                    )
                resolved_references[spec.name] = ()
                continue
            values = tuple(value) if spec.many else (value,)
            resolved_references[spec.name] = tuple(self._include(item) for item in values)

        self._nodes[node_index] = TransportNode(
            owner=owner,
            path=path,
            typeName=description.typeName,
            fields=field_values,
            references=resolved_references,
        )
        return path
