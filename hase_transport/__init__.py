"""Transport-neutral descriptions of HASE frontend object graphs.

The public frontend API does not expose storage mechanics. Frontend types
privately describe their canonical fields and references through this package;
format-specific transports consume the resulting :class:`TransportGraph`.
"""

from .description import (
    FieldDescription,
    PrimitiveDescription,
    ReferenceDescription,
    TransportComposer,
    TransportGraph,
    TransportNode,
    field,
    reference,
)

__all__ = [
    "FieldDescription",
    "PrimitiveDescription",
    "ReferenceDescription",
    "TransportComposer",
    "TransportGraph",
    "TransportNode",
    "field",
    "reference",
]
