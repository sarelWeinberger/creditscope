"""Attribution graph computation and pruning."""

from __future__ import annotations

from typing import TYPE_CHECKING

from circuit_tracer.attribution.graph import AttributionGraph, AttributionNode, AttributionEdge
from circuit_tracer.attribution.pruning import prune_graph

if TYPE_CHECKING:
    from circuit_tracer.attribution.replacement_model import ReplacementModel

__all__ = [
    "AttributionGraph", "AttributionNode", "AttributionEdge",
    "ReplacementModel", "prune_graph",
]


def __getattr__(name: str):
    if name == "ReplacementModel":
        from circuit_tracer.attribution.replacement_model import ReplacementModel
        return ReplacementModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
