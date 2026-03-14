"""Activation and routing data collectors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from circuit_tracer.collectors.architecture_map import ArchitectureMap

if TYPE_CHECKING:
    from circuit_tracer.collectors.activation_collector import ActivationCollector

__all__ = ["ActivationCollector", "ArchitectureMap"]


def __getattr__(name: str):
    if name == "ActivationCollector":
        from circuit_tracer.collectors.activation_collector import ActivationCollector
        return ActivationCollector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
