"""Cross-layer transcoders for MoE output reconstruction."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from circuit_tracer.transcoders.moe_transcoder import MoETranscoder
    from circuit_tracer.transcoders.cross_layer_transcoder import CrossLayerTranscoder

__all__ = ["MoETranscoder", "CrossLayerTranscoder"]


def __getattr__(name: str):
    if name == "MoETranscoder":
        from circuit_tracer.transcoders.moe_transcoder import MoETranscoder
        return MoETranscoder
    if name == "CrossLayerTranscoder":
        from circuit_tracer.transcoders.cross_layer_transcoder import CrossLayerTranscoder
        return CrossLayerTranscoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
