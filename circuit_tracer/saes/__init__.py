"""Sparse autoencoder training and inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from circuit_tracer.saes.sparse_autoencoder import SparseAutoencoder
    from circuit_tracer.saes.trainer import SAETrainer

__all__ = ["SparseAutoencoder", "SAETrainer"]


def __getattr__(name: str):
    if name == "SparseAutoencoder":
        from circuit_tracer.saes.sparse_autoencoder import SparseAutoencoder
        return SparseAutoencoder
    if name == "SAETrainer":
        from circuit_tracer.saes.trainer import SAETrainer
        return SAETrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
