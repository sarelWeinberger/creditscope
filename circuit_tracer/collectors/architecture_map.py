"""
Discover and document the layer-by-layer architecture of a transformer model.

Produces a structured map of each decoder layer: attention type (standard
vs DeltaNet), FFN type (dense vs MoE), hidden dimensions, expert counts,
and residual stream access points.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LayerInfo:
    """Metadata for a single decoder layer."""
    index: int
    attention_type: str = "unknown"     # "standard", "deltanet", "gated_deltanet"
    ffn_type: str = "unknown"           # "dense", "moe"
    num_experts: int = 0                # 0 for dense FFN layers
    top_k: int = 0
    has_shared_expert: bool = False
    d_model: int = 0
    intermediate_size: int = 0
    module_path: str = ""               # e.g. "model.layers.5"


@dataclass
class ArchitectureMap:
    """Full architecture map for a transformer model."""
    model_name: str = ""
    num_layers: int = 0
    d_model: int = 0
    vocab_size: int = 0
    layers: list[LayerInfo] = field(default_factory=list)
    moe_layer_indices: list[int] = field(default_factory=list)
    dense_layer_indices: list[int] = field(default_factory=list)
    attention_layer_indices: list[int] = field(default_factory=list)
    deltanet_layer_indices: list[int] = field(default_factory=list)

    # ── Builder ───────────────────────────────────────────────────────────

    @classmethod
    def from_model(cls, model: Any, model_name: str = "") -> "ArchitectureMap":
        """Inspect a loaded model and build the architecture map."""
        arch = cls(model_name=model_name)

        # Resolve inner model (HookedModel wrapper or raw HF model)
        raw = model
        if hasattr(raw, "model"):
            raw = raw.model
        if hasattr(raw, "model"):
            raw = raw.model  # model.model.layers pattern

        if not hasattr(raw, "layers"):
            logger.warning("cannot_find_layers", type=type(raw).__name__)
            return arch

        layers = raw.layers
        arch.num_layers = len(layers)

        # Detect d_model from embedding
        if hasattr(raw, "embed_tokens"):
            arch.d_model = raw.embed_tokens.embedding_dim
            arch.vocab_size = raw.embed_tokens.num_embeddings

        for idx, layer in enumerate(layers):
            info = _inspect_layer(idx, layer)
            arch.layers.append(info)

            if info.ffn_type == "moe":
                arch.moe_layer_indices.append(idx)
            else:
                arch.dense_layer_indices.append(idx)

            if info.attention_type in ("deltanet", "gated_deltanet"):
                arch.deltanet_layer_indices.append(idx)
            elif info.attention_type == "standard":
                arch.attention_layer_indices.append(idx)

        logger.info(
            "architecture_mapped",
            layers=arch.num_layers,
            moe_layers=len(arch.moe_layer_indices),
            dense_layers=len(arch.dense_layer_indices),
            deltanet_layers=len(arch.deltanet_layer_indices),
            attention_layers=len(arch.attention_layer_indices),
            d_model=arch.d_model,
        )
        return arch

    # ── Serialisation ─────────────────────────────────────────────────────

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        logger.info("architecture_map_saved", path=str(path))

    @classmethod
    def load(cls, path: str | Path) -> "ArchitectureMap":
        data = json.loads(Path(path).read_text())
        layers = [LayerInfo(**l) for l in data.pop("layers", [])]
        return cls(**data, layers=layers)

    # ── Convenience ───────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"Model: {self.model_name}",
            f"Layers: {self.num_layers}  |  d_model: {self.d_model}  |  vocab: {self.vocab_size}",
            f"MoE layers: {len(self.moe_layer_indices)}  |  Dense layers: {len(self.dense_layer_indices)}",
            f"DeltaNet layers: {len(self.deltanet_layer_indices)}  |  Attention layers: {len(self.attention_layer_indices)}",
            "",
            "Layer layout:",
        ]
        for info in self.layers:
            attn = info.attention_type[:4].upper()
            ffn = info.ffn_type.upper()
            experts = f" ({info.num_experts}E top-{info.top_k})" if info.ffn_type == "moe" else ""
            lines.append(f"  [{info.index:3d}] {attn:6s} | {ffn:5s}{experts}")
        return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _inspect_layer(idx: int, layer: Any) -> LayerInfo:
    """Inspect a single decoder layer module."""
    info = LayerInfo(index=idx, module_path=f"model.layers.{idx}")

    # ── Detect attention type ─────────────────────────────────────────
    for name, mod in layer.named_modules():
        mod_type = type(mod).__name__.lower()
        if "deltanet" in mod_type:
            info.attention_type = "gated_deltanet" if "gated" in mod_type else "deltanet"
            break
        if "attention" in mod_type and "deltanet" not in mod_type:
            info.attention_type = "standard"
            # don't break — a deeper module might be DeltaNet

    # ── Detect FFN type ───────────────────────────────────────────────
    for name, mod in layer.named_modules():
        mod_type = type(mod).__name__.lower()
        if any(kw in mod_type for kw in ("fusedmoe", "sparsemlp", "moeblock")):
            info.ffn_type = "moe"
            info.num_experts = getattr(mod, "num_experts", 0) or _count_experts(mod)
            info.top_k = getattr(mod, "top_k", 0) or getattr(mod, "num_experts_per_tok", 0)
            info.has_shared_expert = _has_shared_expert(layer)
            break
        if "expert" in name and "mlp" in name:
            info.ffn_type = "moe"
            break

    if info.ffn_type == "unknown":
        # Check for plain MLP
        for name, mod in layer.named_modules():
            if "mlp" in name.lower() and "expert" not in name.lower():
                info.ffn_type = "dense"
                break

    # ── Dimensions ────────────────────────────────────────────────────
    for name, param in layer.named_parameters():
        if "self_attn" in name and "q_proj" in name and "weight" in name:
            info.d_model = param.shape[1]
            break

    for name, param in layer.named_parameters():
        if ("gate_proj" in name or "up_proj" in name) and "weight" in name:
            info.intermediate_size = param.shape[0]
            break

    return info


def _count_experts(module: Any) -> int:
    """Count expert sub-modules when num_experts attr is missing."""
    count = 0
    for name, _ in module.named_modules():
        if "expert" in name.lower():
            count += 1
    return count


def _has_shared_expert(layer: Any) -> bool:
    """Check if the layer has a shared expert."""
    for name, _ in layer.named_modules():
        if "shared" in name.lower() and "expert" in name.lower():
            return True
    return False
