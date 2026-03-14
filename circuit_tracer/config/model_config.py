"""
Model configuration for circuit tracing.

Decoupled from CreditScope's inference config so the package can be
reused with any HuggingFace model.  When running inside CreditScope
the defaults are read from environment variables set by inference/config.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """All knobs needed by the circuit-tracing pipeline."""

    # ── Model identity ────────────────────────────────────────────────────
    model_path: str = os.getenv("MODEL_PATH", "Qwen/Qwen3.5-35B-A3B-FP8")
    device_map: str = "auto"
    torch_dtype: str = "auto"

    # ── Architecture ──────────────────────────────────────────────────────
    num_layers: int = 64            # total decoder layers
    d_model: int = 2048             # hidden / residual-stream width
    num_experts: int = 64           # experts per MoE layer
    top_k_experts: int = 4          # routed experts per token
    moe_layer_pattern: str = "mlp.experts"
    # Qwen3.5-35B-A3B layout: 10 × (3 DeltaNet-MoE + 1 Attention-MoE) = 40 blocks
    # but has 64 decoder layers total
    attention_layer_indices: list[int] = field(default_factory=list)
    deltanet_layer_indices: list[int] = field(default_factory=list)

    # ── SAE ────────────────────────────────────────────────────────────────
    sae_expansion: int = 8          # feature dict size = d_model × expansion
    sae_l1_coeff: float = 3e-4     # sparsity penalty
    sae_learning_rate: float = 3e-4
    sae_batch_size: int = 4096      # tokens per training step
    sae_num_steps: int = 50_000
    sae_use_jumprelu: bool = True   # JumpReLU vs plain ReLU

    # ── Transcoder ─────────────────────────────────────────────────────────
    tc_expansion: int = 8
    tc_learning_rate: float = 1e-4
    tc_batch_size: int = 2048
    tc_num_steps: int = 100_000

    # ── Activation collection ─────────────────────────────────────────────
    activation_dir: Path = Path("circuit_tracer/data/activations")
    checkpoint_dir: Path = Path("circuit_tracer/data/checkpoints")
    collect_batch_size: int = 4     # sequences per forward pass
    collect_max_seq_len: int = 512  # truncate/pad sequences to this length
    collect_num_tokens: int = 10_000_000  # total tokens to collect

    # ── Attribution ───────────────────────────────────────────────────────
    attribution_threshold: float = 0.01  # min |attribution| to keep an edge
    prune_keep_fraction: float = 0.10    # keep top 10% of nodes after pruning

    # ── SGLang integration (CreditScope-specific) ─────────────────────────
    sglang_url: str = os.getenv("SGLANG_URL", "http://127.0.0.1:8000")

    @property
    def n_features(self) -> int:
        return self.d_model * self.sae_expansion

    @property
    def tc_features(self) -> int:
        return self.d_model * self.tc_expansion


_config: ModelConfig | None = None


def get_config(**overrides) -> ModelConfig:
    """Return the global config singleton, creating it on first call."""
    global _config
    if _config is None:
        _config = ModelConfig(**overrides)
    return _config
