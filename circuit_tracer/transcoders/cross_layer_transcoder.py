"""
Cross-layer transcoder (CLT) for MoE models.

Each feature is *read* from one layer's residual stream and *writes*
to the MoE output at that layer and all downstream layers.  This
captures the cross-layer information flow that attribution graphs need.

For the Qwen3.5-35B-A3B architecture:
- DeltaNet (linear attention) layers are frozen — we only model MoE layers
- The CLT learns a shared feature dictionary where each feature's encoder
  is tied to exactly one source layer but its decoder can contribute to
  any downstream layer's MoE output
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CLTOutput:
    """Output from a single forward pass through the CLT."""
    reconstructions: dict[int, torch.Tensor]   # layer_idx → predicted MoE output
    features: dict[int, torch.Tensor]          # layer_idx → sparse activations
    loss: torch.Tensor
    recon_loss: torch.Tensor
    sparsity_loss: torch.Tensor
    per_layer_l0: dict[int, float]


class CrossLayerTranscoder(nn.Module):
    """
    Cross-layer transcoder for a set of MoE layers.

    Design:
    - One encoder per source layer (reads residual stream → sparse features)
    - One decoder per (source, target) layer pair where target >= source
      (each feature can contribute to downstream MoE outputs)
    - Frozen DeltaNet/attention layers are not modelled

    This is a simplified CLT suitable for training on a single GPU.
    The full Anthropic version trains jointly across all layers, but we
    start with a tractable subset (e.g. every 4th MoE layer) and expand.
    """

    def __init__(
        self,
        d_model: int,
        n_features_per_layer: int,
        moe_layer_indices: list[int],
        max_cross_layer_distance: int = 8,
        l1_coeff: float = 1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_features_per_layer = n_features_per_layer
        self.moe_layer_indices = sorted(moe_layer_indices)
        self.max_distance = max_cross_layer_distance
        self.l1_coeff = l1_coeff

        # One encoder per source layer
        self.encoders = nn.ModuleDict({
            str(idx): nn.Linear(d_model, n_features_per_layer)
            for idx in self.moe_layer_indices
        })

        # Decoders: source → target (target >= source, within distance)
        self.decoders = nn.ModuleDict()
        for src in self.moe_layer_indices:
            for tgt in self.moe_layer_indices:
                if tgt >= src and (tgt - src) <= self.max_distance:
                    key = f"{src}_to_{tgt}"
                    self.decoders[key] = nn.Linear(n_features_per_layer, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        for enc in self.encoders.values():
            nn.init.kaiming_uniform_(enc.weight)
            nn.init.zeros_(enc.bias)
        for dec in self.decoders.values():
            nn.init.kaiming_uniform_(dec.weight)

    def get_features(self, layer_idx: int, residual_stream: torch.Tensor) -> torch.Tensor:
        """Encode the residual stream at a given layer into sparse features."""
        return F.relu(self.encoders[str(layer_idx)](residual_stream))

    def reconstruct_layer(
        self,
        target_layer: int,
        all_features: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Reconstruct the MoE output at target_layer using features from
        all upstream (and same-layer) sources.
        """
        output = None
        for src_layer, features in all_features.items():
            if src_layer > target_layer:
                continue
            if (target_layer - src_layer) > self.max_distance:
                continue
            key = f"{src_layer}_to_{target_layer}"
            if key not in self.decoders:
                continue
            contribution = self.decoders[key](features)
            if output is None:
                output = contribution
            else:
                output = output + contribution
        return output if output is not None else torch.zeros(1, self.d_model, device=next(self.parameters()).device)

    def forward(
        self,
        residual_streams: dict[int, torch.Tensor],
        moe_targets: dict[int, torch.Tensor] | None = None,
    ) -> CLTOutput:
        """
        Full forward pass.

        Args:
            residual_streams: layer_idx → pre-MoE residual [batch, d_model]
            moe_targets:      layer_idx → MoE residual contribution [batch, d_model]
                              (x_post - x_pre at each MoE layer).  None for inference.
        """
        # Encode all layers
        all_features: dict[int, torch.Tensor] = {}
        for idx in self.moe_layer_indices:
            if idx in residual_streams:
                all_features[idx] = self.get_features(idx, residual_streams[idx])

        # Reconstruct all target layers
        reconstructions: dict[int, torch.Tensor] = {}
        for idx in self.moe_layer_indices:
            if idx in residual_streams:
                reconstructions[idx] = self.reconstruct_layer(idx, all_features)

        # Loss
        recon_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        if moe_targets is not None:
            n = 0
            for idx, target in moe_targets.items():
                if idx in reconstructions:
                    recon_loss = recon_loss + F.mse_loss(reconstructions[idx], target)
                    n += 1
            if n > 0:
                recon_loss = recon_loss / n

        sparsity_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        per_layer_l0: dict[int, float] = {}
        for idx, feats in all_features.items():
            sparsity_loss = sparsity_loss + self.l1_coeff * feats.abs().mean()
            per_layer_l0[idx] = (feats > 0).float().sum(dim=-1).mean().item()

        if all_features:
            sparsity_loss = sparsity_loss / len(all_features)

        loss = recon_loss + sparsity_loss

        return CLTOutput(
            reconstructions=reconstructions,
            features=all_features,
            loss=loss,
            recon_loss=recon_loss,
            sparsity_loss=sparsity_loss,
            per_layer_l0=per_layer_l0,
        )

    def save(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "d_model": self.d_model,
                "n_features_per_layer": self.n_features_per_layer,
                "moe_layer_indices": self.moe_layer_indices,
                "max_cross_layer_distance": self.max_distance,
                "l1_coeff": self.l1_coeff,
            },
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CrossLayerTranscoder":
        data = torch.load(path, map_location=device, weights_only=False)
        clt = cls(**data["config"])
        clt.load_state_dict(data["state_dict"])
        return clt
