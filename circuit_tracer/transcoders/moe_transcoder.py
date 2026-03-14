"""
Per-layer MoE transcoder.

Maps the pre-MoE residual stream to the post-MoE output.  From any single
token's perspective the MoE layer acts like one MLP (the weighted combination
of the routed experts + shared expert), so we train a single transcoder per
layer on (input, aggregated-output) pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TranscoderOutput:
    y_hat: torch.Tensor         # predicted MoE output
    z: torch.Tensor             # sparse feature activations
    loss: torch.Tensor
    recon_loss: torch.Tensor
    sparsity_loss: torch.Tensor
    l0: float


class MoETranscoder(nn.Module):
    """
    Transcoder that explains a single MoE layer.

    Unlike an SAE (which reconstructs *the same* vector), a transcoder
    maps the layer *input* to the layer *output*, learning to approximate
    the combined expert computation in a sparse feature basis.

    Architecture::

        z  = ReLU(encoder(x_pre_moe))       # sparse features
        y_hat = decoder(z)                   # predicted MoE output
        target = y_post_moe - x_pre_moe      # residual contribution of MoE
    """

    def __init__(
        self,
        d_model: int,
        n_features: int,
        l1_coeff: float = 1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.l1_coeff = l1_coeff

        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.decoder.weight.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.decoder.weight.data.div_(norms)

    def forward(
        self,
        x_pre: torch.Tensor,
        x_post: torch.Tensor | None = None,
    ) -> TranscoderOutput:
        """
        Args:
            x_pre:  residual stream before MoE layer  [batch, d_model]
            x_post: residual stream after MoE layer   [batch, d_model]
                    If provided, loss is computed against the MoE residual
                    contribution (x_post - x_pre).
        """
        z = F.relu(self.encoder(x_pre))
        y_hat = self.decoder(z)

        if x_post is not None:
            target = x_post - x_pre  # MoE residual contribution
            recon_loss = F.mse_loss(y_hat, target)
        else:
            recon_loss = torch.tensor(0.0, device=x_pre.device)

        sparsity_loss = self.l1_coeff * z.abs().mean()
        loss = recon_loss + sparsity_loss
        l0 = (z > 0).float().sum(dim=-1).mean().item()

        return TranscoderOutput(
            y_hat=y_hat,
            z=z,
            loss=loss,
            recon_loss=recon_loss,
            sparsity_loss=sparsity_loss,
            l0=l0,
        )

    def save(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "d_model": self.d_model,
                "n_features": self.n_features,
                "l1_coeff": self.l1_coeff,
            },
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "MoETranscoder":
        data = torch.load(path, map_location=device, weights_only=False)
        tc = cls(**data["config"])
        tc.load_state_dict(data["state_dict"])
        return tc
