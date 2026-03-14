"""
Sparse autoencoder for residual-stream feature extraction.

Supports both standard ReLU and JumpReLU activation functions.
JumpReLU (from the Anthropic scaling monosemanticity paper) gives
cleaner sparsity by learning per-feature thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEOutput:
    """Structured output from a forward pass."""
    x_hat: torch.Tensor         # reconstruction
    z: torch.Tensor             # latent activations (sparse)
    loss: torch.Tensor          # total loss
    recon_loss: torch.Tensor    # reconstruction MSE
    sparsity_loss: torch.Tensor # L1 penalty
    l0: float                   # average number of active features


class SparseAutoencoder(nn.Module):
    """
    Sparse autoencoder that decomposes residual-stream vectors into
    a sparse set of interpretable features.

    Architecture::

        x_centered = x - bias
        z = activation(encoder(x_centered))   # sparse latent
        x_hat = decoder(z) + bias             # reconstruction

    The decoder columns are unit-normalised after each optimiser step
    so that feature directions are meaningful.
    """

    def __init__(
        self,
        d_model: int,
        n_features: int,
        l1_coeff: float = 3e-4,
        use_jumprelu: bool = True,
        jumprelu_init_threshold: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.l1_coeff = l1_coeff
        self.use_jumprelu = use_jumprelu

        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_model))

        if use_jumprelu:
            self.jump_threshold = nn.Parameter(
                torch.full((n_features,), jumprelu_init_threshold)
            )

        self._init_weights()

    def _init_weights(self):
        # Kaiming uniform for encoder
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        # Transpose-tie initialisation for decoder
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        """Unit-normalise decoder columns (feature directions)."""
        norms = self.decoder.weight.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.decoder.weight.data.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse latent features."""
        x_centered = x - self.bias
        pre_act = self.encoder(x_centered)

        if self.use_jumprelu:
            # JumpReLU: z = pre_act * (pre_act > threshold)
            mask = (pre_act > self.jump_threshold).float()
            z = pre_act * mask
        else:
            z = F.relu(pre_act)

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse latent back to residual-stream space."""
        return self.decoder(z) + self.bias

    def forward(self, x: torch.Tensor) -> SAEOutput:
        z = self.encode(x)
        x_hat = self.decode(z)

        recon_loss = F.mse_loss(x_hat, x)
        sparsity_loss = self.l1_coeff * z.abs().mean()
        loss = recon_loss + sparsity_loss

        l0 = (z > 0).float().sum(dim=-1).mean().item()

        return SAEOutput(
            x_hat=x_hat,
            z=z,
            loss=loss,
            recon_loss=recon_loss,
            sparsity_loss=sparsity_loss,
            l0=l0,
        )

    # ── Feature inspection ────────────────────────────────────────────────

    def feature_directions(self) -> torch.Tensor:
        """Return (n_features, d_model) matrix of decoder directions."""
        return self.decoder.weight.data.T

    def top_activating_features(self, x: torch.Tensor, k: int = 10) -> torch.Tensor:
        """Return indices of the top-k most activated features for input x."""
        z = self.encode(x)
        return z.topk(k, dim=-1).indices

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "d_model": self.d_model,
                "n_features": self.n_features,
                "l1_coeff": self.l1_coeff,
                "use_jumprelu": self.use_jumprelu,
            },
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SparseAutoencoder":
        data = torch.load(path, map_location=device, weights_only=False)
        cfg = data["config"]
        sae = cls(**cfg)
        sae.load_state_dict(data["state_dict"])
        return sae
