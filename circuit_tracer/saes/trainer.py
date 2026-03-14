"""
Training loop for sparse autoencoders on saved activations.

Streams activation data from .npy files and trains per-layer SAEs
with live metric logging (Prometheus-compatible when integrated
with CreditScope).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import structlog
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from circuit_tracer.config import get_config
from circuit_tracer.saes.sparse_autoencoder import SparseAutoencoder, SAEOutput

logger = structlog.get_logger(__name__)


class ActivationDataset:
    """Streams shuffled activation vectors from a .npy file."""

    def __init__(self, path: str | Path, batch_size: int = 4096, device: str = "cuda"):
        self.data = torch.from_numpy(np.load(path)).to(torch.float32)
        self.batch_size = batch_size
        self.device = device
        self.n = self.data.shape[0]
        self._perm: torch.Tensor | None = None
        self._pos = 0

    def __len__(self):
        return self.n // self.batch_size

    def shuffle(self):
        self._perm = torch.randperm(self.n)
        self._pos = 0

    def __iter__(self):
        self.shuffle()
        return self

    def __next__(self) -> torch.Tensor:
        if self._perm is None:
            self.shuffle()
        if self._pos + self.batch_size > self.n:
            raise StopIteration
        idx = self._perm[self._pos : self._pos + self.batch_size]
        self._pos += self.batch_size
        return self.data[idx].to(self.device)


class SAETrainer:
    """
    Train a SparseAutoencoder on pre-collected activations.

    Usage::

        trainer = SAETrainer(layer_idx=10)
        sae, metrics = trainer.train()
    """

    def __init__(
        self,
        layer_idx: int,
        activation_path: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        on_step: Callable[[int, dict], None] | None = None,
    ):
        cfg = get_config()
        self.layer_idx = layer_idx
        self.activation_path = Path(
            activation_path or cfg.activation_dir / f"layer_{layer_idx}_residual_pre.npy"
        )
        self.checkpoint_dir = Path(checkpoint_dir or cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.on_step = on_step  # optional callback for live metrics

    def train(
        self,
        d_model: int | None = None,
        n_features: int | None = None,
        num_steps: int | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
        l1_coeff: float | None = None,
        device: str = "cuda",
    ) -> tuple[SparseAutoencoder, dict]:
        """Run training and return the trained SAE + final metrics."""
        cfg = get_config()
        d_model = d_model or cfg.d_model
        n_features = n_features or cfg.n_features
        num_steps = num_steps or cfg.sae_num_steps
        batch_size = batch_size or cfg.sae_batch_size
        lr = lr or cfg.sae_learning_rate
        l1_coeff = l1_coeff or cfg.sae_l1_coeff

        logger.info(
            "sae_training_start",
            layer=self.layer_idx,
            d_model=d_model,
            n_features=n_features,
            steps=num_steps,
            batch_size=batch_size,
            lr=lr,
            l1_coeff=l1_coeff,
            activation_path=str(self.activation_path),
        )

        sae = SparseAutoencoder(
            d_model=d_model,
            n_features=n_features,
            l1_coeff=l1_coeff,
            use_jumprelu=cfg.sae_use_jumprelu,
        ).to(device)

        optimizer = Adam(sae.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=lr * 0.1)

        dataset = ActivationDataset(self.activation_path, batch_size=batch_size, device=device)

        step = 0
        best_recon = float("inf")
        metrics_history: list[dict] = []
        t0 = time.time()

        while step < num_steps:
            for batch in dataset:
                if step >= num_steps:
                    break

                optimizer.zero_grad()
                out: SAEOutput = sae(batch)
                out.loss.backward()
                optimizer.step()
                scheduler.step()

                # Normalise decoder after each step
                sae._normalize_decoder()

                # Logging
                if step % 500 == 0:
                    metrics = {
                        "step": step,
                        "loss": out.loss.item(),
                        "recon_loss": out.recon_loss.item(),
                        "sparsity_loss": out.sparsity_loss.item(),
                        "l0": out.l0,
                        "lr": scheduler.get_last_lr()[0],
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                    metrics_history.append(metrics)
                    logger.info("sae_train_step", layer=self.layer_idx, **metrics)

                    if self.on_step:
                        self.on_step(step, metrics)

                # Checkpoint best model
                if step % 5000 == 0 and out.recon_loss.item() < best_recon:
                    best_recon = out.recon_loss.item()
                    ckpt_path = self.checkpoint_dir / f"sae_layer_{self.layer_idx}_best.pt"
                    sae.save(str(ckpt_path))
                    logger.info("sae_checkpoint_saved", path=str(ckpt_path), recon=best_recon)

                step += 1

        # Final save
        final_path = self.checkpoint_dir / f"sae_layer_{self.layer_idx}_final.pt"
        sae.save(str(final_path))
        logger.info("sae_training_complete", layer=self.layer_idx, steps=step, elapsed_s=round(time.time() - t0, 1))

        final_metrics = {
            "layer": self.layer_idx,
            "total_steps": step,
            "best_recon_loss": best_recon,
            "final_loss": metrics_history[-1]["loss"] if metrics_history else None,
            "final_l0": metrics_history[-1]["l0"] if metrics_history else None,
            "elapsed_s": round(time.time() - t0, 1),
        }
        return sae, final_metrics
