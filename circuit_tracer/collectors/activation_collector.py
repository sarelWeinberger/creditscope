"""
Collect residual-stream activations from a transformer model for SAE / transcoder
training.

Captures the residual stream at layer boundaries (before and after each decoder
layer) as well as MoE sub-layer inputs and outputs when requested.

Activations are saved as memory-mapped tensors for streaming into the SAE
trainer without loading everything into RAM.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator

import structlog
import torch
import numpy as np

from circuit_tracer.config import get_config

logger = structlog.get_logger(__name__)


class ActivationCollector:
    """
    Streams activations from a HookedModel to disk.

    Usage::

        model = HookedModel.load()
        collector = ActivationCollector(model, layer_indices=[0, 10, 20, 30, 39])
        collector.collect(texts)
        # Saved to circuit_tracer/data/activations/layer_XX_residual_pre.npy
    """

    def __init__(
        self,
        model: Any,
        layer_indices: list[int] | None = None,
        capture_points: list[str] | None = None,
        save_dir: str | Path | None = None,
    ):
        """
        Args:
            model:          HookedModel instance.
            layer_indices:  Which decoder layers to capture.  None → all.
            capture_points: What to capture at each layer.
                            Options: "residual_pre", "residual_post", "moe_input", "moe_output"
                            Default: ["residual_pre"]
            save_dir:       Where to write activation files.
        """
        self.model = model
        cfg = get_config()
        self.layer_indices = layer_indices or list(range(model.num_layers))
        self.capture_points = capture_points or ["residual_pre"]
        self.save_dir = Path(save_dir or cfg.activation_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._buffers: dict[str, list[torch.Tensor]] = {}
        self._hooks: list[Any] = []

    # ── Public API ────────────────────────────────────────────────────────

    def collect(
        self,
        texts: list[str],
        batch_size: int | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Path]:
        """
        Run forward passes and save activations to disk.

        Returns dict mapping capture keys (e.g. "layer_10_residual_pre")
        to their .npy file paths.
        """
        cfg = get_config()
        batch_size = batch_size or cfg.collect_batch_size
        max_tokens = max_tokens or cfg.collect_num_tokens
        collected_tokens = 0

        self._setup_hooks()
        self._reset_buffers()

        t0 = time.time()
        try:
            for batch in self._batched(texts, batch_size):
                encoded = self.model.tokenize(batch)
                self.model.forward(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded.get("attention_mask"),
                )
                collected_tokens += encoded["input_ids"].numel()

                if collected_tokens % 100_000 < batch_size * cfg.collect_max_seq_len:
                    logger.info("collecting_activations", tokens=collected_tokens)

                if collected_tokens >= max_tokens:
                    break
        finally:
            self.model.clear_hooks()
            self._hooks.clear()

        elapsed = time.time() - t0
        logger.info(
            "activation_collection_complete",
            tokens=collected_tokens,
            elapsed_s=round(elapsed, 1),
        )

        return self._save_buffers(collected_tokens)

    def collect_single(
        self,
        text: str,
    ) -> dict[str, torch.Tensor]:
        """
        Collect activations for a single input (for attribution graphs).

        Returns dict mapping capture keys to tensors (on CPU, not saved to disk).
        """
        self._setup_hooks()
        self._reset_buffers()

        try:
            encoded = self.model.tokenize(text)
            self.model.forward(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
            )
        finally:
            self.model.clear_hooks()
            self._hooks.clear()

        result = {}
        for key, chunks in self._buffers.items():
            if chunks:
                result[key] = torch.cat(chunks, dim=0)
        return result

    # ── Hook registration ─────────────────────────────────────────────────

    def _setup_hooks(self):
        self.model.clear_hooks()
        self._hooks.clear()

        for layer_idx in self.layer_indices:
            layer = self.model.layers[layer_idx]

            if "residual_pre" in self.capture_points:
                key = f"layer_{layer_idx}_residual_pre"
                h = layer.register_forward_pre_hook(self._capture_pre(key))
                self._hooks.append(h)

            if "residual_post" in self.capture_points:
                key = f"layer_{layer_idx}_residual_post"
                h = layer.register_forward_hook(self._capture_post(key))
                self._hooks.append(h)

            if "moe_input" in self.capture_points or "moe_output" in self.capture_points:
                moe = self._find_moe_submodule(layer)
                if moe is not None:
                    if "moe_input" in self.capture_points:
                        key = f"layer_{layer_idx}_moe_input"
                        h = moe.register_forward_pre_hook(self._capture_pre(key))
                        self._hooks.append(h)
                    if "moe_output" in self.capture_points:
                        key = f"layer_{layer_idx}_moe_output"
                        h = moe.register_forward_hook(self._capture_post(key))
                        self._hooks.append(h)

    def _capture_pre(self, key: str):
        """Create a forward-pre hook that saves the input tensor."""
        def hook_fn(module, args):
            x = args[0] if isinstance(args, tuple) else args
            if isinstance(x, torch.Tensor):
                self._buffers.setdefault(key, []).append(
                    x.detach().cpu().to(torch.float32)
                )
        return hook_fn

    def _capture_post(self, key: str):
        """Create a forward hook that saves the output tensor."""
        def hook_fn(module, args, output):
            x = output[0] if isinstance(output, tuple) else output
            if isinstance(x, torch.Tensor):
                self._buffers.setdefault(key, []).append(
                    x.detach().cpu().to(torch.float32)
                )
        return hook_fn

    @staticmethod
    def _find_moe_submodule(layer) -> Any | None:
        """Locate the MoE / FusedMoE submodule inside a decoder layer."""
        for name, mod in layer.named_modules():
            mod_type = type(mod).__name__.lower()
            if any(kw in mod_type for kw in ("fusedmoe", "sparsemlp", "moeblock")):
                return mod
            if "expert" in name and "mlp" in name:
                return mod
        return None

    # ── Buffer management ─────────────────────────────────────────────────

    def _reset_buffers(self):
        self._buffers = {}

    def _save_buffers(self, total_tokens: int) -> dict[str, Path]:
        """Concatenate buffered tensors and write to disk as .npy files."""
        paths: dict[str, Path] = {}
        for key, chunks in self._buffers.items():
            if not chunks:
                continue
            tensor = torch.cat(chunks, dim=0)
            # Flatten batch × seq → (total_positions, d_model)
            if tensor.ndim == 3:
                tensor = tensor.reshape(-1, tensor.shape[-1])
            path = self.save_dir / f"{key}.npy"
            np.save(path, tensor.numpy())
            paths[key] = path
            logger.info("activations_saved", key=key, shape=list(tensor.shape), path=str(path))
        self._reset_buffers()
        return paths

    # ── Utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def _batched(items: list, batch_size: int) -> Iterator[list]:
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    @staticmethod
    def load_activations(path: str | Path) -> torch.Tensor:
        """Load saved activations back as a torch tensor."""
        return torch.from_numpy(np.load(path))
