"""
Load a HuggingFace transformer model with hook access for activation capture.

Uses nnsight when available for clean tracing, falls back to manual
PyTorch forward hooks otherwise.
"""

from __future__ import annotations

from typing import Any

import structlog
import torch

from circuit_tracer.config import get_config

logger = structlog.get_logger(__name__)


class HookedModel:
    """
    Thin wrapper around a HuggingFace model that provides:
    - Access to every decoder layer's input / output residual stream
    - Registering arbitrary forward hooks by layer index
    - Convenience methods for single-forward-pass activation capture

    This is **not** the SGLang server — it loads the model directly for
    offline analysis.  The SGLang server keeps running in parallel for
    the live app; this loader is used only by the circuit-tracing pipeline.
    """

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def load(cls, model_path: str | None = None, **kwargs) -> "HookedModel":
        """Load model + tokenizer from HuggingFace hub or local path."""
        cfg = get_config()
        path = model_path or cfg.model_path

        logger.info("loading_model_for_tracing", path=path)

        try:
            from nnsight import LanguageModel

            lm = LanguageModel(path, device_map=cfg.device_map, torch_dtype=cfg.torch_dtype)
            logger.info("model_loaded_via_nnsight")
            return cls(model=lm, tokenizer=lm.tokenizer)
        except ImportError:
            logger.info("nnsight_not_available_using_transformers")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map=cfg.device_map,
            torch_dtype=cfg.torch_dtype,
            trust_remote_code=True,
            **kwargs,
        )
        model.eval()
        return cls(model=model, tokenizer=tokenizer)

    # ── Layer access ──────────────────────────────────────────────────────

    @property
    def layers(self):
        """Return the list of decoder layers."""
        # HuggingFace standard: model.model.layers
        inner = getattr(self.model, "model", self.model)
        return inner.layers

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def embed_tokens(self):
        inner = getattr(self.model, "model", self.model)
        return inner.embed_tokens

    @property
    def lm_head(self):
        return self.model.lm_head

    # ── Hooks ─────────────────────────────────────────────────────────────

    def register_hook(self, layer_idx: int, hook_fn, hook_type: str = "forward"):
        """Register a forward (or forward_pre) hook on a decoder layer."""
        layer = self.layers[layer_idx]
        if hook_type == "forward_pre":
            h = layer.register_forward_pre_hook(hook_fn)
        else:
            h = layer.register_forward_hook(hook_fn)
        self._hooks.append(h)
        return h

    def clear_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── Tokenisation helpers ──────────────────────────────────────────────

    def tokenize(self, texts: str | list[str], max_length: int | None = None):
        """Tokenize text(s) and move to model device."""
        cfg = get_config()
        max_len = max_length or cfg.collect_max_seq_len
        if isinstance(texts, str):
            texts = [texts]
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        device = next(self.model.parameters()).device
        return {k: v.to(device) for k, v in encoded.items()}

    # ── Forward pass ──────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """Run a single forward pass and return the full model output."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def __repr__(self):
        return f"HookedModel(layers={self.num_layers}, model={type(self.model).__name__})"
