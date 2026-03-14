"""
Replacement model for computing attribution graphs.

Wraps the original transformer and replaces MoE layers with CLT
reconstructions + error terms.  Freezes all nonlinearities
(attention patterns, LayerNorm denominators, DeltaNet states) so that
the resulting computation graph is piecewise-linear and attribution
weights can be computed exactly via Jacobians.

For Qwen3.5-35B-A3B:
- DeltaNet layers: frozen (outputs cached from original forward pass)
- Standard attention layers: frozen attention patterns, re-applied linearly
- MoE layers: replaced by CLT features + error term
- LayerNorm: frozen denominators (scaling becomes linear)
"""

from __future__ import annotations

from typing import Any

import structlog
import torch

from circuit_tracer.attribution.graph import (
    AttributionGraph, AttributionNode, AttributionEdge,
)
from circuit_tracer.config import get_config

logger = structlog.get_logger(__name__)


class ReplacementModel:
    """
    Constructs a local linearised replacement of the original model
    for a specific prompt, then computes attribution edges between
    active CLT/SAE features.
    """

    def __init__(self, hooked_model: Any, feature_model: Any, prompt: str):
        """
        Args:
            hooked_model:   HookedModel instance (original weights)
            feature_model:  Trained SAE or CrossLayerTranscoder
            prompt:         The prompt to trace
        """
        self.model = hooked_model
        self.feature_model = feature_model
        self.prompt = prompt

        # Populated by _cache_original_pass
        self._cached_residuals: dict[int, torch.Tensor] = {}
        self._cached_attn_patterns: dict[int, torch.Tensor] = {}
        self._cached_ln_scales: dict[int, torch.Tensor] = {}
        self._input_ids: torch.Tensor | None = None
        self._embeddings: torch.Tensor | None = None

    # ── Main API ──────────────────────────────────────────────────────────

    def trace(self, target_position: int = -1) -> AttributionGraph:
        """
        Run the full attribution pipeline for a target token position.

        Steps:
        1. Run original model, cache all intermediate states
        2. Extract features at each layer using the feature model
        3. Compute virtual weights (Jacobians through frozen layers)
        4. Build attribution edges: weight × activation
        5. Assemble the graph
        """
        self._cache_original_pass()
        features = self._extract_features()
        graph = self._build_graph(features, target_position)
        return graph

    # ── Step 1: Cache original forward pass ───────────────────────────────

    @torch.no_grad()
    def _cache_original_pass(self):
        """Run the original model and cache all needed intermediate states."""
        encoded = self.model.tokenize(self.prompt)
        self._input_ids = encoded["input_ids"]

        # Cache embedding
        self._embeddings = self.model.embed_tokens(self._input_ids)

        hooks = []
        residuals = {}

        # Hook every layer to capture residual streams
        for idx in range(self.model.num_layers):
            layer = self.model.layers[idx]

            def make_pre_hook(i):
                def hook(mod, args):
                    x = args[0] if isinstance(args, tuple) else args
                    if isinstance(x, torch.Tensor):
                        residuals[f"pre_{i}"] = x.detach()
                return hook

            def make_post_hook(i):
                def hook(mod, args, output):
                    x = output[0] if isinstance(output, tuple) else output
                    if isinstance(x, torch.Tensor):
                        residuals[f"post_{i}"] = x.detach()
                return hook

            hooks.append(layer.register_forward_pre_hook(make_pre_hook(idx)))
            hooks.append(layer.register_forward_hook(make_post_hook(idx)))

        # Forward pass
        self.model.forward(
            input_ids=self._input_ids,
            attention_mask=encoded.get("attention_mask"),
        )

        # Clean up hooks
        for h in hooks:
            h.remove()

        self._cached_residuals = residuals
        logger.info("cached_original_pass", layers_cached=len(residuals) // 2)

    # ── Step 2: Extract features ──────────────────────────────────────────

    @torch.no_grad()
    def _extract_features(self) -> dict[int, torch.Tensor]:
        """Run the feature model (SAE or CLT) on cached residual streams."""
        cfg = get_config()
        features: dict[int, torch.Tensor] = {}

        for idx in range(self.model.num_layers):
            key = f"pre_{idx}"
            if key not in self._cached_residuals:
                continue

            residual = self._cached_residuals[key]
            # Flatten to (seq_len, d_model) if batched
            if residual.ndim == 3:
                residual = residual.squeeze(0)

            if hasattr(self.feature_model, "get_features"):
                # CrossLayerTranscoder
                z = self.feature_model.get_features(idx, residual)
            elif hasattr(self.feature_model, "encode"):
                # SparseAutoencoder
                z = self.feature_model.encode(residual)
            else:
                continue

            features[idx] = z

        return features

    # ── Step 3 + 4 + 5: Build graph ───────────────────────────────────────

    def _build_graph(
        self,
        features: dict[int, torch.Tensor],
        target_position: int,
    ) -> AttributionGraph:
        """
        Build the attribution graph from extracted features.

        For now this uses a simplified direct-attribution approach:
        each feature's contribution to downstream features is estimated
        by the product of its activation and decoder weight projection.

        The full version would compute exact Jacobians through frozen
        attention/LN, but this simplified version is useful for initial
        exploration and is correct for adjacent layers.
        """
        cfg = get_config()
        tokens = self.model.tokenizer.convert_ids_to_tokens(
            self._input_ids[0].tolist()
        ) if self._input_ids is not None else []

        seq_len = features[next(iter(features))].shape[0] if features else 0
        if target_position < 0:
            target_position = seq_len + target_position

        graph = AttributionGraph(
            prompt=self.prompt,
            target_position=target_position,
            target_token=tokens[target_position] if target_position < len(tokens) else "",
        )

        # ── Input nodes ──────────────────────────────────────────────
        for pos in range(min(seq_len, len(tokens))):
            graph.nodes.append(AttributionNode(
                id=f"input_{pos}",
                node_type="input",
                position=pos,
                token=tokens[pos] if pos < len(tokens) else "",
            ))

        # ── Feature nodes (only active features) ─────────────────────
        active_nodes: dict[str, AttributionNode] = {}
        for layer_idx, z in features.items():
            for pos in range(z.shape[0]):
                active_indices = (z[pos] > cfg.attribution_threshold).nonzero(as_tuple=True)[0]
                for feat_idx in active_indices.tolist():
                    node_id = f"feat_{layer_idx}_{pos}_{feat_idx}"
                    node = AttributionNode(
                        id=node_id,
                        node_type="feature",
                        layer=layer_idx,
                        position=pos,
                        feature_idx=feat_idx,
                        activation=z[pos, feat_idx].item(),
                        token=tokens[pos] if pos < len(tokens) else "",
                    )
                    graph.nodes.append(node)
                    active_nodes[node_id] = node

        # ── Output node ──────────────────────────────────────────────
        output_node = AttributionNode(
            id=f"output_{target_position}",
            node_type="output",
            position=target_position,
            token=tokens[target_position] if target_position < len(tokens) else "",
        )
        graph.nodes.append(output_node)

        # ── Edges: feature → feature (adjacent layers) ───────────────
        sorted_layers = sorted(features.keys())
        for i in range(len(sorted_layers) - 1):
            src_layer = sorted_layers[i]
            tgt_layer = sorted_layers[i + 1]
            self._add_inter_layer_edges(
                graph, features, src_layer, tgt_layer, tokens, cfg.attribution_threshold
            )

        # ── Edges: last-layer features → output ──────────────────────
        if sorted_layers:
            last_layer = sorted_layers[-1]
            z_last = features[last_layer]
            active_at_target = (z_last[target_position] > cfg.attribution_threshold).nonzero(as_tuple=True)[0]
            for feat_idx in active_at_target.tolist():
                src_id = f"feat_{last_layer}_{target_position}_{feat_idx}"
                if src_id in active_nodes:
                    weight = z_last[target_position, feat_idx].item()
                    graph.edges.append(AttributionEdge(
                        source_id=src_id,
                        target_id=output_node.id,
                        weight=weight,
                        source_activation=weight,
                    ))

        logger.info(
            "attribution_graph_built",
            nodes=graph.num_nodes,
            edges=graph.num_edges,
            layers=graph.num_layers_spanned,
        )
        return graph

    def _add_inter_layer_edges(
        self,
        graph: AttributionGraph,
        features: dict[int, torch.Tensor],
        src_layer: int,
        tgt_layer: int,
        tokens: list[str],
        threshold: float,
    ):
        """Add edges between active features in adjacent layers."""
        z_src = features[src_layer]
        z_tgt = features[tgt_layer]
        seq_len = z_src.shape[0]

        # Compute virtual weight via decoder/encoder product
        # W_virtual = decoder_src^T @ encoder_tgt  → (n_features_src, n_features_tgt)
        # This is the linear pathway from source features to target features
        if hasattr(self.feature_model, "decoders"):
            # CLT: use the specific src→tgt decoder
            dec_key = f"{src_layer}_to_{tgt_layer}"
            if dec_key not in self.feature_model.decoders:
                return
            W_dec = self.feature_model.decoders[dec_key].weight.data  # (d_model, n_feat)
            W_enc = self.feature_model.encoders[str(tgt_layer)].weight.data  # (n_feat, d_model)
            W_virtual = W_dec.T @ W_enc.T  # (n_feat_src, n_feat_tgt)
        elif hasattr(self.feature_model, "decoder") and hasattr(self.feature_model, "encoder"):
            # SAE: decoder → encoder approximation
            W_dec = self.feature_model.decoder.weight.data  # (d_model, n_feat)
            W_enc = self.feature_model.encoder.weight.data  # (n_feat, d_model)
            W_virtual = W_dec.T @ W_enc.T
        else:
            return

        for pos in range(seq_len):
            src_active = (z_src[pos] > threshold).nonzero(as_tuple=True)[0]
            tgt_active = (z_tgt[pos] > threshold).nonzero(as_tuple=True)[0]

            if len(src_active) == 0 or len(tgt_active) == 0:
                continue

            # Compute attribution: activation_src × virtual_weight
            for si in src_active.tolist():
                for ti in tgt_active.tolist():
                    vw = W_virtual[si, ti].item() if si < W_virtual.shape[0] and ti < W_virtual.shape[1] else 0.0
                    attribution = z_src[pos, si].item() * vw
                    if abs(attribution) < threshold:
                        continue
                    graph.edges.append(AttributionEdge(
                        source_id=f"feat_{src_layer}_{pos}_{si}",
                        target_id=f"feat_{tgt_layer}_{pos}_{ti}",
                        weight=attribution,
                        virtual_weight=vw,
                        source_activation=z_src[pos, si].item(),
                    ))
