"""
Feature steering and ablation for causal validation of discovered circuits.

Provides tools to:
- Clamp a feature to a specific activation value and observe output changes
- Ablate (zero out) a set of features to measure circuit importance
- Inject features to test counterfactual predictions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
import torch

logger = structlog.get_logger(__name__)


@dataclass
class InterventionResult:
    """Result of a feature intervention experiment."""
    prompt: str
    intervention_type: str              # "clamp", "ablate", "inject"
    targets: list[dict]                 # list of {layer, position, feature_idx, value}
    baseline_output: str                # model output without intervention
    intervened_output: str              # model output with intervention
    baseline_logits: dict = field(default_factory=dict)   # top-k token logits before
    intervened_logits: dict = field(default_factory=dict)  # top-k token logits after
    logit_diff: float = 0.0            # change in target token logit


class FeatureSteering:
    """
    Manipulate individual SAE/CLT features during a forward pass
    to validate circuit hypotheses.
    """

    def __init__(self, hooked_model: Any, feature_model: Any):
        self.model = hooked_model
        self.feature_model = feature_model

    def clamp_feature(
        self,
        prompt: str,
        layer: int,
        position: int,
        feature_idx: int,
        value: float,
        max_new_tokens: int = 50,
    ) -> InterventionResult:
        """
        Set a specific feature activation to a fixed value and
        compare the model's output.
        """
        baseline = self._generate(prompt, max_new_tokens=max_new_tokens)
        intervened = self._generate_with_intervention(
            prompt,
            interventions=[{"layer": layer, "position": position, "feature_idx": feature_idx, "value": value}],
            max_new_tokens=max_new_tokens,
        )

        return InterventionResult(
            prompt=prompt,
            intervention_type="clamp",
            targets=[{"layer": layer, "position": position, "feature_idx": feature_idx, "value": value}],
            baseline_output=baseline,
            intervened_output=intervened,
        )

    def ablate_circuit(
        self,
        prompt: str,
        circuit_nodes: list[dict],
        max_new_tokens: int = 50,
    ) -> InterventionResult:
        """
        Zero out all features in a circuit and measure output degradation.

        Args:
            circuit_nodes: list of {layer, position, feature_idx}
        """
        baseline = self._generate(prompt, max_new_tokens=max_new_tokens)
        interventions = [
            {**node, "value": 0.0} for node in circuit_nodes
        ]
        ablated = self._generate_with_intervention(
            prompt,
            interventions=interventions,
            max_new_tokens=max_new_tokens,
        )

        return InterventionResult(
            prompt=prompt,
            intervention_type="ablate",
            targets=interventions,
            baseline_output=baseline,
            intervened_output=ablated,
        )

    # ── Internal ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        encoded = self.model.tokenize(prompt)
        output_ids = self.model.model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        new_tokens = output_ids[0, encoded["input_ids"].shape[1]:]
        return self.model.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @torch.no_grad()
    def _generate_with_intervention(
        self,
        prompt: str,
        interventions: list[dict],
        max_new_tokens: int = 50,
    ) -> str:
        """Generate with hooks that modify feature activations."""
        # Group interventions by layer
        layer_interventions: dict[int, list[dict]] = {}
        for iv in interventions:
            layer_interventions.setdefault(iv["layer"], []).append(iv)

        hooks = []

        for layer_idx, ivs in layer_interventions.items():
            layer = self.model.layers[layer_idx]

            def make_hook(interventions_for_layer):
                def hook_fn(module, args, output):
                    x = output[0] if isinstance(output, tuple) else output
                    if not isinstance(x, torch.Tensor):
                        return output

                    # Encode → modify features → decode back
                    residual = x.squeeze(0) if x.ndim == 3 else x
                    if hasattr(self.feature_model, "encode"):
                        z = self.feature_model.encode(residual)
                        for iv in interventions_for_layer:
                            pos = iv["position"]
                            feat = iv["feature_idx"]
                            if pos < z.shape[0] and feat < z.shape[1]:
                                z[pos, feat] = iv["value"]
                        modified = self.feature_model.decode(z)
                        if x.ndim == 3:
                            modified = modified.unsqueeze(0)
                        if isinstance(output, tuple):
                            return (modified,) + output[1:]
                        return modified

                    return output
                return hook_fn

            h = layer.register_forward_hook(make_hook(ivs))
            hooks.append(h)

        try:
            result = self._generate(prompt, max_new_tokens=max_new_tokens)
        finally:
            for h in hooks:
                h.remove()

        return result
