"""
Chain-of-Thought budget and mode control for Qwen3.5-35B-A3B.

Three control levels:
1. Hard switch: enable/disable thinking via chat_template_kwargs
2. Thinking budget: token-level budget enforcement via logits processor
3. Thinking visibility: control what the banker sees of thinking content
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

BUDGET_PRESETS: dict[str, int] = {
    "none": 0,
    "minimal": 128,
    "short": 512,
    "standard": 2048,
    "extended": 8192,
    "deep": 32768,
    "unlimited": -1,
}

# Sampling parameters per thinking mode (from Qwen3.5 documentation)
THINKING_ON_SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0,
}

THINKING_OFF_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0,
}

COT_PRESETS = {
    "quick_lookup": {
        "name": "Quick Lookup",
        "description": "Fast response with no thinking — for simple lookups",
        "mode": "off",
        "budget": "none",
        "visibility": "hidden",
    },
    "standard_analysis": {
        "name": "Standard Analysis",
        "description": "Balanced thinking for typical credit analysis",
        "mode": "on",
        "budget": "standard",
        "visibility": "collapsed",
    },
    "deep_review": {
        "name": "Deep Review",
        "description": "Extended thinking for complex or edge-case decisions",
        "mode": "on",
        "budget": "deep",
        "visibility": "streaming",
    },
    "debug_mode": {
        "name": "Debug Mode",
        "description": "Unlimited thinking with full visibility and all observability panels",
        "mode": "on",
        "budget": "unlimited",
        "visibility": "full",
    },
}


class CoTController:
    """
    Manages Chain-of-Thought behavior for each inference request.

    Builds SGLang-compatible request parameters from the banker's CoT config.
    """

    def __init__(self):
        self.default_mode = "on"
        self.default_budget = "standard"
        self.default_visibility = "collapsed"

    def build_request_params(self, cot_config: dict) -> dict:
        """
        Build SGLang-compatible request parameters from CoT configuration.

        Args:
            cot_config: Dict with keys 'mode', 'budget', 'visibility'

        Returns:
            Dict to merge into the SGLang API call.
        """
        mode = cot_config.get("mode", self.default_mode)
        budget = cot_config.get("budget", self.default_budget)
        visibility = cot_config.get("visibility", self.default_visibility)

        params: dict = {}

        # Level 1: Hard switch
        if mode == "off":
            params["chat_template_kwargs"] = {"enable_thinking": False}
            params.update(THINKING_OFF_SAMPLING)
        else:
            params["chat_template_kwargs"] = {"enable_thinking": True}
            params.update(THINKING_ON_SAMPLING)

        # Level 2: Budget (consumed by ThinkingBudgetProcessor)
        params["_thinking_budget"] = self.resolve_budget(budget)

        # Level 3: Visibility (consumed by response streaming layer)
        params["_thinking_visibility"] = visibility

        logger.info(
            "cot_params_built",
            mode=mode,
            budget=budget,
            resolved_budget=params["_thinking_budget"],
            visibility=visibility,
        )

        return params

    def resolve_budget(self, budget: str | int) -> int:
        """Resolve a budget preset name or integer to a token count."""
        if isinstance(budget, int):
            return budget
        return BUDGET_PRESETS.get(str(budget), BUDGET_PRESETS["standard"])

    def get_presets(self) -> list[dict]:
        """Return all available CoT presets."""
        return [
            {"id": k, **v}
            for k, v in COT_PRESETS.items()
        ]

    def get_preset(self, name: str) -> dict | None:
        """Return a specific preset by ID."""
        preset = COT_PRESETS.get(name)
        if preset:
            return {"id": name, **preset}
        return None

    def get_budget_presets(self) -> dict[str, int]:
        """Return all budget presets with their token counts."""
        return dict(BUDGET_PRESETS)
