"""
Chain-of-Thought (CoT) controller for Qwen3.5-35B-A3B.

Manages thinking behavior at three control levels:
1. Hard switch (enable_thinking parameter)
2. Thinking budget (token-level budget enforcement)
3. Thinking visibility & streaming
"""

from __future__ import annotations

import structlog

from inference.config import (
    THINKING_BUDGET_PRESETS,
    SAMPLING_THINKING_ON,
    SAMPLING_THINKING_OFF,
)

logger = structlog.get_logger(__name__)


class CoTController:
    """
    Manages Chain-of-Thought behavior at three control levels.

    Level 1 - Hard Switch: enable_thinking parameter via chat_template_kwargs
    Level 2 - Thinking Budget: token-level budget enforcement via logits processor
    Level 3 - Thinking Visibility: controls what the banker sees of thinking
    """

    def __init__(self):
        self.default_mode = "on"
        self.default_budget = "standard"
        self.default_visibility = "collapsed"

    def build_request_params(self, cot_config: dict) -> dict:
        """
        Build SGLang-compatible request parameters from CoT configuration.

        Args:
            cot_config: Dict with keys: mode, budget, visibility

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
            params.update(SAMPLING_THINKING_OFF)
            logger.debug("cot_mode_off")
        else:
            params["chat_template_kwargs"] = {"enable_thinking": True}
            params.update(SAMPLING_THINKING_ON)
            logger.debug("cot_mode_on")

        # Level 2: Budget (resolved to token count)
        resolved_budget = self._resolve_budget(budget)
        params["_thinking_budget"] = resolved_budget
        logger.debug("cot_budget_set", budget=budget, resolved=resolved_budget)

        # Level 3: Visibility (handled by response streaming layer)
        params["_thinking_visibility"] = visibility

        return params

    def _resolve_budget(self, budget: str | int) -> int:
        """Resolve a budget preset name or raw int to a token count."""
        if isinstance(budget, int):
            return max(budget, -1)
        return THINKING_BUDGET_PRESETS.get(budget, THINKING_BUDGET_PRESETS["standard"])

    @staticmethod
    def get_presets() -> list[dict]:
        """Return all available CoT presets for the UI."""
        descriptions = {
            "none": "No thinking — equivalent to disabling CoT",
            "minimal": "128 tokens — quick sanity check only",
            "short": "512 tokens — brief reasoning",
            "standard": "2048 tokens — balanced reasoning (default)",
            "extended": "8192 tokens — complex multi-step analysis",
            "deep": "32768 tokens — maximum reasoning depth",
            "unlimited": "No limit — think until done",
        }
        return [
            {
                "name": name,
                "budget": tokens,
                "description": descriptions.get(name, ""),
            }
            for name, tokens in THINKING_BUDGET_PRESETS.items()
        ]

    @staticmethod
    def get_workflow_presets() -> list[dict]:
        """Pre-configured presets for common banking workflows."""
        return [
            {
                "name": "Quick Lookup",
                "description": "Fast response, no thinking",
                "mode": "off",
                "budget": "none",
                "visibility": "hidden",
            },
            {
                "name": "Standard Analysis",
                "description": "Balanced reasoning, collapsed view",
                "mode": "on",
                "budget": "standard",
                "visibility": "collapsed",
            },
            {
                "name": "Deep Review",
                "description": "Extended reasoning, streaming view",
                "mode": "on",
                "budget": "deep",
                "visibility": "streaming",
            },
            {
                "name": "Debug Mode",
                "description": "Unlimited thinking, full observability",
                "mode": "on",
                "budget": "unlimited",
                "visibility": "full",
            },
        ]
