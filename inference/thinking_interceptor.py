"""
Thinking token stream parser and budget enforcer for Qwen3.5-35B-A3B.

Implements:
- ThinkingBudgetProcessor: Logits processor that enforces thinking token budgets
- ThinkingStreamParser: Real-time parser that separates thinking from response tokens
"""

from __future__ import annotations

import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ThinkingBudgetProcessor:
    """
    Custom logits processor that enforces thinking token budgets.

    Mechanism:
    1. Tracks tokens generated inside <think>...</think>
    2. At 95% budget: gently biases towards wrapping up
    3. At budget limit: force-generates </think> to close thinking block
    """

    def __init__(self, tokenizer: Any, max_thinking_tokens: int):
        self.tokenizer = tokenizer
        self.max_thinking_tokens = max_thinking_tokens
        self.thinking_tokens_generated = 0
        self.in_thinking_block = False
        self.stopped_thinking = False

        # Resolve special token IDs
        self.think_start_token = self._get_token_id("<think>")
        self.think_end_token = self._get_token_id("</think>")
        self.nl_token = self._get_token_id("\n")
        self.neg_inf = float("-inf")

    def _get_token_id(self, text: str) -> int:
        """Resolve a text string to its token ID."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return ids[0] if ids else 0

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        """Process logits to enforce thinking budget."""
        if self.stopped_thinking or self.max_thinking_tokens < 0:
            return scores  # unlimited or already stopped

        # Detect thinking block entry
        if not self.in_thinking_block:
            if len(input_ids[0]) > 0 and int(input_ids[0][-1]) == self.think_start_token:
                self.in_thinking_block = True
                self.thinking_tokens_generated = 0
            return scores

        # Inside thinking block — count tokens
        self.thinking_tokens_generated += 1

        # Budget enforcement
        if self.max_thinking_tokens == 0:
            # No thinking allowed — immediately close
            scores[:] = self.neg_inf
            scores[0][self.think_end_token] = 0
            self.stopped_thinking = True

        elif self.thinking_tokens_generated >= self.max_thinking_tokens - 1:
            # At budget limit — force close
            if self.thinking_tokens_generated == self.max_thinking_tokens - 1:
                # Penultimate token: force newline for clean transition
                scores[:] = self.neg_inf
                scores[0][self.nl_token] = 0
            else:
                # Final token: force </think>
                scores[:] = self.neg_inf
                scores[0][self.think_end_token] = 0
                self.stopped_thinking = True

        elif self.thinking_tokens_generated >= int(self.max_thinking_tokens * 0.95):
            # Last 5%: gently bias towards wrapping up
            scores[0][self.nl_token] += 3.0
            scores[0][self.think_end_token] += 2.0

        return scores

    def get_thinking_stats(self) -> dict:
        """Return stats about the thinking phase for observability."""
        budget = self.max_thinking_tokens
        used = self.thinking_tokens_generated
        return {
            "thinking_tokens_used": used,
            "thinking_budget": budget,
            "budget_utilization_pct": (
                round(used / budget * 100, 1) if budget > 0 else None
            ),
            "was_budget_enforced": self.stopped_thinking,
            "mode": "unlimited" if budget < 0 else "budgeted",
        }


class ThinkingStreamParser:
    """
    Parses streaming token output to separate thinking from response content.

    SGLang with --reasoning-parser qwen3 separates thinking into
    reasoning_content. This parser adds real-time streaming, metrics,
    phase transitions, and MoE correlation.

    Output stream events:
        { "type": "thinking_start", "timestamp": ... }
        { "type": "thinking_delta", "content": "...", "token_index": N }
        { "type": "thinking_end", "stats": { tokens_used, duration_ms, budget_pct } }
        { "type": "response_start", "timestamp": ... }
        { "type": "response_delta", "content": "..." }
        { "type": "response_end", "stats": { ... } }
    """

    def __init__(self):
        self.phase: str = "idle"  # idle | thinking | response
        self.thinking_content: list[str] = []
        self.response_content: list[str] = []
        self.thinking_start_time: float | None = None
        self.thinking_end_time: float | None = None
        self.thinking_token_count: int = 0
        self.response_token_count: int = 0

    def process_chunk(self, chunk: Any) -> list[dict]:
        """Process a streaming chunk from SGLang and emit typed events."""
        events: list[dict] = []

        if not chunk.choices:
            return events

        delta = chunk.choices[0].delta

        # Handle reasoning content (thinking tokens)
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            if self.phase != "thinking":
                self.phase = "thinking"
                self.thinking_start_time = time.time()
                events.append({
                    "type": "thinking_start",
                    "timestamp": self.thinking_start_time,
                })

            self.thinking_content.append(reasoning)
            self.thinking_token_count += 1
            events.append({
                "type": "thinking_delta",
                "content": reasoning,
                "token_index": self.thinking_token_count,
            })

        # Handle response content
        content = getattr(delta, "content", None)
        if content:
            if self.phase == "thinking":
                self.thinking_end_time = time.time()
                events.append({
                    "type": "thinking_end",
                    "stats": self._thinking_stats(),
                })
                self.phase = "response"
                events.append({
                    "type": "response_start",
                    "timestamp": time.time(),
                })
            elif self.phase != "response":
                self.phase = "response"
                events.append({
                    "type": "response_start",
                    "timestamp": time.time(),
                })

            self.response_content.append(content)
            self.response_token_count += 1
            events.append({
                "type": "response_delta",
                "content": content,
            })

        # Handle finish
        finish_reason = getattr(chunk.choices[0], "finish_reason", None)
        if finish_reason:
            if self.phase == "thinking":
                self.thinking_end_time = time.time()
                events.append({
                    "type": "thinking_end",
                    "stats": self._thinking_stats(),
                })
            events.append({
                "type": "response_end",
                "stats": self._response_stats(),
            })

        return events

    def _thinking_stats(self) -> dict:
        start = self.thinking_start_time or time.time()
        end = self.thinking_end_time or time.time()
        duration = (end - start) * 1000
        return {
            "tokens_used": self.thinking_token_count,
            "duration_ms": round(duration, 1),
            "full_thinking_content": "".join(self.thinking_content),
        }

    def _response_stats(self) -> dict:
        return {
            "response_tokens": self.response_token_count,
            "full_response_content": "".join(self.response_content),
            "thinking_tokens": self.thinking_token_count,
        }

    def get_full_thinking_content(self) -> str:
        """Return the complete thinking content."""
        return "".join(self.thinking_content)

    def get_full_response_content(self) -> str:
        """Return the complete response content."""
        return "".join(self.response_content)
