"""Pydantic models for Chain-of-Thought control."""

from typing import Literal
from pydantic import BaseModel


class CoTConfig(BaseModel):
    """Per-request Chain-of-Thought configuration."""
    mode: Literal["on", "off"] = "on"
    budget: str | int = "standard"
    visibility: Literal["hidden", "collapsed", "streaming", "full"] = "collapsed"


class CoTPreset(BaseModel):
    """Named preset for quick switching."""
    name: str
    description: str
    mode: Literal["on", "off"]
    budget: str | int
    visibility: str


class ThinkingTrace(BaseModel):
    """Captured thinking data for observability."""
    thinking_content: str | None = None
    thinking_tokens_used: int = 0
    thinking_budget: int = 2048
    budget_utilization_pct: float | None = None
    was_budget_enforced: bool = False
    thinking_duration_ms: float = 0.0
    phase_moe_comparison: dict | None = None
