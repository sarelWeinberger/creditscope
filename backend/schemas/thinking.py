"""Pydantic models for Chain-of-Thought control."""

from typing import Literal

from pydantic import BaseModel, Field


class CoTConfig(BaseModel):
    """Per-request Chain-of-Thought configuration."""
    mode: Literal["on", "off"] = "on"
    budget: str | int = "standard"
    visibility: Literal["hidden", "collapsed", "streaming", "full"] = "collapsed"
    auto: bool = False


class CoTPreset(BaseModel):
    """Named preset for quick switching."""
    name: str
    description: str
    mode: Literal["on", "off"]
    budget: str | int
    visibility: Literal["hidden", "collapsed", "streaming", "full"]


class ThinkingTrace(BaseModel):
    """Captured thinking data for observability."""
    thinking_content: str | None = None
    thinking_tokens_used: int = 0
    thinking_budget: int = 2048
    budget_utilization_pct: float | None = None
    was_budget_enforced: bool = False
    thinking_duration_ms: float = 0.0
    phase_moe_comparison: dict | None = None


class WorkflowPreset(BaseModel):
    """Quick-select workflow preset used by the banker UI."""

    name: str
    description: str
    mode: Literal["on", "off"]
    budget: str | int
    visibility: Literal["hidden", "collapsed", "streaming", "full"]


class ThinkingStatsAggregate(BaseModel):
    """Aggregate stats for all captured thinking requests."""

    total_requests: int = 0
    total_thinking_tokens: int = 0
    avg_budget_utilization: float = 0.0
    budget_enforced_count: int = 0
    requests: list[dict] = Field(default_factory=list)
