"""
Pydantic v2 schemas for Chain-of-Thought / thinking configuration and traces.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CoTConfig(BaseModel):
    """Configuration for a thinking request."""
    mode: str = Field("auto", description="Thinking mode: on/off/auto")
    budget: str = Field("standard", description="Budget preset or token count")
    visibility: str = Field(
        "collapsed", description="UI visibility: hidden/collapsed/streaming/full"
    )
    enable_thinking: bool = True
    auto_classify: bool = False  # Let AdaptiveCoTStrategy pick the budget

    model_config = {"from_attributes": True}


class CoTPreset(BaseModel):
    """A named thinking preset."""
    name: str
    description: str
    mode: str  # on/off/auto
    budget: str  # preset name or token count
    visibility: str = "collapsed"
    budget_tokens: int = 0
    latency_impact: str = "medium"

    model_config = {"from_attributes": True}


class ThinkingTrace(BaseModel):
    """Full thinking trace for a single request."""
    request_id: str
    session_id: str
    mode: str
    budget_preset: str
    budget_tokens: int

    thinking_content: str = ""
    thinking_tokens_used: int = 0
    thinking_duration_ms: Optional[float] = None

    response_content: str = ""
    response_tokens: int = 0

    was_budget_enforced: bool = False
    utilization_pct: float = 0.0

    # Phase comparison
    thinking_phase_experts: Dict[str, Any] = Field(default_factory=dict)
    response_phase_experts: Dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"from_attributes": True}


class ThinkingStats(BaseModel):
    """Aggregated thinking statistics."""
    total_requests: int = 0
    thinking_on_count: int = 0
    thinking_off_count: int = 0
    avg_thinking_tokens: float = 0.0
    avg_budget_utilization_pct: float = 0.0
    budget_enforced_count: int = 0
    avg_thinking_duration_ms: float = 0.0
    preset_distribution: Dict[str, int] = Field(default_factory=dict)
    mode_distribution: Dict[str, int] = Field(default_factory=dict)

    model_config = {"from_attributes": True}


class PhaseComparisonResponse(BaseModel):
    """Comparison of expert activations between thinking and response phases."""
    request_id: str
    thinking_phase: Dict[str, Any] = Field(default_factory=dict)
    response_phase: Dict[str, Any] = Field(default_factory=dict)
    divergence_score: float = 0.0
    common_experts: List[int] = Field(default_factory=list)
    thinking_only_experts: List[int] = Field(default_factory=list)
    response_only_experts: List[int] = Field(default_factory=list)
    analysis: str = ""

    model_config = {"from_attributes": True}
