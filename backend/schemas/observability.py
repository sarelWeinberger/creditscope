"""
Pydantic v2 schemas for MoE observability data.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MoELayerTrace(BaseModel):
    """Trace data for a single MoE layer."""
    layer_id: str
    layer_index: int = 0
    experts_activated: List[int] = Field(default_factory=list)
    gating_weights: List[float] = Field(default_factory=list)
    expert_load: List[float] = Field(default_factory=list)
    router_logits: Optional[List[float]] = None
    entropy: float = 0.0
    top_k: int = 2
    num_experts: int = 8

    model_config = {"from_attributes": True}


class MoERequestTrace(BaseModel):
    """Full MoE trace for a single request."""
    request_id: str
    layers: List[MoELayerTrace] = Field(default_factory=list)
    thinking_phase_experts: Dict[str, Any] = Field(default_factory=dict)
    response_phase_experts: Dict[str, Any] = Field(default_factory=dict)
    total_expert_activations: int = 0
    duration_ms: Optional[float] = None
    started_at: Optional[float] = None

    model_config = {"from_attributes": True}


class ExpertHeatmapCell(BaseModel):
    """A single cell in the expert heatmap."""
    count: int = 0
    frequency: float = 0.0
    avg_weight: float = 0.0


class ExpertHeatmapData(BaseModel):
    """Heatmap data: layers x experts matrix."""
    layers: List[str] = Field(default_factory=list)
    experts: List[int] = Field(default_factory=list)
    data: List[List[ExpertHeatmapCell]] = Field(default_factory=list)
    total_requests: int = 0

    model_config = {"from_attributes": True}


class LayerActivityData(BaseModel):
    """Activity statistics for a single layer."""
    layer_id: str
    layer_index: int = 0
    top_experts: List[Dict[str, Any]] = Field(default_factory=list)  # [{expert_id, count, weight}]
    entropy: float = 0.0
    load_distribution: List[float] = Field(default_factory=list)
    gating_weight_stats: Dict[str, float] = Field(default_factory=dict)  # {min, max, mean, std}
    activation_count: int = 0

    model_config = {"from_attributes": True}


class EntropyDataPoint(BaseModel):
    """Time-series data point for router entropy."""
    timestamp: float
    layer: str
    entropy: float
    request_id: Optional[str] = None


class MetricsSnapshot(BaseModel):
    """Current metrics snapshot."""
    total_requests: int = 0
    total_tool_calls: int = 0
    total_thinking_tokens: int = 0
    avg_inference_latency_ms: float = 0.0
    avg_thinking_duration_ms: float = 0.0
    thinking_budget_enforcement_rate: float = 0.0
    expert_activation_counts: Dict[str, int] = Field(default_factory=dict)
    token_throughput: Dict[str, float] = Field(default_factory=dict)

    model_config = {"from_attributes": True}
