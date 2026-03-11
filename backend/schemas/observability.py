"""Pydantic models for MoE observability and thinking trace responses."""

from pydantic import BaseModel


class MoELayerTraceResponse(BaseModel):
    layer_id: str
    experts_activated: list[int]
    gating_weights: list[float]
    entropy: float
    num_tokens: int


class MoETraceResponse(BaseModel):
    request_id: str
    timestamp: float
    layers: list[MoELayerTraceResponse]
    thinking_phase_experts: list[MoELayerTraceResponse] | None = None
    response_phase_experts: list[MoELayerTraceResponse] | None = None
    total_tokens: int


class ExpertHeatmapResponse(BaseModel):
    layers: list[str]
    experts: list[int]
    data: list[list[int]]


class EntropyTimeseriesEntry(BaseModel):
    request_id: str
    timestamp: float
    layers: dict[str, float]


class LayerActivationResponse(BaseModel):
    layer_id: str
    layer_type: str  # "dense" or "moe"
    expert_selections: list[int] | None = None
    entropy: float | None = None
    load_distribution: dict[int, int] | None = None
    gating_weight_distribution: list[float] | None = None


class ThinkingStatsResponse(BaseModel):
    thinking_tokens_used: int
    thinking_budget: int
    budget_utilization_pct: float | None = None
    was_budget_enforced: bool
    thinking_duration_ms: float
    mode: str


class PhaseComparisonResponse(BaseModel):
    request_id: str
    thinking_phase: list[MoELayerTraceResponse] | None = None
    response_phase: list[MoELayerTraceResponse] | None = None
    expert_overlap_pct: float | None = None
    specialization_score: float | None = None
