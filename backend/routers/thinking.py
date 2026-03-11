"""
CoT thinking mode control endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from inference.cot_controller import CoTController

router = APIRouter()

_controller = CoTController()

# In-memory store for custom presets and per-request stats
_custom_presets: list[dict] = []
_request_stats: dict[str, dict] = {}


class PresetCreate(BaseModel):
    name: str
    description: str
    mode: str = "on"
    budget: str | int = "standard"
    visibility: str = "collapsed"


@router.get("/thinking/presets")
async def list_presets():
    """List all CoT presets (budget presets + workflow presets + custom)."""
    return {
        "budget_presets": _controller.get_presets(),
        "workflow_presets": _controller.get_workflow_presets(),
        "custom_presets": _custom_presets,
    }


@router.get("/thinking/presets/{name}")
async def get_preset(name: str):
    """Get a specific preset by name."""
    for p in _controller.get_presets():
        if p["name"] == name:
            return p
    for p in _controller.get_workflow_presets():
        if p["name"] == name:
            return p
    for p in _custom_presets:
        if p["name"] == name:
            return p
    return {"error": f"Preset '{name}' not found"}


@router.post("/thinking/presets")
async def create_preset(preset: PresetCreate):
    """Create a custom thinking preset."""
    entry = preset.model_dump()
    _custom_presets.append(entry)
    return {"success": True, "preset": entry}


@router.get("/thinking/stats")
async def get_thinking_stats():
    """Get aggregate thinking stats across the session."""
    if not _request_stats:
        return {
            "total_requests": 0,
            "total_thinking_tokens": 0,
            "avg_budget_utilization": 0,
            "budget_enforced_count": 0,
        }

    stats = list(_request_stats.values())
    total = len(stats)
    total_tokens = sum(s.get("thinking_tokens_used", 0) for s in stats)
    utilizations = [s.get("budget_utilization_pct", 0) for s in stats if s.get("budget_utilization_pct") is not None]
    enforced = sum(1 for s in stats if s.get("was_budget_enforced"))

    return {
        "total_requests": total,
        "total_thinking_tokens": total_tokens,
        "avg_budget_utilization": round(sum(utilizations) / len(utilizations), 1) if utilizations else 0,
        "budget_enforced_count": enforced,
        "requests": stats,
    }


@router.get("/thinking/stats/{request_id}")
async def get_request_thinking_stats(request_id: str):
    """Get thinking stats for a specific request."""
    stats = _request_stats.get(request_id)
    if not stats:
        return {"error": f"No stats found for request {request_id}"}
    return stats


@router.get("/thinking/phase-comparison/{request_id}")
async def get_phase_comparison(request_id: str):
    """Get MoE expert comparison between thinking and response phases."""
    from inference.moe_hooks import get_collector
    collector = get_collector()

    # Look for traces matching this request
    thinking_traces = []
    response_traces = []

    for trace in collector.traces:
        if trace.request_id == request_id:
            if trace.phase == "thinking":
                thinking_traces.append(trace)
            elif trace.phase == "response":
                response_traces.append(trace)

    if not thinking_traces and not response_traces:
        return {"error": f"No phase data for request {request_id}"}

    def summarize_traces(traces):
        if not traces:
            return None
        expert_freq: dict[int, int] = {}
        for t in traces:
            for lt in t.layer_traces:
                for eid, count in lt.expert_load.items():
                    expert_freq[eid] = expert_freq.get(eid, 0) + count
        return {
            "expert_frequency": expert_freq,
            "total_layers": sum(len(t.layer_traces) for t in traces),
        }

    thinking_summary = summarize_traces(thinking_traces)
    response_summary = summarize_traces(response_traces)

    # Calculate overlap
    overlap = None
    if thinking_summary and response_summary:
        t_experts = set(thinking_summary["expert_frequency"].keys())
        r_experts = set(response_summary["expert_frequency"].keys())
        if t_experts or r_experts:
            overlap = len(t_experts & r_experts) / len(t_experts | r_experts) * 100

    return {
        "request_id": request_id,
        "thinking_phase": thinking_summary,
        "response_phase": response_summary,
        "expert_overlap_pct": round(overlap, 1) if overlap is not None else None,
    }


def record_request_stats(request_id: str, stats: dict):
    """Store thinking stats for a request (called from agent)."""
    _request_stats[request_id] = stats
