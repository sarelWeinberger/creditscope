"""
Thinking/CoT configuration and stats router.
"""
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from schemas.thinking import CoTConfig, CoTPreset, PhaseComparisonResponse, ThinkingStats, ThinkingTrace

router = APIRouter(prefix="/thinking", tags=["thinking"])

# ─── In-memory storage ────────────────────────────────────────────────────────

_thinking_traces: Dict[str, ThinkingTrace] = {}  # request_id -> trace
_session_traces: Dict[str, List[str]] = defaultdict(list)  # session_id -> [request_ids]
_custom_presets: Dict[str, CoTPreset] = {}  # name -> preset

# ─── Built-in presets ─────────────────────────────────────────────────────────

BUILTIN_PRESETS: Dict[str, CoTPreset] = {
    "none": CoTPreset(
        name="none",
        description="No thinking — direct response. Fastest, minimal latency.",
        mode="off",
        budget="none",
        visibility="hidden",
        budget_tokens=0,
        latency_impact="minimal",
    ),
    "minimal": CoTPreset(
        name="minimal",
        description="256 token thinking budget. Quick sanity check pass.",
        mode="on",
        budget="minimal",
        visibility="collapsed",
        budget_tokens=256,
        latency_impact="low",
    ),
    "short": CoTPreset(
        name="short",
        description="512 token budget. Brief reasoning for simple queries.",
        mode="on",
        budget="short",
        visibility="collapsed",
        budget_tokens=512,
        latency_impact="low",
    ),
    "standard": CoTPreset(
        name="standard",
        description="1024 token budget. Balanced analysis for most credit queries.",
        mode="on",
        budget="standard",
        visibility="collapsed",
        budget_tokens=1024,
        latency_impact="medium",
    ),
    "extended": CoTPreset(
        name="extended",
        description="2048 token budget. Thorough multi-factor reasoning.",
        mode="on",
        budget="extended",
        visibility="streaming",
        budget_tokens=2048,
        latency_impact="medium-high",
    ),
    "deep": CoTPreset(
        name="deep",
        description="4096 token budget. Complex analysis, stress testing, regulatory review.",
        mode="on",
        budget="deep",
        visibility="streaming",
        budget_tokens=4096,
        latency_impact="high",
    ),
    "unlimited": CoTPreset(
        name="unlimited",
        description="No budget cap. Full extended thinking for maximum accuracy.",
        mode="on",
        budget="unlimited",
        visibility="full",
        budget_tokens=-1,
        latency_impact="very-high",
    ),
}


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/presets", response_model=List[CoTPreset])
async def list_presets():
    """List all available thinking presets (built-in + custom)."""
    all_presets = {**BUILTIN_PRESETS, **_custom_presets}
    return list(all_presets.values())


@router.get("/presets/{name}", response_model=CoTPreset)
async def get_preset(name: str):
    """Get a specific preset by name."""
    all_presets = {**BUILTIN_PRESETS, **_custom_presets}
    preset = all_presets.get(name)
    if not preset:
        raise HTTPException(status_code=404, detail=f"Preset '{name}' not found")
    return preset


@router.post("/presets", response_model=CoTPreset)
async def create_preset(preset: CoTPreset):
    """Create a custom thinking preset."""
    if preset.name in BUILTIN_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot override built-in preset '{preset.name}'",
        )
    _custom_presets[preset.name] = preset
    return preset


@router.delete("/presets/{name}")
async def delete_preset(name: str):
    """Delete a custom preset."""
    if name in BUILTIN_PRESETS:
        raise HTTPException(status_code=400, detail="Cannot delete built-in presets")
    if name not in _custom_presets:
        raise HTTPException(status_code=404, detail=f"Custom preset '{name}' not found")
    del _custom_presets[name]
    return {"status": "deleted", "name": name}


@router.get("/stats", response_model=ThinkingStats)
async def get_thinking_stats():
    """Get aggregated thinking statistics across all sessions."""
    total = len(_thinking_traces)
    if total == 0:
        return ThinkingStats()

    on_count = sum(1 for t in _thinking_traces.values() if t.mode == "on")
    off_count = total - on_count

    thinking_only = [t for t in _thinking_traces.values() if t.mode == "on"]
    avg_tokens = (
        sum(t.thinking_tokens_used for t in thinking_only) / len(thinking_only)
        if thinking_only else 0.0
    )
    avg_util = (
        sum(t.utilization_pct for t in thinking_only) / len(thinking_only)
        if thinking_only else 0.0
    )
    enforced = sum(1 for t in thinking_only if t.was_budget_enforced)
    durations = [t.thinking_duration_ms for t in thinking_only if t.thinking_duration_ms]
    avg_duration = sum(durations) / len(durations) if durations else 0.0

    preset_dist: Dict[str, int] = defaultdict(int)
    mode_dist: Dict[str, int] = defaultdict(int)
    for t in _thinking_traces.values():
        preset_dist[t.budget_preset] += 1
        mode_dist[t.mode] += 1

    return ThinkingStats(
        total_requests=total,
        thinking_on_count=on_count,
        thinking_off_count=off_count,
        avg_thinking_tokens=round(avg_tokens, 1),
        avg_budget_utilization_pct=round(avg_util, 1),
        budget_enforced_count=enforced,
        avg_thinking_duration_ms=round(avg_duration, 1),
        preset_distribution=dict(preset_dist),
        mode_distribution=dict(mode_dist),
    )


@router.get("/stats/{request_id}", response_model=ThinkingTrace)
async def get_request_thinking(request_id: str):
    """Get thinking trace for a specific request."""
    trace = _thinking_traces.get(request_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"No thinking trace for request '{request_id}'")
    return trace


@router.post("/stats/{request_id}")
async def record_thinking_trace(request_id: str, trace: ThinkingTrace):
    """Record a thinking trace (called by the agent after processing)."""
    trace.request_id = request_id
    _thinking_traces[request_id] = trace
    _session_traces[trace.session_id].append(request_id)
    return {"status": "recorded", "request_id": request_id}


@router.get("/phase-comparison/{request_id}", response_model=PhaseComparisonResponse)
async def get_phase_comparison(request_id: str):
    """
    Compare expert activations between thinking and response phases.
    Requires MoE data from the sidecar.
    """
    import httpx
    import os

    sidecar_url = os.environ.get("SIDECAR_URL", "http://localhost:8001")

    # Fetch trace from sidecar
    moe_trace = None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{sidecar_url}/moe/latest", params={"request_id": request_id}
            )
            if resp.status_code == 200:
                data = resp.json()
                moe_trace = data.get("trace")
    except Exception:
        pass

    if not moe_trace:
        # Try to get thinking trace for phase info
        thinking_trace = _thinking_traces.get(request_id)
        if thinking_trace:
            return PhaseComparisonResponse(
                request_id=request_id,
                thinking_phase=thinking_trace.thinking_phase_experts,
                response_phase=thinking_trace.response_phase_experts,
                analysis="MoE sidecar data not available; using cached thinking trace.",
            )
        raise HTTPException(
            status_code=404,
            detail=f"No MoE data found for request '{request_id}'",
        )

    # Parse phase data from trace
    thinking_experts: set = set()
    response_experts: set = set()

    for key, value in moe_trace.items():
        if isinstance(value, list):
            for event in value:
                if isinstance(event, dict):
                    selected = event.get("selected_experts", [])
                    flat = []
                    for s in selected:
                        if isinstance(s, list):
                            flat.extend(s)
                        else:
                            flat.append(s)
                    if "thinking" in key.lower():
                        thinking_experts.update(int(e) for e in flat)
                    else:
                        response_experts.update(int(e) for e in flat)

    common = sorted(thinking_experts & response_experts)
    think_only = sorted(thinking_experts - response_experts)
    response_only = sorted(response_experts - thinking_experts)

    total = len(thinking_experts | response_experts) or 1
    divergence = len(set(think_only) | set(response_only)) / total

    if divergence < 0.2:
        analysis = "High overlap between thinking and response phase experts. Consistent reasoning pathway."
    elif divergence < 0.5:
        analysis = "Moderate divergence between thinking and response phases. Some specialization observed."
    else:
        analysis = "High divergence between phases. Thinking used significantly different expert circuits than response generation."

    return PhaseComparisonResponse(
        request_id=request_id,
        thinking_phase={"experts": sorted(thinking_experts)},
        response_phase={"experts": sorted(response_experts)},
        divergence_score=round(divergence, 4),
        common_experts=common,
        thinking_only_experts=think_only,
        response_only_experts=response_only,
        analysis=analysis,
    )


@router.get("/session/{session_id}/history")
async def get_session_thinking_history(session_id: str):
    """Get all thinking traces for a session."""
    request_ids = _session_traces.get(session_id, [])
    traces = [_thinking_traces[rid] for rid in request_ids if rid in _thinking_traces]
    return {
        "session_id": session_id,
        "traces": traces,
        "count": len(traces),
    }
