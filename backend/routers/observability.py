"""
Observability router: MoE trace and metrics endpoints.
"""
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse

router = APIRouter(prefix="/observability", tags=["observability"])

SIDECAR_URL = os.environ.get("SIDECAR_URL", "http://localhost:8001")

# In-memory time-series buffer for entropy data (max 1000 points)
import collections
_entropy_history: collections.deque = collections.deque(maxlen=1000)
_layer_data_cache: Dict[str, Any] = {}
_last_layer_update: float = 0.0


async def _call_sidecar(path: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    """Call the MoE sidecar server."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{SIDECAR_URL}{path}")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return None


@router.get("/moe/latest")
async def get_latest_moe_trace(request_id: Optional[str] = Query(None)):
    """Get the most recent MoE trace from the sidecar."""
    params = {}
    if request_id:
        params["request_id"] = request_id

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{SIDECAR_URL}/moe/latest", params=params)
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        # Return empty trace if sidecar unavailable
        pass

    return {"trace": None, "status": "sidecar_unavailable"}


@router.get("/moe/heatmap")
async def get_moe_heatmap():
    """Get expert activation heatmap data."""
    data = await _call_sidecar("/moe/heatmap")
    if data is None:
        # Return mock data structure if sidecar not running
        return {
            "heatmap": {
                "layers": [],
                "experts": [],
                "data": [],
                "total_requests": 0,
            },
            "status": "sidecar_unavailable",
        }
    return data


@router.get("/moe/entropy")
async def get_moe_entropy(
    window: int = Query(100, ge=1, le=1000, description="Number of recent data points to return"),
):
    """Get router entropy time-series data."""
    # Fetch latest from sidecar
    data = await _call_sidecar("/moe/heatmap")
    if data and "heatmap" in data:
        heatmap = data["heatmap"]
        layers = heatmap.get("layers", [])
        layer_data = heatmap.get("data", [])

        # Compute entropy approximation from load distribution
        import math
        timestamp = time.time()

        for i, layer in enumerate(layers):
            if i < len(layer_data):
                row = layer_data[i]
                freqs = [cell.get("frequency", 0) for cell in row]
                total = sum(freqs) or 1
                probs = [f / total for f in freqs]
                entropy = -sum(p * math.log2(p + 1e-10) for p in probs if p > 0)

                _entropy_history.append(
                    {"timestamp": timestamp, "layer": layer, "entropy": round(entropy, 4)}
                )

    recent = list(_entropy_history)[-window:]
    return {"data": recent, "count": len(recent)}


@router.get("/metrics")
async def get_prometheus_metrics():
    """Return Prometheus metrics text."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{SIDECAR_URL.replace(':8001', ':8000')}/metrics")
            if resp.status_code == 200:
                return PlainTextResponse(resp.text)
    except Exception:
        pass

    # Fallback: return local metrics if available
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from inference.observability import metrics_collector
        return PlainTextResponse(
            metrics_collector.get_metrics_text(),
            media_type="text/plain; version=0.0.4",
        )
    except Exception:
        return PlainTextResponse("# Metrics unavailable\n")


@router.get("/layers")
async def get_layer_activity(
    top_k: int = Query(5, ge=1, le=20, description="Top-K experts to return per layer"),
):
    """Get per-layer expert activity statistics."""
    global _last_layer_update, _layer_data_cache

    now = time.time()
    cache_ttl = 2.0  # seconds

    if now - _last_layer_update < cache_ttl and _layer_data_cache:
        return _layer_data_cache

    data = await _call_sidecar("/moe/heatmap")
    if not data or "heatmap" not in data:
        return {"layers": [], "status": "sidecar_unavailable"}

    heatmap = data["heatmap"]
    layers = heatmap.get("layers", [])
    experts = heatmap.get("experts", [])
    matrix = heatmap.get("data", [])

    import math
    layer_activities = []

    for i, layer in enumerate(layers):
        if i >= len(matrix):
            continue
        row = matrix[i]

        # Build expert stats
        expert_stats = []
        for j, cell in enumerate(row):
            expert_id = experts[j] if j < len(experts) else j
            expert_stats.append(
                {
                    "expert_id": expert_id,
                    "count": cell.get("count", 0),
                    "frequency": cell.get("frequency", 0.0),
                    "avg_weight": cell.get("avg_weight", 0.0),
                }
            )

        # Sort by activation count
        expert_stats.sort(key=lambda x: x["count"], reverse=True)
        top_experts = expert_stats[:top_k]

        # Load distribution
        freqs = [e["frequency"] for e in expert_stats]
        total = sum(freqs) or 1
        probs = [f / total for f in freqs]
        entropy = -sum(p * math.log2(p + 1e-10) for p in probs if p > 0)

        # Weight stats
        weights = [e["avg_weight"] for e in expert_stats if e["avg_weight"] > 0]
        if weights:
            weight_stats = {
                "min": round(min(weights), 4),
                "max": round(max(weights), 4),
                "mean": round(sum(weights) / len(weights), 4),
            }
        else:
            weight_stats = {}

        layer_activities.append(
            {
                "layer_id": layer,
                "layer_index": i,
                "top_experts": top_experts,
                "entropy": round(entropy, 4),
                "load_distribution": freqs,
                "gating_weight_stats": weight_stats,
                "activation_count": sum(e["count"] for e in expert_stats),
            }
        )

    result = {
        "layers": layer_activities,
        "total_layers": len(layer_activities),
        "total_experts": len(experts),
    }

    _layer_data_cache = result
    _last_layer_update = now
    return result


@router.get("/stats")
async def get_observability_stats():
    """Get aggregated observability statistics."""
    sidecar_data = await _call_sidecar("/moe/stats")

    return {
        "sidecar_connected": sidecar_data is not None,
        "sidecar_stats": sidecar_data,
        "entropy_buffer_size": len(_entropy_history),
    }
