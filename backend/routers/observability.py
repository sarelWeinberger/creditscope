"""
MoE observability API endpoints.
"""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter
from fastapi.responses import Response

from inference.moe_hooks import get_collector
from inference.observability import get_metrics

router = APIRouter()
INFERENCE_BASE_URL = os.getenv("SGLANG_URL", "http://127.0.0.1:8000")
PROXY_INFERENCE_OBSERVABILITY = os.getenv(
    "PROXY_INFERENCE_OBSERVABILITY", "false"
).lower() == "true"


async def _try_inference_json(path: str, params: dict | None = None) -> dict | list | None:
    if not PROXY_INFERENCE_OBSERVABILITY:
        return None

    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            response = await client.get(f"{INFERENCE_BASE_URL}{path}", params=params)
            if response.status_code == 200:
                return response.json()
    except Exception:
        return None
    return None


async def _try_inference_metrics() -> bytes | None:
    if not PROXY_INFERENCE_OBSERVABILITY:
        return None

    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            response = await client.get(f"{INFERENCE_BASE_URL}/metrics")
            if response.status_code == 200:
                return response.content
    except Exception:
        return None
    return None


@router.get("/observability/moe/latest")
async def get_latest_moe_trace():
    """Get the latest MoE trace data."""
    proxied = await _try_inference_json("/moe_trace")
    if proxied is not None:
        return proxied

    collector = get_collector()
    trace = collector.get_latest_trace()
    if not trace:
        return {"message": "No traces available yet"}

    layers = []
    for lt in trace.layer_traces:
        layers.append({
            "layer_id": lt.layer_name,
            "experts_activated": lt.selected_experts.tolist() if hasattr(lt.selected_experts, 'tolist') else [],
            "gating_weights": lt.gating_weights.tolist() if hasattr(lt.gating_weights, 'tolist') else [],
            "entropy": lt.entropy,
            "num_tokens": lt.num_tokens,
        })

    return {
        "request_id": trace.request_id,
        "timestamp": trace.timestamp,
        "layers": layers,
        "total_tokens": trace.total_tokens,
        "phase": trace.phase,
    }


@router.get("/observability/moe/heatmap")
async def get_expert_heatmap(num_requests: int = 50):
    """Get aggregated expert activation heatmap data."""
    proxied = await _try_inference_json("/moe_trace/heatmap", params={"num_requests": num_requests})
    if proxied is not None:
        return proxied

    collector = get_collector()
    return collector.get_expert_heatmap(num_requests)


@router.get("/observability/moe/entropy")
async def get_entropy_timeseries(num_requests: int = 50):
    """Get router entropy time series data."""
    proxied = await _try_inference_json("/moe_trace/entropy", params={"num_requests": num_requests})
    if proxied is not None:
        return proxied

    collector = get_collector()
    return collector.get_entropy_timeseries(num_requests)


@router.get("/observability/metrics")
async def get_prometheus_metrics():
    """Get Prometheus-format metrics."""
    proxied = await _try_inference_metrics()
    if proxied is not None:
        return Response(content=proxied, media_type="text/plain; charset=utf-8")

    metrics = get_metrics()
    return Response(content=metrics, media_type="text/plain; charset=utf-8")


@router.get("/observability/layers")
async def get_layer_activations():
    """Get per-layer activation data."""
    proxied = await _try_inference_json("/moe_trace/layers")
    if isinstance(proxied, dict) and "layers" in proxied:
        return proxied["layers"]

    collector = get_collector()
    layer_map = collector.get_layer_map()

    layers = []
    for name, layer_type in sorted(layer_map.items()):
        entry = {
            "layer_id": name,
            "layer_type": layer_type,
        }
        if layer_type == "moe":
            trace = collector.get_latest_trace()
            if trace:
                for lt in trace.layer_traces:
                    if lt.layer_name == name:
                        entry["expert_selections"] = lt.selected_experts.tolist() if hasattr(lt.selected_experts, 'tolist') else []
                        entry["entropy"] = lt.entropy
                        entry["load_distribution"] = lt.expert_load
                        entry["gating_weight_distribution"] = lt.gating_weights.tolist() if hasattr(lt.gating_weights, 'tolist') else []
                        break
        layers.append(entry)

    return layers
