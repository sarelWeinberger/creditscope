"""
MoE observability API endpoints.
"""

from __future__ import annotations

import os
from statistics import mean

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


def collect_backend_metrics() -> bytes:
    """Collect backend-local Prometheus metrics, including circuit tracing metrics when available."""
    parts = [get_metrics().decode("utf-8").strip()]

    try:
        from prometheus_client import generate_latest

        from circuit_tracer.metrics import REGISTRY as circuit_registry

        circuit_metrics = generate_latest(circuit_registry).decode("utf-8").strip()
        if circuit_metrics:
            parts.append(circuit_metrics)
    except ImportError:
        pass

    payload = "\n".join(part for part in parts if part).rstrip()
    return f"{payload}\n".encode("utf-8") if payload else b""


def _empty_heatmap(collector) -> dict:
    return {
        "heatmap": {
            "layers": [],
            "experts": list(range(getattr(collector, "_num_experts", 64))),
            "data": [],
            "total_requests": 0,
        }
    }


def _normalize_heatmap(raw: dict, total_requests: int) -> dict:
    layers = raw.get("layers") or []
    experts = raw.get("experts") or []
    rows = raw.get("data") or []
    max_total = max((max(row) for row in rows if row), default=0)
    normalized_rows = []
    for row in rows:
        normalized_row = []
        for value in row:
            frequency = (value / max_total) if max_total > 0 else 0.0
            normalized_row.append(
                {
                    "count": int(value),
                    "frequency": round(frequency, 4),
                    "avg_weight": 0.0,
                }
            )
        normalized_rows.append(normalized_row)

    return {
        "heatmap": {
            "layers": layers,
            "experts": experts,
            "data": normalized_rows,
            "total_requests": total_requests,
        }
    }


def _normalize_entropy(raw: list[dict]) -> dict:
    points = []
    for entry in raw:
        timestamp = entry.get("timestamp")
        request_id = entry.get("request_id")
        for layer, entropy in (entry.get("layers") or {}).items():
            points.append(
                {
                    "timestamp": timestamp,
                    "request_id": request_id,
                    "layer": layer,
                    "entropy": entropy,
                }
            )
    return {"data": points}


def _normalize_layers(collector) -> dict:
    trace = collector.get_latest_trace()
    if not trace or not trace.layer_traces:
        return {"layers": []}

    layers = []
    for idx, lt in enumerate(trace.layer_traces):
        top_experts = [
            {
                "expert_id": expert_id,
                "count": count,
                "frequency": round(count / max(lt.num_tokens, 1), 4),
                "avg_weight": 0.0,
            }
            for expert_id, count in sorted(
                lt.expert_load.items(), key=lambda item: item[1], reverse=True
            )[:8]
        ]
        weights = lt.gating_weights.tolist() if hasattr(lt.gating_weights, "tolist") else []
        flat_weights = [value for row in weights for value in row] if weights else []
        layers.append(
            {
                "layer_id": lt.layer_name,
                "layer_index": idx,
                "top_experts": top_experts,
                "entropy": lt.entropy,
                "load_distribution": [lt.expert_load.get(expert_id, 0) for expert_id in range(getattr(collector, "_num_experts", 64))],
                "gating_weight_stats": {
                    "mean": round(mean(flat_weights), 4) if flat_weights else 0.0,
                    "max": round(max(flat_weights), 4) if flat_weights else 0.0,
                    "min": round(min(flat_weights), 4) if flat_weights else 0.0,
                },
                "activation_count": sum(lt.expert_load.values()),
            }
        )

    return {"layers": layers}


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
        if isinstance(proxied, dict) and "heatmap" in proxied:
            return proxied
        total_requests = proxied.get("total_requests", 0) if isinstance(proxied, dict) else 0
        return _normalize_heatmap(proxied if isinstance(proxied, dict) else {}, total_requests)

    collector = get_collector()
    raw = collector.get_expert_heatmap(num_requests)
    recent = list(collector.traces)[-num_requests:]
    if not raw.get("layers"):
        return _empty_heatmap(collector)
    return _normalize_heatmap(raw, len(recent))


@router.get("/observability/moe/entropy")
async def get_entropy_timeseries(num_requests: int = 50):
    """Get router entropy time series data."""
    proxied = await _try_inference_json("/moe_trace/entropy", params={"num_requests": num_requests})
    if proxied is not None:
        if isinstance(proxied, dict) and "data" in proxied:
            return proxied
        if isinstance(proxied, list):
            return _normalize_entropy(proxied)
        return {"data": []}

    collector = get_collector()
    raw = collector.get_entropy_timeseries(num_requests)
    return _normalize_entropy(raw)


@router.get("/observability/metrics")
async def get_prometheus_metrics():
    """Get Prometheus-format metrics."""
    proxied = await _try_inference_metrics()
    if proxied is not None:
        return Response(content=proxied, media_type="text/plain; charset=utf-8")

    metrics = collect_backend_metrics()
    return Response(content=metrics, media_type="text/plain; charset=utf-8")


@router.get("/observability/layers")
async def get_layer_activations():
    """Get per-layer activation data."""
    proxied = await _try_inference_json("/moe_trace/layers")
    if isinstance(proxied, dict) and "layers" in proxied:
        return proxied
    if isinstance(proxied, list):
        return {"layers": proxied}

    collector = get_collector()
    return _normalize_layers(collector)
