"""
Prometheus metrics and expert trace collector for MoE observability.
"""

import time
from contextlib import contextmanager

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CollectorRegistry,
    generate_latest,
)
import structlog

logger = structlog.get_logger(__name__)

# Custom registry to avoid conflicts with SGLang's built-in metrics
REGISTRY = CollectorRegistry()

# --- MoE Expert Metrics ---
moe_expert_activation_total = Counter(
    "creditscope_moe_expert_activation_total",
    "Total expert activations",
    ["layer", "expert_id"],
    registry=REGISTRY,
)

moe_expert_load_distribution = Histogram(
    "creditscope_moe_expert_load_distribution",
    "Distribution of tokens per expert",
    ["layer"],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
    registry=REGISTRY,
)

moe_router_entropy = Gauge(
    "creditscope_moe_router_entropy",
    "Router entropy per layer (high=uniform, low=specialized)",
    ["layer"],
    registry=REGISTRY,
)

moe_gating_weight_mean = Gauge(
    "creditscope_moe_gating_weight_mean",
    "Mean gating weight per expert per layer",
    ["layer", "expert_id"],
    registry=REGISTRY,
)

# --- Inference Metrics ---
inference_latency_seconds = Histogram(
    "creditscope_inference_latency_seconds",
    "End-to-end inference latency",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY,
)

tool_calls_total = Counter(
    "creditscope_tool_calls_total",
    "Total tool calls by tool name",
    ["tool_name"],
    registry=REGISTRY,
)

tokens_processed_total = Counter(
    "creditscope_tokens_processed_total",
    "Total tokens processed",
    ["direction"],
    registry=REGISTRY,
)

# --- Thinking / CoT Metrics ---
thinking_tokens_total = Counter(
    "creditscope_thinking_tokens_total",
    "Total thinking tokens generated",
    registry=REGISTRY,
)

thinking_budget_utilization = Histogram(
    "creditscope_thinking_budget_utilization",
    "Thinking budget utilization percentage",
    buckets=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
    registry=REGISTRY,
)

thinking_budget_enforced_total = Counter(
    "creditscope_thinking_budget_enforced_total",
    "Times budget processor forced thinking to close",
    registry=REGISTRY,
)

thinking_duration_seconds = Histogram(
    "creditscope_thinking_duration_seconds",
    "Duration of thinking phase",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY,
)

thinking_mode_requests_total = Counter(
    "creditscope_thinking_mode_requests_total",
    "Requests by thinking mode",
    ["mode"],
    registry=REGISTRY,
)

thinking_budget_preset_total = Counter(
    "creditscope_thinking_budget_preset_total",
    "Requests by thinking budget preset",
    ["preset_name"],
    registry=REGISTRY,
)


def record_moe_trace(trace) -> None:
    """Record MoE trace data into Prometheus metrics."""
    for layer_trace in trace.layer_traces:
        layer = layer_trace.layer_name

        # Expert activations
        for expert_id, count in layer_trace.expert_load.items():
            moe_expert_activation_total.labels(
                layer=layer, expert_id=str(expert_id)
            ).inc(count)
            moe_expert_load_distribution.labels(layer=layer).observe(count)

        # Router entropy
        moe_router_entropy.labels(layer=layer).set(layer_trace.entropy)

        # Mean gating weights
        if layer_trace.gating_weights is not None:
            import numpy as np
            for i, expert_id in enumerate(
                range(layer_trace.selected_experts.shape[1])
            ):
                mean_weight = float(np.mean(layer_trace.gating_weights[:, i]))
                moe_gating_weight_mean.labels(
                    layer=layer, expert_id=str(expert_id)
                ).set(mean_weight)


def record_thinking_stats(stats: dict) -> None:
    """Record thinking/CoT statistics into Prometheus metrics."""
    tokens = stats.get("thinking_tokens_used", 0)
    thinking_tokens_total.inc(tokens)

    utilization = stats.get("budget_utilization_pct")
    if utilization is not None:
        thinking_budget_utilization.observe(utilization)

    if stats.get("was_budget_enforced"):
        thinking_budget_enforced_total.inc()

    duration = stats.get("duration_ms")
    if duration is not None:
        thinking_duration_seconds.observe(duration / 1000.0)


def record_thinking_mode(mode: str, preset: str | None = None) -> None:
    """Record thinking mode selection."""
    thinking_mode_requests_total.labels(mode=mode).inc()
    if preset:
        thinking_budget_preset_total.labels(preset_name=preset).inc()


@contextmanager
def track_inference_latency():
    """Context manager to track inference latency."""
    start = time.time()
    yield
    inference_latency_seconds.observe(time.time() - start)


def record_tokens(input_tokens: int, output_tokens: int, thinking_tokens: int = 0) -> None:
    """Record token counts."""
    tokens_processed_total.labels(direction="input").inc(input_tokens)
    tokens_processed_total.labels(direction="output").inc(output_tokens)
    if thinking_tokens:
        thinking_tokens_total.inc(thinking_tokens)


def record_request_tokens(
    input_tokens: int,
    output_tokens: int,
    thinking_tokens: int = 0,
) -> None:
    """Backward-compatible wrapper for request token accounting."""
    record_tokens(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
    )


def record_tool_call(tool_name: str) -> None:
    """Record a tool call."""
    tool_calls_total.labels(tool_name=tool_name).inc()


def get_metrics() -> bytes:
    """Generate Prometheus metrics output."""
    return generate_latest(REGISTRY)
