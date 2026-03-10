"""
Prometheus metrics and expert trace collector for MoE observability.

Exposes metrics for:
- Expert activation patterns across MoE layers
- Router entropy (routing decisiveness)
- Inference latency
- Tool call frequency
- Token throughput
- Thinking/CoT metrics
"""

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()

# --- MoE Expert Metrics ---
MOE_EXPERT_ACTIVATION = Counter(
    "creditscope_moe_expert_activation_total",
    "Total expert activations",
    ["layer", "expert_id"],
    registry=REGISTRY,
)

MOE_EXPERT_LOAD = Histogram(
    "creditscope_moe_expert_load_distribution",
    "Distribution of tokens per expert within a layer",
    ["layer"],
    buckets=[0, 1, 5, 10, 25, 50, 100, 250, 500, 1000],
    registry=REGISTRY,
)

MOE_ROUTER_ENTROPY = Gauge(
    "creditscope_moe_router_entropy",
    "Shannon entropy of the router distribution (high=uniform, low=specialized)",
    ["layer"],
    registry=REGISTRY,
)

MOE_GATING_WEIGHT_MEAN = Gauge(
    "creditscope_moe_gating_weight_mean",
    "Mean gating weight for an expert in a layer",
    ["layer", "expert_id"],
    registry=REGISTRY,
)

# --- Inference Metrics ---
INFERENCE_LATENCY = Histogram(
    "creditscope_inference_latency_seconds",
    "End-to-end inference latency",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY,
)

TOOL_CALLS = Counter(
    "creditscope_tool_calls_total",
    "Total tool calls by tool name",
    ["tool_name"],
    registry=REGISTRY,
)

TOKENS_PROCESSED = Counter(
    "creditscope_tokens_processed_total",
    "Total tokens processed",
    ["direction"],
    registry=REGISTRY,
)

# --- Thinking / CoT Metrics ---
THINKING_TOKENS = Counter(
    "creditscope_thinking_tokens_total",
    "Total thinking tokens generated",
    registry=REGISTRY,
)

THINKING_BUDGET_UTILIZATION = Histogram(
    "creditscope_thinking_budget_utilization",
    "Thinking budget utilization percentage",
    buckets=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
    registry=REGISTRY,
)

THINKING_BUDGET_ENFORCED = Counter(
    "creditscope_thinking_budget_enforced_total",
    "Times the budget processor had to force-close thinking",
    registry=REGISTRY,
)

THINKING_DURATION = Histogram(
    "creditscope_thinking_duration_seconds",
    "Duration of thinking phases",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY,
)

THINKING_MODE_REQUESTS = Counter(
    "creditscope_thinking_mode_requests_total",
    "Requests by thinking mode",
    ["mode"],
    registry=REGISTRY,
)

THINKING_BUDGET_PRESET = Counter(
    "creditscope_thinking_budget_preset_total",
    "Requests by thinking budget preset",
    ["preset_name"],
    registry=REGISTRY,
)


def setup_prometheus_metrics() -> None:
    """Initialize Prometheus metrics (called at server startup)."""
    pass  # Metrics are registered at import time


def record_moe_trace(trace_data: dict) -> None:
    """Record MoE trace data into Prometheus metrics."""
    for layer_data in trace_data.get("layers", []):
        layer = layer_data["layer"]
        MOE_ROUTER_ENTROPY.labels(layer=layer).set(layer_data.get("entropy", 0))

        expert_load = layer_data.get("expert_load", [])
        for expert_id, load in enumerate(expert_load):
            if load > 0:
                MOE_EXPERT_ACTIVATION.labels(
                    layer=layer, expert_id=str(expert_id)
                ).inc(load)
            MOE_EXPERT_LOAD.labels(layer=layer).observe(load)

        avg_weights = layer_data.get("avg_gating_weights", [])
        for expert_id, weight in enumerate(avg_weights):
            MOE_GATING_WEIGHT_MEAN.labels(
                layer=layer, expert_id=str(expert_id)
            ).set(weight)


def record_thinking_stats(stats: dict) -> None:
    """Record thinking/CoT metrics."""
    tokens = stats.get("thinking_tokens_used", 0)
    if tokens > 0:
        THINKING_TOKENS.inc(tokens)

    utilization = stats.get("budget_utilization_pct")
    if utilization is not None:
        THINKING_BUDGET_UTILIZATION.observe(utilization)

    if stats.get("was_budget_enforced", False):
        THINKING_BUDGET_ENFORCED.inc()

    duration = stats.get("thinking_duration_ms", 0)
    if duration > 0:
        THINKING_DURATION.observe(duration / 1000.0)


def record_inference(latency: float, input_tokens: int, output_tokens: int) -> None:
    """Record inference latency and token counts."""
    INFERENCE_LATENCY.observe(latency)
    TOKENS_PROCESSED.labels(direction="input").inc(input_tokens)
    TOKENS_PROCESSED.labels(direction="output").inc(output_tokens)


def record_tool_call(tool_name: str) -> None:
    """Record a tool call."""
    TOOL_CALLS.labels(tool_name=tool_name).inc()


def get_metrics() -> bytes:
    """Generate Prometheus-format metrics output."""
    return generate_latest(REGISTRY)
