"""
Prometheus metrics for circuit tracing.

Exposes SAE training progress, CLT fidelity, and attribution graph stats
to the existing CreditScope Prometheus + Grafana stack.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry

REGISTRY = CollectorRegistry()

# ─── SAE training metrics ─────────────────────────────────────────────────────

sae_training_step = Counter(
    "circuit_sae_training_step_total",
    "Total SAE training steps completed",
    ["layer"],
    registry=REGISTRY,
)

sae_recon_loss = Gauge(
    "circuit_sae_reconstruction_loss",
    "Current SAE reconstruction loss",
    ["layer"],
    registry=REGISTRY,
)

sae_sparsity_loss = Gauge(
    "circuit_sae_sparsity_loss",
    "Current SAE L1 sparsity loss",
    ["layer"],
    registry=REGISTRY,
)

sae_l0 = Gauge(
    "circuit_sae_l0",
    "Average number of active features per token",
    ["layer"],
    registry=REGISTRY,
)

# ─── CLT metrics ──────────────────────────────────────────────────────────────

clt_recon_loss = Gauge(
    "circuit_clt_reconstruction_loss",
    "CLT reconstruction loss (aggregated)",
    registry=REGISTRY,
)

clt_per_layer_l0 = Gauge(
    "circuit_clt_per_layer_l0",
    "CLT active features per layer",
    ["layer"],
    registry=REGISTRY,
)

# ─── Attribution graph metrics ────────────────────────────────────────────────

trace_requests_total = Counter(
    "circuit_trace_requests_total",
    "Number of circuit trace requests",
    registry=REGISTRY,
)

trace_duration_seconds = Histogram(
    "circuit_trace_duration_seconds",
    "Time to compute an attribution graph",
    buckets=[1, 5, 10, 30, 60, 120, 300],
    registry=REGISTRY,
)

graph_nodes_total = Histogram(
    "circuit_graph_nodes_total",
    "Number of nodes in attribution graphs",
    buckets=[10, 50, 100, 500, 1000, 5000],
    registry=REGISTRY,
)

graph_edges_total = Histogram(
    "circuit_graph_edges_total",
    "Number of edges in attribution graphs",
    buckets=[50, 200, 1000, 5000, 20000],
    registry=REGISTRY,
)

# ─── Intervention metrics ─────────────────────────────────────────────────────

steering_requests_total = Counter(
    "circuit_steering_requests_total",
    "Number of feature steering/ablation experiments",
    ["intervention_type"],
    registry=REGISTRY,
)


# ─── Helper to record SAE training step metrics ──────────────────────────────

def record_sae_step(layer: int, metrics: dict):
    """Record metrics from a single SAE training step."""
    layer_str = str(layer)
    sae_training_step.labels(layer=layer_str).inc()
    if "recon_loss" in metrics:
        sae_recon_loss.labels(layer=layer_str).set(metrics["recon_loss"])
    if "sparsity_loss" in metrics:
        sae_sparsity_loss.labels(layer=layer_str).set(metrics["sparsity_loss"])
    if "l0" in metrics:
        sae_l0.labels(layer=layer_str).set(metrics["l0"])


def record_trace(num_nodes: int, num_edges: int, duration_s: float):
    """Record metrics from a circuit trace."""
    trace_requests_total.inc()
    trace_duration_seconds.observe(duration_s)
    graph_nodes_total.observe(num_nodes)
    graph_edges_total.observe(num_edges)
