"""
Custom MoE expert routing capture hooks for SGLang.

Hooks into SGLang's FusedMoE layer execution to capture expert routing data
for the Qwen3.5-35B-A3B architecture:
- Gated Delta Networks combined with sparse MoE
- 64 experts per MoE layer, top-4 routing
- Captures router logits, expert selections, gating weights, and load distribution
"""

import collections
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from inference.config import MOE_TRACE_BUFFER_SIZE, NUM_EXPERTS, TOP_K_EXPERTS

logger = structlog.get_logger(__name__)


@dataclass
class LayerTrace:
    """Trace data for a single MoE layer during one forward pass."""
    layer_name: str
    timestamp: float
    router_logits: np.ndarray  # [num_tokens, num_experts]
    selected_experts: np.ndarray  # [num_tokens, top_k]
    gating_weights: np.ndarray  # [num_tokens, top_k]
    expert_load: np.ndarray  # [num_experts] — token count per expert
    entropy: float  # Shannon entropy of routing distribution
    num_tokens: int


@dataclass
class RequestTrace:
    """Complete trace for a single inference request across all MoE layers."""
    request_id: str
    timestamp: float
    layer_traces: list[LayerTrace] = field(default_factory=list)
    phase: str = "unknown"  # "thinking" or "response"
    total_tokens: int = 0


class MoETraceCollector:
    """
    Hooks into SGLang's MoE layer execution to capture expert routing data.

    For Qwen3.5-35B-A3B architecture:
    - Uses Gated Delta Networks combined with sparse MoE
    - Captures router logits before top-k selection
    - Records which experts were activated per token per layer
    - Tracks gating weights (contribution of each expert)
    - Monitors expert load distribution across the batch

    Implementation approach:
    1. Register forward hooks on FusedMoE layers via SGLang's model internals
    2. On each forward pass, extract routing data
    3. Store traces in a ring buffer (last N requests)
    4. Expose via /moe_trace API endpoint
    """

    def __init__(self, buffer_size: int = MOE_TRACE_BUFFER_SIZE):
        self.traces: collections.deque[RequestTrace] = collections.deque(
            maxlen=buffer_size
        )
        self.layer_hooks: list[Any] = []
        self._lock = threading.Lock()
        self._current_request_id: str | None = None
        self._current_trace: RequestTrace | None = None
        self._layer_map: dict[str, str] = {}  # layer_name -> "dense" | "moe"

    def register_hooks(self, model: Any) -> None:
        """Walk model layers, register forward hooks on MoE gate modules."""
        logger.info("registering_moe_hooks")
        layer_index = 0

        for name, module in model.named_modules():
            if self._is_moe_gate(name, module):
                hook = module.register_forward_hook(self._create_hook(name))
                self.layer_hooks.append(hook)
                self._layer_map[name] = "moe"
                logger.info(
                    "moe_layer_found",
                    layer=name,
                    index=layer_index,
                    type="MoE",
                    experts=NUM_EXPERTS,
                    top_k=TOP_K_EXPERTS,
                )
            else:
                if self._is_ffn_layer(name, module):
                    self._layer_map[name] = "dense"
                    logger.info(
                        "dense_layer_found",
                        layer=name,
                        index=layer_index,
                        type="Dense FFN",
                    )
            layer_index += 1

        logger.info(
            "moe_hooks_registered",
            total_hooks=len(self.layer_hooks),
            layer_map_size=len(self._layer_map),
        )

    def _is_moe_gate(self, name: str, module: Any) -> bool:
        """Check if a module is a MoE gate/router layer."""
        module_type = type(module).__name__.lower()
        moe_indicators = ["fusedmoe", "moegate", "mixtralgate", "sparsemlp", "moe"]
        name_lower = name.lower()
        return any(ind in module_type for ind in moe_indicators) or (
            "gate" in name_lower and "moe" in name_lower
        )

    def _is_ffn_layer(self, name: str, module: Any) -> bool:
        """Check if a module is a dense FFN layer."""
        module_type = type(module).__name__.lower()
        return any(ind in module_type for ind in ["mlp", "ffn", "feedforward"])

    def _create_hook(self, layer_name: str):
        """Create a forward hook for a specific MoE layer."""

        def hook_fn(module: Any, input: tuple, output: Any) -> None:
            try:
                self._process_moe_output(layer_name, module, input, output)
            except Exception as e:
                logger.warning(
                    "moe_hook_error", layer=layer_name, error=str(e)
                )

        return hook_fn

    def _process_moe_output(
        self, layer_name: str, module: Any, input: tuple, output: Any
    ) -> None:
        """Extract routing data from a MoE layer's forward pass output."""
        # Extract tensors — handle different SGLang MoE output formats
        router_logits = self._extract_router_logits(output)
        selected_experts = self._extract_selected_experts(output)
        gating_weights = self._extract_gating_weights(output)

        if router_logits is None:
            return

        # Convert to numpy for storage
        router_logits_np = router_logits.detach().cpu().float().numpy()
        selected_np = (
            selected_experts.detach().cpu().numpy()
            if selected_experts is not None
            else self._compute_topk(router_logits_np)
        )
        weights_np = (
            gating_weights.detach().cpu().float().numpy()
            if gating_weights is not None
            else self._compute_gating_weights(router_logits_np, selected_np)
        )

        # Compute expert load and entropy
        expert_load = self._compute_load(selected_np)
        entropy = self._compute_entropy(router_logits_np)
        num_tokens = router_logits_np.shape[0]

        trace = LayerTrace(
            layer_name=layer_name,
            timestamp=time.time(),
            router_logits=router_logits_np,
            selected_experts=selected_np,
            gating_weights=weights_np,
            expert_load=expert_load,
            entropy=float(entropy),
            num_tokens=num_tokens,
        )

        with self._lock:
            if self._current_trace is not None:
                self._current_trace.layer_traces.append(trace)
                self._current_trace.total_tokens += num_tokens

    def _extract_router_logits(self, output: Any) -> Any:
        """Extract router logits from various output formats."""
        if hasattr(output, "router_logits"):
            return output.router_logits
        if isinstance(output, tuple) and len(output) >= 2:
            # Some implementations return (hidden_states, router_logits, ...)
            for item in output:
                if hasattr(item, "shape") and len(item.shape) == 2:
                    if item.shape[-1] == NUM_EXPERTS:
                        return item
        return None

    def _extract_selected_experts(self, output: Any) -> Any:
        """Extract selected expert indices from output."""
        if hasattr(output, "selected_experts"):
            return output.selected_experts
        if hasattr(output, "topk_indices"):
            return output.topk_indices
        return None

    def _extract_gating_weights(self, output: Any) -> Any:
        """Extract gating weights from output."""
        if hasattr(output, "gating_weights"):
            return output.gating_weights
        if hasattr(output, "topk_weights"):
            return output.topk_weights
        return None

    def _compute_topk(self, router_logits: np.ndarray) -> np.ndarray:
        """Compute top-k expert indices from router logits."""
        return np.argsort(router_logits, axis=-1)[:, -TOP_K_EXPERTS:][:, ::-1]

    def _compute_gating_weights(
        self, router_logits: np.ndarray, selected: np.ndarray
    ) -> np.ndarray:
        """Compute softmax gating weights for selected experts."""
        # Gather logits for selected experts
        batch_idx = np.arange(router_logits.shape[0])[:, None]
        selected_logits = router_logits[batch_idx, selected]
        # Softmax over selected experts
        exp_logits = np.exp(selected_logits - selected_logits.max(axis=-1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    def _compute_load(self, selected_experts: np.ndarray) -> np.ndarray:
        """Compute per-expert token load (how many tokens routed to each expert)."""
        load = np.zeros(NUM_EXPERTS, dtype=np.int64)
        for expert_id in selected_experts.flatten():
            if 0 <= expert_id < NUM_EXPERTS:
                load[expert_id] += 1
        return load

    def _compute_entropy(self, router_logits: np.ndarray) -> float:
        """Compute Shannon entropy of the routing distribution (averaged across tokens)."""
        # Softmax across experts
        exp_logits = np.exp(router_logits - router_logits.max(axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        # Shannon entropy per token
        log_probs = np.log(probs + 1e-10)
        entropy_per_token = -np.sum(probs * log_probs, axis=-1)
        return float(np.mean(entropy_per_token))

    # --- Public API ---

    def start_request(self, request_id: str | None = None, phase: str = "unknown") -> str:
        """Begin tracing a new inference request."""
        req_id = request_id or str(uuid.uuid4())
        with self._lock:
            self._current_request_id = req_id
            self._current_trace = RequestTrace(
                request_id=req_id,
                timestamp=time.time(),
                phase=phase,
            )
        return req_id

    def end_request(self) -> RequestTrace | None:
        """Finalize and store the current request trace."""
        with self._lock:
            trace = self._current_trace
            if trace is not None:
                self.traces.append(trace)
            self._current_trace = None
            self._current_request_id = None
        return trace

    def get_latest_trace(self, request_id: str | None = None) -> dict | None:
        """Return structured trace data for the observability API."""
        with self._lock:
            if request_id:
                for trace in reversed(self.traces):
                    if trace.request_id == request_id:
                        return self._serialize_trace(trace)
                return None
            if self.traces:
                return self._serialize_trace(self.traces[-1])
        return None

    def get_expert_heatmap(self, last_n: int = 50) -> dict:
        """Aggregate expert activation frequency across recent requests."""
        with self._lock:
            traces = list(self.traces)[-last_n:]

        if not traces:
            return {"layers": [], "heatmap": []}

        layer_names = set()
        activation_counts: dict[str, np.ndarray] = {}

        for trace in traces:
            for lt in trace.layer_traces:
                layer_names.add(lt.layer_name)
                if lt.layer_name not in activation_counts:
                    activation_counts[lt.layer_name] = np.zeros(NUM_EXPERTS)
                activation_counts[lt.layer_name] += lt.expert_load

        sorted_layers = sorted(layer_names)
        heatmap = []
        for layer in sorted_layers:
            counts = activation_counts.get(layer, np.zeros(NUM_EXPERTS))
            total = counts.sum()
            normalized = (counts / total * 100).tolist() if total > 0 else counts.tolist()
            heatmap.append({
                "layer": layer,
                "activation_counts": counts.tolist(),
                "activation_pct": normalized,
            })

        return {"layers": sorted_layers, "heatmap": heatmap}

    def get_entropy_timeseries(self, last_n: int = 50) -> list[dict]:
        """Return router entropy over time for each layer."""
        with self._lock:
            traces = list(self.traces)[-last_n:]

        series = []
        for trace in traces:
            entry = {
                "request_id": trace.request_id,
                "timestamp": trace.timestamp,
                "phase": trace.phase,
                "layers": {},
            }
            for lt in trace.layer_traces:
                entry["layers"][lt.layer_name] = lt.entropy
            series.append(entry)
        return series

    def get_layer_details(self) -> list[dict]:
        """Return per-layer activation data from the latest request."""
        with self._lock:
            if not self.traces:
                return []
            trace = self.traces[-1]

        details = []
        for lt in trace.layer_traces:
            details.append({
                "layer": lt.layer_name,
                "num_tokens": lt.num_tokens,
                "selected_experts": lt.selected_experts.tolist(),
                "gating_weights": lt.gating_weights.tolist(),
                "expert_load": lt.expert_load.tolist(),
                "entropy": lt.entropy,
                "top_experts": self._get_top_experts(lt),
            })
        return details

    def _get_top_experts(self, lt: LayerTrace, top_n: int = 8) -> list[dict]:
        """Get the most frequently activated experts for a layer."""
        top_indices = np.argsort(lt.expert_load)[-top_n:][::-1]
        total = lt.expert_load.sum()
        return [
            {
                "expert_id": int(idx),
                "token_count": int(lt.expert_load[idx]),
                "activation_pct": round(float(lt.expert_load[idx] / total * 100), 2)
                if total > 0
                else 0,
            }
            for idx in top_indices
            if lt.expert_load[idx] > 0
        ]

    def _serialize_trace(self, trace: RequestTrace) -> dict:
        """Serialize a RequestTrace for JSON API response."""
        return {
            "request_id": trace.request_id,
            "timestamp": trace.timestamp,
            "phase": trace.phase,
            "total_tokens": trace.total_tokens,
            "num_layers": len(trace.layer_traces),
            "layers": [
                {
                    "layer": lt.layer_name,
                    "timestamp": lt.timestamp,
                    "num_tokens": lt.num_tokens,
                    "entropy": lt.entropy,
                    "expert_load": lt.expert_load.tolist(),
                    "top_experts": self._get_top_experts(lt),
                    "avg_gating_weights": lt.gating_weights.mean(axis=0).tolist()
                    if lt.gating_weights.size > 0
                    else [],
                }
                for lt in trace.layer_traces
            ],
        }


# Global singleton
_collector: MoETraceCollector | None = None


def get_collector() -> MoETraceCollector:
    """Get or create the global MoE trace collector."""
    global _collector
    if _collector is None:
        _collector = MoETraceCollector()
    return _collector
