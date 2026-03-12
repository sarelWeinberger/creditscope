"""
MoE expert routing capture hooks for SGLang.

Hooks into SGLang's FusedMoE layer execution to capture expert routing data
for the Qwen3.5-35B-A3B architecture.

Architecture notes:
- Qwen3.5-35B-A3B uses Gated Delta Networks combined with sparse MoE
- FFN layers alternate between dense and MoE (sparse)
- MoE layers: 64 experts, top-4 routing
- Hooks are placed only on FFN MoE layers, not attention layers
"""

import collections
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from inference.config import NUM_EXPERTS, TOP_K_EXPERTS, MOE_LAYER_PATTERN

logger = structlog.get_logger(__name__)


@dataclass
class MoELayerTrace:
    """Trace data captured from a single MoE layer forward pass."""
    layer_name: str
    timestamp: float
    router_logits: np.ndarray       # [num_tokens, num_experts]
    selected_experts: np.ndarray    # [num_tokens, top_k]
    gating_weights: np.ndarray      # [num_tokens, top_k]
    expert_load: dict[int, int]     # expert_id -> token_count
    num_tokens: int
    entropy: float                  # Shannon entropy of routing distribution


@dataclass
class RequestTrace:
    """Full MoE trace for a single inference request."""
    request_id: str
    timestamp: float
    layer_traces: list[MoELayerTrace] = field(default_factory=list)
    phase: str = "unknown"  # "thinking" or "response"
    total_tokens: int = 0


class MoETraceCollector:
    """
    Hooks into SGLang's MoE layer execution to capture expert routing data.

    For Qwen3.5-35B-A3B architecture:
    - Registers forward hooks on FusedMoE layers
    - Captures router logits before top-k selection
    - Records which experts were activated per token per layer
    - Tracks gating weights (contribution of each expert)
    - Monitors expert load distribution across the batch
    - Stores traces in a ring buffer (last N requests)
    """

    def __init__(self, buffer_size: int = 100):
        self.traces: collections.deque[RequestTrace] = collections.deque(maxlen=buffer_size)
        self.layer_hooks: list[Any] = []
        self._current_trace: RequestTrace | None = None
        self._lock = threading.Lock()
        self._layer_map: dict[str, str] = {}  # layer_name -> "dense" | "moe"
        self._num_experts = NUM_EXPERTS
        self._top_k = TOP_K_EXPERTS

    def register_hooks(self, model: Any) -> None:
        """Walk model layers, register forward hooks on MoE gate modules."""
        logger.info("registering_moe_hooks")
        layer_idx = 0

        for name, module in model.named_modules():
            if self._is_moe_gate(name, module):
                hook = module.register_forward_hook(self._create_hook(name))
                self.layer_hooks.append(hook)
                self._layer_map[name] = "moe"
                logger.info("moe_hook_registered", layer=name, index=layer_idx)
            elif self._is_dense_ffn(name, module):
                self._layer_map[name] = "dense"

            layer_idx += 1

        logger.info(
            "moe_hooks_registration_complete",
            moe_layers=sum(1 for v in self._layer_map.values() if v == "moe"),
            dense_layers=sum(1 for v in self._layer_map.values() if v == "dense"),
        )

    def _is_moe_gate(self, name: str, module: Any) -> bool:
        """Check if a module is an MoE gate/router layer."""
        name_lower = name.lower()
        module_type = type(module).__name__.lower()

        return (
            MOE_LAYER_PATTERN in name_lower
            or "fusedmoe" in module_type
            or "moegate" in module_type
            or ("gate" in name_lower and "moe" in name_lower)
        )

    def _is_dense_ffn(self, name: str, module: Any) -> bool:
        """Check if a module is a dense FFN layer."""
        name_lower = name.lower()
        return "mlp" in name_lower and "expert" not in name_lower

    def _create_hook(self, layer_name: str):
        """Create a forward hook for a specific MoE layer."""
        def hook_fn(module: Any, input: tuple, output: Any) -> None:
            with self._lock:
                if self._current_trace is None:
                    return

                try:
                    trace = self._extract_trace(layer_name, module, input, output)
                    if trace:
                        self._current_trace.layer_traces.append(trace)
                except Exception as e:
                    logger.warning("moe_hook_error", layer=layer_name, error=str(e))

        return hook_fn

    def _extract_trace(
        self, layer_name: str, module: Any, input: tuple, output: Any
    ) -> MoELayerTrace | None:
        """Extract MoE routing data from a forward pass.

        Uses non-blocking GPU operations where possible and avoids
        transferring full router_logits to CPU.  Only the top-k indices,
        gating weights, and a scalar entropy value are moved to host memory.
        """
        try:
            import torch

            # ── Resolve raw tensors (stay on GPU) ─────────────────────────
            raw_logits = None
            raw_selected = None
            raw_weights = None

            if hasattr(output, "router_logits"):
                raw_logits = output.router_logits.detach()
                raw_selected = output.selected_experts.detach()
                raw_weights = output.gating_weights.detach()
            elif isinstance(output, tuple) and len(output) >= 2:
                if hasattr(output[1], "detach"):
                    raw_logits = output[1].detach()
            elif hasattr(module, "last_router_logits"):
                raw_logits = module.last_router_logits.detach()

            if raw_logits is None:
                return None

            num_tokens = raw_logits.shape[0]

            # ── Top-k selection on GPU ────────────────────────────────────
            if raw_selected is None:
                raw_selected = torch.topk(raw_logits, self._top_k, dim=-1).indices

            if raw_weights is None:
                max_logits = raw_logits.max(dim=-1, keepdim=True).values
                exp_logits = torch.exp(raw_logits - max_logits)
                softmax = exp_logits / exp_logits.sum(dim=-1, keepdim=True)
                raw_weights = torch.gather(softmax, 1, raw_selected)

            # ── Entropy on GPU (scalar) ───────────────────────────────────
            max_logits = raw_logits.max(dim=-1, keepdim=True).values
            probs = torch.exp(raw_logits - max_logits)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            entropy_val = float(
                -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
            )

            # ── Single CPU transfer (small tensors only) ─────────────────
            selected_experts = raw_selected.cpu().numpy()
            gating_weights = raw_weights.cpu().numpy()

            # Expert load from the small selected_experts array
            expert_load = {}
            for expert_id in range(self._num_experts):
                count = int(np.sum(selected_experts == expert_id))
                if count > 0:
                    expert_load[expert_id] = count

            return MoELayerTrace(
                layer_name=layer_name,
                timestamp=time.time(),
                router_logits=np.empty(0),  # not stored to save memory
                selected_experts=selected_experts,
                gating_weights=gating_weights,
                expert_load=expert_load,
                num_tokens=num_tokens,
                entropy=entropy_val,
            )
        except Exception as e:
            logger.warning("trace_extraction_failed", layer=layer_name, error=str(e))
            return None

    def begin_trace(self, request_id: str | None = None, phase: str = "unknown") -> str:
        """Start collecting traces for a new request."""
        rid = request_id or str(uuid.uuid4())
        with self._lock:
            self._current_trace = RequestTrace(
                request_id=rid,
                timestamp=time.time(),
                phase=phase,
            )
        return rid

    def end_trace(self) -> RequestTrace | None:
        """Finish trace collection and store in buffer."""
        with self._lock:
            trace = self._current_trace
            if trace:
                trace.total_tokens = sum(lt.num_tokens for lt in trace.layer_traces)
                self.traces.append(trace)
            self._current_trace = None
            return trace

    def get_latest_trace(self, request_id: str | None = None) -> RequestTrace | None:
        """Return structured trace data for the observability API."""
        with self._lock:
            if request_id:
                for trace in reversed(self.traces):
                    if trace.request_id == request_id:
                        return trace
                return None
            return self.traces[-1] if self.traces else None

    def get_expert_heatmap(self, num_requests: int = 50) -> dict:
        """Aggregate expert activation frequency across recent requests."""
        with self._lock:
            recent = list(self.traces)[-num_requests:]

        if not recent:
            return {"layers": [], "experts": list(range(self._num_experts)), "data": []}

        layer_expert_counts: dict[str, dict[int, int]] = {}

        for trace in recent:
            for lt in trace.layer_traces:
                if lt.layer_name not in layer_expert_counts:
                    layer_expert_counts[lt.layer_name] = {i: 0 for i in range(self._num_experts)}
                for expert_id, count in lt.expert_load.items():
                    layer_expert_counts[lt.layer_name][expert_id] += count

        layers = sorted(layer_expert_counts.keys())
        data = []
        for layer in layers:
            row = [layer_expert_counts[layer].get(i, 0) for i in range(self._num_experts)]
            data.append(row)

        return {
            "layers": layers,
            "experts": list(range(self._num_experts)),
            "data": data,
        }

    def get_entropy_timeseries(self, num_requests: int = 50) -> list[dict]:
        """Get router entropy over time for each layer."""
        with self._lock:
            recent = list(self.traces)[-num_requests:]

        series = []
        for trace in recent:
            entry = {
                "request_id": trace.request_id,
                "timestamp": trace.timestamp,
                "layers": {},
            }
            for lt in trace.layer_traces:
                entry["layers"][lt.layer_name] = lt.entropy
            series.append(entry)

        return series

    def get_layer_map(self) -> dict[str, str]:
        """Return the dense/MoE layer map discovered at registration."""
        return dict(self._layer_map)

    def cleanup(self) -> None:
        """Remove all hooks."""
        for hook in self.layer_hooks:
            hook.remove()
        self.layer_hooks.clear()
        logger.info("moe_hooks_removed")


# Global collector instance
_collector: MoETraceCollector | None = None


def get_collector() -> MoETraceCollector:
    global _collector
    if _collector is None:
        _collector = MoETraceCollector()
    return _collector
