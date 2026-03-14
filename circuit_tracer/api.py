"""
FastAPI router for circuit tracing.

Integrates with CreditScope's existing backend but is designed to be
portable to any FastAPI project.

Endpoints:
    POST /circuit/trace       — Trace a prompt and return the attribution graph
    POST /circuit/trace/prune — Trace + prune a prompt
    GET  /circuit/saes        — List trained SAE checkpoints
    GET  /circuit/architecture — Return the model architecture map
    POST /circuit/steer       — Run a feature steering intervention
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from circuit_tracer.config import get_config

router = APIRouter()


# ─── Request / Response schemas ───────────────────────────────────────────────

class TraceRequest(BaseModel):
    prompt: str
    target_position: int = -1
    prune: bool = False
    keep_fraction: float = 0.10


class SteerRequest(BaseModel):
    prompt: str
    interventions: list[dict]  # [{layer, position, feature_idx, value}]
    max_new_tokens: int = 50


class TraceResponse(BaseModel):
    prompt: str
    target_token: str
    target_position: int
    num_nodes: int
    num_edges: int
    num_layers: int
    total_attribution: float
    top_features: list[dict]
    top_edges: list[dict]
    graph_json_path: str | None = None


class SteerResponse(BaseModel):
    prompt: str
    intervention_type: str
    baseline_output: str
    intervened_output: str
    targets: list[dict]


# ─── Lazy-loaded singletons ──────────────────────────────────────────────────

_hooked_model = None
_feature_model = None
_arch_map = None


def _get_model():
    global _hooked_model
    if _hooked_model is None:
        from circuit_tracer.collectors.model_loader import HookedModel
        _hooked_model = HookedModel.load()
    return _hooked_model


def _get_feature_model():
    """Load the best available feature model (CLT > SAE)."""
    global _feature_model
    if _feature_model is not None:
        return _feature_model

    cfg = get_config()
    ckpt_dir = cfg.checkpoint_dir

    # Try CLT first
    clt_path = ckpt_dir / "clt_best.pt"
    if clt_path.exists():
        from circuit_tracer.transcoders import CrossLayerTranscoder
        _feature_model = CrossLayerTranscoder.load(str(clt_path), device="cuda")
        return _feature_model

    # Try SAE (pick any layer's best checkpoint)
    sae_files = sorted(ckpt_dir.glob("sae_layer_*_best.pt"))
    if sae_files:
        from circuit_tracer.saes import SparseAutoencoder
        _feature_model = SparseAutoencoder.load(str(sae_files[0]), device="cuda")
        return _feature_model

    raise HTTPException(
        status_code=503,
        detail="No trained feature model found. Run SAE or CLT training first.",
    )


def _get_arch_map():
    global _arch_map
    if _arch_map is not None:
        return _arch_map

    cfg = get_config()
    map_path = cfg.checkpoint_dir / "architecture_map.json"
    if map_path.exists():
        from circuit_tracer.collectors.architecture_map import ArchitectureMap
        _arch_map = ArchitectureMap.load(map_path)
        return _arch_map

    # Generate on the fly
    model = _get_model()
    from circuit_tracer.collectors.architecture_map import ArchitectureMap
    _arch_map = ArchitectureMap.from_model(model.model, model_name=cfg.model_path)
    _arch_map.save(map_path)
    return _arch_map


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/circuit/trace", response_model=TraceResponse)
async def trace_circuit(request: TraceRequest):
    """Trace the computational circuit for a prompt."""
    from circuit_tracer.attribution.replacement_model import ReplacementModel
    from circuit_tracer.attribution.pruning import prune_graph as do_prune
    from circuit_tracer.visualization.export import export_summary, export_anthropic_format

    model = _get_model()
    feature_model = _get_feature_model()

    rm = ReplacementModel(model, feature_model, request.prompt)
    graph = rm.trace(target_position=request.target_position)

    if request.prune:
        graph = do_prune(graph, keep_fraction=request.keep_fraction)

    # Save full graph JSON
    cfg = get_config()
    graph_dir = Path("circuit_tracer/data/graphs")
    graph_dir.mkdir(parents=True, exist_ok=True)
    slug = request.prompt[:40].replace(" ", "_").replace("/", "_")
    graph_path = graph_dir / f"trace_{slug}.json"
    export_anthropic_format(graph, graph_path)

    summary = export_summary(graph)
    return TraceResponse(
        **summary,
        graph_json_path=str(graph_path),
    )


@router.get("/circuit/architecture")
async def get_architecture():
    """Return the model's layer-by-layer architecture map."""
    arch = _get_arch_map()
    return {
        "model_name": arch.model_name,
        "num_layers": arch.num_layers,
        "d_model": arch.d_model,
        "moe_layers": arch.moe_layer_indices,
        "dense_layers": arch.dense_layer_indices,
        "deltanet_layers": arch.deltanet_layer_indices,
        "attention_layers": arch.attention_layer_indices,
        "summary": arch.summary(),
    }


@router.get("/circuit/saes")
async def list_saes():
    """List available trained SAE/CLT checkpoints."""
    cfg = get_config()
    ckpt_dir = cfg.checkpoint_dir
    if not ckpt_dir.exists():
        return {"checkpoints": []}

    checkpoints = []
    for f in sorted(ckpt_dir.glob("*.pt")):
        checkpoints.append({
            "name": f.name,
            "path": str(f),
            "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
        })
    return {"checkpoints": checkpoints}


@router.post("/circuit/steer", response_model=SteerResponse)
async def steer_feature(request: SteerRequest):
    """Run a feature steering experiment."""
    from circuit_tracer.interventions.steering import FeatureSteering

    model = _get_model()
    feature_model = _get_feature_model()

    steerer = FeatureSteering(model, feature_model)
    result = steerer.clamp_feature(
        prompt=request.prompt,
        layer=request.interventions[0]["layer"],
        position=request.interventions[0]["position"],
        feature_idx=request.interventions[0]["feature_idx"],
        value=request.interventions[0].get("value", 0.0),
        max_new_tokens=request.max_new_tokens,
    )

    return SteerResponse(
        prompt=result.prompt,
        intervention_type=result.intervention_type,
        baseline_output=result.baseline_output,
        intervened_output=result.intervened_output,
        targets=result.targets,
    )
