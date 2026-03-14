"""
Export attribution graphs for visualization.

Supports:
- JSON format compatible with Anthropic's attribution-graphs-frontend
- Compact summary format for API responses
- Graphviz DOT export for quick inspection
"""

from __future__ import annotations

import json
from pathlib import Path

from circuit_tracer.attribution.graph import AttributionGraph


def export_anthropic_format(graph: AttributionGraph, path: str | Path) -> dict:
    """
    Export in the format expected by the Anthropic attribution-graphs-frontend.

    See: https://github.com/anthropics/attribution-graphs-frontend
    """
    nodes = []
    for n in graph.nodes:
        node_data = {
            "id": n.id,
            "type": n.node_type,
            "layer": n.layer,
            "position": n.position,
            "feature_idx": n.feature_idx,
            "activation": round(n.activation, 6),
            "label": n.label or n.token or n.id,
            "token": n.token,
        }
        if n.metadata:
            node_data["metadata"] = n.metadata
        nodes.append(node_data)

    edges = []
    for e in graph.edges:
        edges.append({
            "source": e.source_id,
            "target": e.target_id,
            "weight": round(e.weight, 6),
            "virtual_weight": round(e.virtual_weight, 6),
        })

    data = {
        "title": f"Circuit: {graph.prompt[:60]}",
        "prompt": graph.prompt,
        "target_position": graph.target_position,
        "target_token": graph.target_token,
        "nodes": nodes,
        "edges": edges,
        "metadata": graph.metadata,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return data


def export_summary(graph: AttributionGraph) -> dict:
    """Compact summary suitable for API responses."""
    # Top features by activation
    features = sorted(
        [n for n in graph.feature_nodes],
        key=lambda n: abs(n.activation),
        reverse=True,
    )[:20]

    # Top edges by weight
    top_edges = sorted(
        graph.edges,
        key=lambda e: abs(e.weight),
        reverse=True,
    )[:30]

    return {
        "prompt": graph.prompt,
        "target_token": graph.target_token,
        "target_position": graph.target_position,
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "num_layers": graph.num_layers_spanned,
        "total_attribution": round(graph.total_attribution(), 6),
        "top_features": [
            {
                "layer": f.layer,
                "position": f.position,
                "feature_idx": f.feature_idx,
                "activation": round(f.activation, 6),
                "token": f.token,
            }
            for f in features
        ],
        "top_edges": [
            {
                "source": e.source_id,
                "target": e.target_id,
                "weight": round(e.weight, 6),
            }
            for e in top_edges
        ],
    }


def export_dot(graph: AttributionGraph, path: str | Path | None = None) -> str:
    """Export as Graphviz DOT for quick CLI visualization."""
    lines = ["digraph circuit {", '  rankdir=LR;', '  node [shape=box, fontsize=10];']

    # Group nodes by layer
    layers: dict[int, list] = {}
    for n in graph.nodes:
        layers.setdefault(n.layer, []).append(n)

    for layer_idx in sorted(layers.keys()):
        lines.append(f"  subgraph cluster_{layer_idx} {{")
        lines.append(f'    label="Layer {layer_idx}";')
        for n in layers[layer_idx]:
            label = f"{n.token}\\n{n.node_type}"
            if n.node_type == "feature":
                label = f"f{n.feature_idx}\\n{n.activation:.3f}"
            color = {"input": "lightblue", "feature": "lightyellow", "output": "lightgreen"}.get(n.node_type, "white")
            lines.append(f'    "{n.id}" [label="{label}", style=filled, fillcolor={color}];')
        lines.append("  }")

    for e in graph.edges:
        color = "red" if e.weight < 0 else "blue"
        width = max(0.5, min(3.0, abs(e.weight) * 5))
        lines.append(f'  "{e.source_id}" -> "{e.target_id}" [penwidth={width:.1f}, color={color}];')

    lines.append("}")
    dot_str = "\n".join(lines)

    if path:
        Path(path).write_text(dot_str)

    return dot_str
