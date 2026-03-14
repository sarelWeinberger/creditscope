"""
Graph pruning to reduce attribution graphs to their essential circuits.

The raw graph from the replacement model can have thousands of nodes
and tens of thousands of edges.  Pruning keeps only the nodes and edges
that significantly influence the target output logit.
"""

from __future__ import annotations

from collections import defaultdict

import structlog

from circuit_tracer.attribution.graph import AttributionGraph, AttributionNode

logger = structlog.get_logger(__name__)


def prune_graph(
    graph: AttributionGraph,
    keep_fraction: float = 0.10,
    min_nodes: int = 5,
    max_nodes: int = 500,
) -> AttributionGraph:
    """
    Prune the attribution graph to keep only high-impact nodes.

    Algorithm:
    1. Score each node by its total influence on the output
       (sum of |attribution| on all paths to output nodes)
    2. Keep the top-k nodes that explain the bulk of the signal
    3. Drop edges between pruned nodes
    4. Always keep input and output nodes

    Args:
        graph:          The full attribution graph.
        keep_fraction:  Fraction of feature nodes to keep.
        min_nodes:      Minimum feature nodes to keep regardless of fraction.
        max_nodes:      Maximum feature nodes to keep.

    Returns:
        A new, pruned AttributionGraph.
    """
    importance = _compute_node_importance(graph)

    # Separate feature nodes from input/output (which are always kept)
    feature_nodes = [n for n in graph.nodes if n.node_type == "feature"]
    keep_nodes = {n.id for n in graph.nodes if n.node_type != "feature"}

    # Sort features by importance
    scored = [(n, importance.get(n.id, 0.0)) for n in feature_nodes]
    scored.sort(key=lambda x: abs(x[1]), reverse=True)

    # Determine how many to keep
    k = max(min_nodes, min(max_nodes, int(len(scored) * keep_fraction)))
    for node, score in scored[:k]:
        keep_nodes.add(node.id)

    # Build pruned graph
    pruned = AttributionGraph(
        prompt=graph.prompt,
        target_position=graph.target_position,
        target_token=graph.target_token,
        target_logit=graph.target_logit,
        metadata={**graph.metadata, "pruned": True, "original_nodes": graph.num_nodes, "original_edges": graph.num_edges},
    )

    pruned.nodes = [n for n in graph.nodes if n.id in keep_nodes]
    pruned.edges = [
        e for e in graph.edges
        if e.source_id in keep_nodes and e.target_id in keep_nodes
    ]

    logger.info(
        "graph_pruned",
        original_nodes=graph.num_nodes,
        pruned_nodes=pruned.num_nodes,
        original_edges=graph.num_edges,
        pruned_edges=pruned.num_edges,
    )
    return pruned


def _compute_node_importance(graph: AttributionGraph) -> dict[str, float]:
    """
    Score each node by its total contribution to the output.

    Uses backward accumulation: starting from output nodes, propagate
    importance scores backward through edges.  A node's importance is
    the sum of |edge.weight| for all downstream paths reaching an output.
    """
    # Build adjacency (target → list of incoming edges)
    incoming: dict[str, list] = defaultdict(list)
    for edge in graph.edges:
        incoming[edge.target_id].append(edge)

    importance: dict[str, float] = {}

    # Output nodes have importance 1.0
    output_ids = {n.id for n in graph.output_nodes}
    for oid in output_ids:
        importance[oid] = 1.0

    # Process layers in reverse (output → input)
    visited = set()
    queue = list(output_ids)

    while queue:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)

        node_imp = importance.get(node_id, 0.0)
        for edge in incoming.get(node_id, []):
            propagated = abs(edge.weight) * node_imp
            importance[edge.source_id] = importance.get(edge.source_id, 0.0) + propagated
            if edge.source_id not in visited:
                queue.append(edge.source_id)

    return importance
