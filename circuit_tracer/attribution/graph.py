"""
Attribution graph data structures.

An attribution graph represents the causal flow of information from
input tokens through intermediate features to the output logit for
a specific token prediction.

Nodes are either:
- Input embeddings at (position, token_id)
- CLT/SAE features at (layer, position, feature_idx)
- Output logit contributions at (position, vocab_idx)

Edges carry a signed attribution weight: how much the source node's
activation contributes to the target node's activation through the
linearised network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class AttributionNode:
    """A node in the attribution graph."""
    id: str                          # unique identifier
    node_type: str                   # "input", "feature", "output"
    layer: int = -1                  # decoder layer index (-1 for input/output)
    position: int = 0               # sequence position
    feature_idx: int = -1           # SAE/CLT feature index (-1 for input/output)
    activation: float = 0.0        # feature activation value
    label: str = ""                 # human-readable label (auto-generated or manual)
    token: str = ""                # the token string at this position
    metadata: dict = field(default_factory=dict)


@dataclass
class AttributionEdge:
    """A directed edge in the attribution graph."""
    source_id: str
    target_id: str
    weight: float                   # signed attribution (activation × virtual_weight)
    virtual_weight: float = 0.0     # Jacobian through frozen layers
    source_activation: float = 0.0  # activation of source node


@dataclass
class AttributionGraph:
    """
    Complete attribution graph for a single prompt + target token.

    The graph is a DAG from input token embeddings through intermediate
    features to the target output logit.
    """
    prompt: str = ""
    target_position: int = -1
    target_token: str = ""
    target_logit: float = 0.0
    nodes: list[AttributionNode] = field(default_factory=list)
    edges: list[AttributionEdge] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # ── Accessors ─────────────────────────────────────────────────────────

    @property
    def input_nodes(self) -> list[AttributionNode]:
        return [n for n in self.nodes if n.node_type == "input"]

    @property
    def feature_nodes(self) -> list[AttributionNode]:
        return [n for n in self.nodes if n.node_type == "feature"]

    @property
    def output_nodes(self) -> list[AttributionNode]:
        return [n for n in self.nodes if n.node_type == "output"]

    def get_node(self, node_id: str) -> AttributionNode | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def incoming_edges(self, node_id: str) -> list[AttributionEdge]:
        return [e for e in self.edges if e.target_id == node_id]

    def outgoing_edges(self, node_id: str) -> list[AttributionEdge]:
        return [e for e in self.edges if e.source_id == node_id]

    # ── Stats ─────────────────────────────────────────────────────────────

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def num_layers_spanned(self) -> int:
        layers = {n.layer for n in self.feature_nodes}
        return len(layers)

    def total_attribution(self) -> float:
        """Sum of all edge attributions flowing into output nodes."""
        output_ids = {n.id for n in self.output_nodes}
        return sum(e.weight for e in self.edges if e.target_id in output_ids)

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "target_position": self.target_position,
            "target_token": self.target_token,
            "target_logit": self.target_logit,
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "metadata": {
                **self.metadata,
                "num_nodes": self.num_nodes,
                "num_edges": self.num_edges,
                "num_layers": self.num_layers_spanned,
            },
        }

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, data: dict) -> "AttributionGraph":
        nodes = [AttributionNode(**n) for n in data.get("nodes", [])]
        edges = [AttributionEdge(**e) for e in data.get("edges", [])]
        return cls(
            prompt=data.get("prompt", ""),
            target_position=data.get("target_position", -1),
            target_token=data.get("target_token", ""),
            target_logit=data.get("target_logit", 0.0),
            nodes=nodes,
            edges=edges,
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "AttributionGraph":
        return cls.from_dict(json.loads(Path(path).read_text()))
