"""Small, explicit graph data contract; no renderer concepts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Graph:
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    _ids: set[str] = field(default_factory=set, repr=False)
    _edge_ids: set[str] = field(default_factory=set, repr=False)

    def add_node(self, node: dict[str, Any]) -> str:
        if node["id"] in self._ids:
            raise ValueError(f"Duplicate node ID: {node['id']}")
        self.nodes.append(node)
        self._ids.add(node["id"])
        return node["id"]

    def add_edge(self, edge: dict[str, Any]) -> None:
        if edge["id"] in self._edge_ids:
            raise ValueError(f"Duplicate edge ID: {edge['id']}")
        self.edges.append(edge)
        self._edge_ids.add(edge["id"])

    def issue(self, *, kind: str, path: str, message: str, span: dict | None = None) -> None:
        record: dict = {"kind": kind, "path": path, "message": message}
        if span is not None:
            record["span"] = span
        self.diagnostics.append(record)
