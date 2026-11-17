"""Provenance-preserving, ViT-Tiny-only instance graph hierarchy.

Summaries collapse *observed source-proven* operation/edge IDs. They are
presentation groups, not additional network operations or execution claims.
The original full Operation view is unchanged.
"""
from __future__ import annotations

from collections import defaultdict

from ae.sae.ir.canonical import stable_id
from .health import issue


def _unsupported(level: str, reason: str) -> dict:
    return {
        "view": "instance", "level": level, "nodes": [], "links": [],
        "issues": [issue("UNSUPPORTED_INSTANCE_PROJECTION", reason, severity="error")],
    }


def _stage_sequence(nodes: list[dict]) -> list[str]:
    current = "input"
    result = []
    for node in nodes:
        target = node["target"]
        if node["op"] == "placeholder":
            stage = "input"
        elif node["op"] == "output":
            stage = "output"
        elif target.startswith("blocks."):
            parts = target.split(".")
            stage = "blocks." + parts[1] if len(parts) > 2 and parts[1].isdigit() else current
        elif (target.startswith("patch_embed.") or target in
              {"pos_embed", "cls_token", "pos_drop", "patch_drop", "norm_pre"}):
            stage = "embedding"
        elif (target == "norm.weight" or target == "norm.bias" or target in
              {"fc_norm", "head_drop", "head"}):
            stage = "head"
        elif current == "input":
            stage = "embedding"
        else:
            stage = current
        current = stage
        result.append(stage)
    return result


def _unique_refs(nodes: list[dict]) -> list[dict]:
    keyed = {}
    for n in nodes:
        for r in n.get("source_refs", []):
            k = (r.get("path"), r.get("start_byte"), r.get("end_byte"))
            keyed[k] = r
    return [keyed[k] for k in sorted(keyed)]


def _summary(graph: dict, stages: list[str], level: str) -> dict:
    mapped = [
        "transformer" if level == "overview" and s.startswith("blocks.") else s
        for s in stages
    ]
    groups: dict[str, list[dict]] = defaultdict(list)
    ordered_groups: list[str] = []
    for node, key in zip(graph["nodes"], mapped):
        if key not in groups:
            ordered_groups.append(key)
        groups[key].append(node)
    labels = {
        "input": "Image / Optional Attn Mask", "embedding": "Patch / Token Embedding",
        "transformer": "Transformer Encoder (12 blocks)",
        "head": "LayerNorm / Classifier", "output": "Outputs",
    }
    new_nodes = []
    group_id = {}
    member_group = {}
    for key in ordered_groups:
        members = groups[key]
        ident = stable_id("instance-v021", "timm-vit-tiny", level, key)
        group_id[key] = ident
        node_ids = sorted({original for n in members for original in n["ir_node_ids"]})
        for n in members:
            member_group[n["id"]] = key
        last_shape = next((n["shape"] for n in reversed(members)
                           if n.get("shape") != "metadata"), "metadata")
        new_nodes.append({
            "id": ident, "kind": "instance-summary",
            "label": labels.get(key, "Transformer " + key.replace("blocks.", "Block ")),
            "target": labels.get(key, "Transformer " + key.replace("blocks.", "Block ")),
            "shape": last_shape, "evidence": "derived",
            "parent_id": None, "op": "presentation_group",
            "ir_node_ids": node_ids, "ir_edge_ids": [],
            "source_refs": _unique_refs(members),
            "instance_node_id": None,
            "member_instance_node_ids": [n["id"] for n in members],
        })
    combined: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    collapsed_ids: list[str] = []
    for edge in graph["links"]:
        left = member_group[edge["source"]]
        right = member_group[edge["target"]]
        if left == right:
            collapsed_ids.append(edge["id"])
        else:
            combined[left, right, edge["relation_kind"]].append(edge)
    new_links = []
    for (left, right, relation), edges in sorted(
            combined.items(), key=lambda item: (ordered_groups.index(item[0][0]),
                                                ordered_groups.index(item[0][1]),
                                                item[0][2])):
        ids = sorted({eid for edge in edges for eid in edge["ir_edge_ids"]})
        new_links.append({
            "id": stable_id("instance-v021-edge", level, left, right, relation),
            "source": group_id[left], "target": group_id[right],
            "relation_kind": relation, "ir_edge_ids": ids,
            "source_refs": [], "style": (
                "reference" if relation == "STATIC_INSTANCE_SHAPE_METADATA_DEPENDENCY"
                else "static-value"), "input_slot": None,
        })
    # Conservation of evidence: no operation or dependency disappears from
    # the projection without being represented or explicitly collapsed.
    src_nodes = {n["id"] for n in graph["nodes"]}
    assert src_nodes == {oid for n in new_nodes for oid in n["member_instance_node_ids"]}
    src_edges = {e["id"] for e in graph["links"]}
    projected_edges = set(collapsed_ids) | {
        eid for e in new_links for eid in e["ir_edge_ids"]
    }
    assert src_edges == projected_edges, "Hierarchy lost original dependency evidence"
    return {
        **graph, "level": level, "nodes": new_nodes, "links": new_links,
        "collapsed_instance_edge_ids": sorted(collapsed_ids),
        "issues": graph["issues"] + [issue(
            "PRESENTATION_GROUPING", "Operations and within-group edges are "
            "collapsed for readability; original IDs remain in the manifest."
        )],
    }


def _focus(graph: dict, stages: list[str], focus: str) -> dict:
    if focus not in set(stages):
        return _unsupported("operation", "No source-proven ViT stage: " + focus)
    members = {node["id"] for node, stage in zip(graph["nodes"], stages)
               if stage == focus}
    # Show immediate context for incoming / outgoing cross-boundary edges.
    included = set(members)
    for link in graph["links"]:
        if link["source"] in members or link["target"] in members:
            included.update((link["source"], link["target"]))
    nodes = [n for n in graph["nodes"] if n["id"] in included]
    links = [e for e in graph["links"]
             if e["source"] in included and e["target"] in included]
    discarded_nodes = sorted({n["id"] for n in graph["nodes"]} - included)
    discarded_edges = sorted({e["id"] for e in graph["links"]} -
                             {e["id"] for e in links})
    return {
        **graph, "nodes": nodes, "links": links, "focus": focus,
        "omitted_instance_node_ids": discarded_nodes,
        "omitted_instance_edge_ids": discarded_edges,
        "issues": graph["issues"] + [issue(
            "FOCUS_SUBGRAPH", "Only selected source-proven stage and immediate "
            "dependency context are shown; omitted IDs are recorded in manifest."
        )],
    }


def project_vit_hierarchy(graph: dict, *, level: str,
                          focus: str | None, source_scope: str) -> dict:
    if source_scope != "pinned_timm_1_0_22_vit_tiny_patch16_224":
        return _unsupported(level,
                            "Overview/block/focus currently require pinned timm ViT-Tiny")
    stages = _stage_sequence(graph["nodes"])
    expected = {f"blocks.{i}" for i in range(12)}
    if not expected.issubset(set(stages)):
        return _unsupported(level, "Missing exact 12 Transformer Block evidence")
    if focus is not None:
        if level != "operation":
            return _unsupported(level, "Focus requires operation level")
        return _focus(graph, stages, focus)
    if level == "operation":
        return graph
    if level in ("block", "overview"):
        return _summary(graph, stages, level)
    return _unsupported(level, "Unknown instance hierarchy level")
