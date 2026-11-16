"""IR integrity, proof status, source-byte range and edge checks."""
from __future__ import annotations


class ValidationError(ValueError):
    pass


def validate_ir(ir: dict, source_sizes: dict[str, int] | None = None) -> dict:
    errors: list[str] = []
    if ir.get("schema_version") != "1.0":
        errors.append("schema_version must equal 1.0")
    nodes = ir.get("nodes", [])
    edges = ir.get("edges", [])
    ids = [item.get("id") for item in nodes]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate node IDs")
    node_set = set(ids)
    edge_ids = [edge.get("id") for edge in edges]
    if len(edge_ids) != len(set(edge_ids)):
        errors.append("Duplicate edge IDs")
    for node in nodes:
        if node.get("evidence_class") not in ("explicit", "derived"):
            errors.append(f"Invalid evidence_class: {node.get('id')}")
        if node.get("evidence_class") == "derived" and not node.get("analysis_rule"):
            errors.append(f"Missing analysis rule: {node.get('id')}")
        if not node.get("source_refs"):
            errors.append(f"Missing provenance: {node.get('id')}")
        for span in node.get("source_refs", []):
            path = span.get("path")
            if source_sizes is not None:
                size = source_sizes.get(path)
                if size is None or not (0 <= span.get("start_byte", -1) <= span.get("end_byte", -1) <= size):
                    errors.append(f"Invalid source span: {node.get('id')}")
    for edge in edges:
        if edge.get("source") not in node_set or edge.get("target") not in node_set:
            errors.append(f"Dangling graph edge: {edge.get('id')}")
        for span in edge.get("source_refs", []):
            if source_sizes is not None:
                size = source_sizes.get(span.get("path"))
                if size is None or not (0 <= span.get("start_byte", -1) <= span.get("end_byte", -1) <= size):
                    errors.append(f"Invalid edge source span: {edge.get('id')}")
        if not edge.get("source_refs"):
            errors.append(f"Missing edge provenance: {edge.get('id')}")
        if edge.get("evidence_class") == "derived" and not edge.get("analysis_rule"):
            errors.append(f"Missing edge inference rule: {edge.get('id')}")
    if errors:
        raise ValidationError("; ".join(errors[:15]))
    return {"valid": True, "node_count": len(nodes), "edge_count": len(edges)}
