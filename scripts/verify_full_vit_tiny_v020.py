"""Independent AE v0.20 acceptance gate against unchanged timm ViT-Tiny.

Original model execution is confined to preflight_vit_tiny_v020; AE receives
only original source bytes, concrete input shape and explicit attention branch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ae.sae.api import extract, validate_bundle
from ae.presenter.api import present

from scripts.preflight_vit_tiny_v020 import audit


def verify(output: Path):
    reference = audit(output)
    assert reference["status"] == "reference_preflight_only"
    results = []
    for case in reference["cases"]:
        label = case["configuration"]
        folder = output / label
        original = json.loads((folder / "fx.json").read_text())
        bundle = folder / "ae-bundle"
        shape = tuple(case["shape"])
        fused = case["fused_attention"]
        extract(output / "source", bundle,
                entrypoint="timm.models.vision_transformer:vit_tiny_patch16_224",
                input_shape=shape, attention_mode="fused" if fused else "unfused")
        assert validate_bundle(bundle)["valid"], label
        static = json.loads((bundle / "instance-forward.json").read_text())
        assert static["status"] == "expanded_static", (label, static.get("issues"))
        got_nodes = static["nodes"]
        ref_nodes = original["nodes"]
        assert len(got_nodes) == len(ref_nodes), (label, len(got_nodes), len(ref_nodes))
        assert len(static["edges"]) == len(original["edges"]), (
            label, len(static["edges"]), len(original["edges"]))
        by_id = {node["id"]: node for node in got_nodes}
        assert len(by_id) == len(got_nodes), "Duplicate AE node IDs"
        id_to_fx = dict(zip((node["id"] for node in got_nodes),
                            (node["name"] for node in ref_nodes)))
        errors = []
        matches = 0
        for index, (a, b) in enumerate(zip(got_nodes, ref_nodes)):
            expected = (b["op"], b["target"], b["tensor_shape"], b["input_nodes"])
            observed = (a["op"], a["target"], a["shape"],
                        [id_to_fx[oid] for oid in a["input_ids"]])
            if observed != expected:
                errors.append({"index": index, "fx": expected, "ae": observed})
            else:
                matches += 1
            if not a["source_refs"] or not a["source_ir_ids"]:
                errors.append({"index": index, "missing_source_provenance": True})
            if a["op"] == "call_module" and a["module_instance"] != a["target"]:
                errors.append({"index": index, "missing_module_identity": True})
        actual_edges = {
            (id_to_fx[e["source"]], id_to_fx[e["target"]], e["input_slot"])
            for e in static["edges"]
        }
        expected_edges = {
            (e["source"], e["target"], e["slot"]) for e in original["edges"]
        }
        if actual_edges != expected_edges:
            errors.append({
                "kind": "operand_edge_mismatch",
                "missing_from_ae": sorted(expected_edges - actual_edges)[:8],
                "extra_in_ae": sorted(actual_edges - expected_edges)[:8],
            })
        for e in static["edges"]:
            src = by_id[e["source"]]
            dst = by_id[e["target"]]
            expected_relation = ("STATIC_INSTANCE_SHAPE_METADATA_DEPENDENCY"
                                 if src["shape"] is None or dst["shape"] is None
                                 else "STATIC_INSTANCE_TENSOR_DEPENDENCY")
            if e["relation"] != expected_relation:
                errors.append({"kind": "edge_kind", "edge": e["id"]})
        if static["output_shape"] != case["output_shape"]:
            errors.append({"kind": "output_shape"})
        if errors:
            (folder / "ae-mismatches.json").write_text(
                json.dumps(errors[:40], indent=2))
            raise AssertionError(f"{label}: {json.dumps(errors[:3], default=str)}")
        output_folder = folder / "diagram"
        presentation = present(bundle, output_folder, view="instance",
                               level="operation", format="all")
        assert presentation["status"] == "partial" and presentation["render_allowed"]
        manifest = json.loads((output_folder / "presentation-manifest.json").read_text())
        assert set(manifest["visual_elements"]) == set(by_id)
        assert set(manifest["visual_links"]) == {edge["id"] for edge in static["edges"]}
        assert all((output_folder / f"architecture.{ext}").stat().st_size > 0
                   for ext in ("svg", "pdf", "tex"))
        results.append({
            "case": label, "operations_matched": matches,
            "ordered_dependencies_matched": len(static["edges"]),
            "tensor_shapes_matched": sum(x["tensor_shape"] is not None for x in ref_nodes),
            "runtime_invocations_checked_by_reference": case["leaf_runtime_invocations"],
            "figure_nodes": len(manifest["visual_elements"]),
            "figure_edges": len(manifest["visual_links"]),
            "status": "verified_for_selected_input_and_attention_branch",
        })
    verified = {
        "model": "timm.vit_tiny_patch16_224",
        "scope": "pinned original timm 1.0.22; two batch sizes, fused/unfused branches",
        "status": "verified_for_selected_inputs",
        "results": results,
    }
    (output / "ae-acceptance-summary.json").write_text(json.dumps(verified, indent=2))
    print(json.dumps(verified, indent=2), flush=True)
    return verified


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark-output/vit-tiny-v020"))
    args = parser.parse_args()
    verify(args.output)
