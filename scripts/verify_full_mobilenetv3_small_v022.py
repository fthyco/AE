"""AE v0.22 independent end-to-end MobileNetV3-Small acceptance gate.

The original model is imported ONLY by the original-model preflight. SAE reads
audited source-file bytes without torch/torchvision imports or execution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ae.sae.api import extract, validate_bundle
from ae.presenter.api import present

from preflight_mobilenetv3_small_v022 import audit


def verify(output: Path) -> dict:
    ref = audit(output)
    results = []
    for case in ref["cases"]:
        label = case["configuration"]
        shape = tuple(case["shape"])
        folder = output / label
        original = json.loads((folder / "fx.json").read_text())
        bundle = folder / "ae-bundle"
        extract(output / "source", bundle,
                entrypoint="torchvision.models.mobilenetv3:mobilenet_v3_small",
                input_shape=shape)
        validity = validate_bundle(bundle)
        assert validity["valid"], (label, validity)
        data = json.loads((bundle / "instance-forward.json").read_text())
        assert data["status"] == "expanded_static", (label, data.get("issues"))
        got, fx = data["nodes"], original["nodes"]
        assert len(got) == len(fx) == 159, (label, len(got), len(fx))
        assert len(data["edges"]) == len(original["edges"]) == 173
        ids = {n["id"] for n in got}
        assert len(ids) == len(got), "Nonunique static invocation IDs"
        id_to_name = dict(zip((n["id"] for n in got), (n["name"] for n in fx)))
        mismatches = []
        for i, (a, b) in enumerate(zip(got, fx)):
            wanted = (b["op"], b["target"], b["shape"], b["inputs"])
            produced = (a["op"], a["target"], a["shape"],
                        [id_to_name[uid] for uid in a["input_ids"]])
            if produced != wanted:
                mismatches.append({"index": i, "expected": wanted, "observed": produced})
            if not a["source_refs"] or not a["source_ir_ids"]:
                mismatches.append({"index": i, "missing_source_provenance": True})
            if a["op"] == "call_module" and a["module_instance"] != a["target"]:
                mismatches.append({"index": i, "wrong_module_instance": True})
        ae_edges = {
            (id_to_name[e["source"]], id_to_name[e["target"]], e["input_slot"])
            for e in data["edges"]
        }
        fx_edges = {(e["source"], e["target"], e["slot"])
                    for e in original["edges"]}
        if ae_edges != fx_edges:
            mismatches.append({
                "edge_difference": True,
                "missing": sorted(fx_edges - ae_edges)[:20],
                "extra": sorted(ae_edges - fx_edges)[:20],
            })
        assert all(e["relation"] == "STATIC_INSTANCE_TENSOR_DEPENDENCY"
                   for e in data["edges"])
        if data["output_shape"] != case["output_shape"]:
            mismatches.append({"incorrect_output_shape": data["output_shape"]})
        if mismatches:
            (folder / "ae-fx-mismatch.json").write_text(
                json.dumps(mismatches[:40], indent=2))
            raise AssertionError(f"{label}: {json.dumps(mismatches[:3])}")
        result = present(bundle, folder / "figures", view="instance",
                         level="operation", format="all")
        assert result["render_allowed"] and result["status"] == "partial", (
            label, result)
        manifest = json.loads(
            (folder / "figures/presentation-manifest.json").read_text()
        )
        assert set(manifest["visual_elements"]) == ids
        assert set(manifest["visual_links"]) == {e["id"] for e in data["edges"]}
        for ext in ("svg", "pdf", "tex"):
            path = folder / "figures" / f"architecture.{ext}"
            assert path.stat().st_size > 100, (label, ext)
        results.append({
            "case": label, "operations": len(got),
            "ordered_edges": len(data["edges"]),
            "verified_shapes": sum(n["shape"] is not None for n in fx),
            "runtime_leaf_calls": case["runtime_leaf_module_calls"],
            "visual_nodes": len(manifest["visual_elements"]),
            "visual_edges": len(manifest["visual_links"]),
            "status": "source_only_ae_equals_original_fx_hooks_shapes",
        })
    summary = {"status": "verified_for_pinned_source_and_inputs",
               "scope": "TorchVision 0.25.0 MobileNetV3 Small default factory",
               "cases": results}
    (output / "ae-acceptance-summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=Path("benchmark-output/mobilenetv3-small-v022"))
    verify(ap.parse_args().output)
