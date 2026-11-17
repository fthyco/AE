"""Source-only benchmark against pinned unmodified research repositories.

This test does NOT claim runtime/FX equivalence for dynamically configured models.
No third-party repository code is imported or executed.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import subprocess
from pathlib import Path

from ae.sae.api import extract, validate_bundle
from ae.presenter.api import present


CASES = {
    "awn": {
        "repo": "fthyco/Adaptive-Width-Neural-Networks-WITH-ORDERING-GUIDENCE",
        "commit": "e75ff0959a3b21ff997c393a26711e23745e96c7",
        "file": "model.py", "symbol": "AWN", "entrypoint": "model:AWN",
    },
    "chessformer-act": {
        "repo": "fthyco/maia3",
        "commit": "853785c4a8a512b6b2181520bfef21ff1a74566f",
        "file": "maia3/models_cf_act.py", "symbol": "MAIA3ActModel",
        "entrypoint": "maia3.models_cf_act:MAIA3ActModel",
    },
    "chessformer-cah": {
        "repo": "fthyco/maia3",
        "commit": "f9f66ed3feb7ab5a4498472d708c426ca8914436",
        "file": "maia3/models_cf_cah.py", "symbol": "MAIA3CahModel",
        "entrypoint": "maia3.models_cf_cah:MAIA3CahModel",
    },
}


def source_oracle(root: Path, spec: dict) -> dict:
    target = root / spec["file"]
    raw = target.read_bytes()
    tree = ast.parse(raw, filename=str(target))
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == spec["symbol"]]
    assert len(classes) == 1, "Original target class missing or ambiguous"
    cls = classes[0]
    methods = {node.name for node in cls.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "__init__" in methods and "forward" in methods, "Model interface changed"
    return {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "target_class": spec["symbol"],
        "target_methods": sorted(methods),
        "ast_class_lineno": cls.lineno,
        "has_conditional_construction": any(isinstance(n, ast.If) for n in ast.walk(cls)),
        "has_dynamic_iterations": any(isinstance(n, (ast.For, ast.While))
                                      for n in ast.walk(cls)),
    }


def git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, text=True, capture_output=True,
    )
    return result.stdout.strip()


def audit(case: str, root: Path, output: Path) -> dict:
    spec = CASES[case]
    root = root.resolve()
    output = output.resolve()
    if git_head(root) != spec["commit"]:
        raise AssertionError(f"Repository revision drift for {case}")
    oracle = source_oracle(root, spec)
    output.mkdir(parents=True, exist_ok=True)

    # Guard against accidental imports of target projects and ML runtimes.
    original_import = builtins.__import__
    forbidden = {"torch", "torchvision", "timm", "chess",
                 "model", "dynamic_architecture", "maia3", "graphyco"}
    def safe_import(name, *args, **kwargs):
        if name.split(".")[0] in forbidden:
            raise AssertionError("Source-only guarantee violated: " + name)
        return original_import(name, *args, **kwargs)

    builtins.__import__ = safe_import
    try:
        first = output / "run-a"
        second = output / "run-b"
        extract(root, first, entrypoint=spec["entrypoint"])
        extract(root, second, entrypoint=spec["entrypoint"])
        validation = validate_bundle(first)
        assert validation["valid"], "Canonical source bundle invalid"
        deterministic = all(
            (first / p).read_bytes() == (second / p).read_bytes()
            for p in ("architecture.ir.json", "model-index.json")
        )
        assert deterministic, "Nondeterministic static extraction"
        ir = json.loads((first / "architecture.ir.json").read_text())
        assert ir.get("entrypoint_target_id"), "Target entrypoint not resolved"
        assert ir["entrypoint_target_id"] in {n["id"] for n in ir["nodes"]}
        assert not any(e.get("relation") == "RUNTIME_TENSOR_FLOW"
                       for e in ir["edges"]), "Fabricated runtime edge"
        views = {}
        for level in ("overview", "block", "operation"):
            figures = output / ("view-" + level)
            result = present(first, figures, view="nested", level=level,
                             format="svg", mode="exploration")
            health = json.loads((figures / "presentation-health.json").read_text())
            manifest = json.loads((figures / "presentation-manifest.json").read_text())
            assert manifest["status"] == health["status"]
            assert manifest["rendered"] == result["render_allowed"]
            if not result["render_allowed"]:
                assert not (figures / "architecture.svg").exists()
            views[level] = {
                "status": health["status"],
                "render_allowed": result["render_allowed"],
                "visible_nodes": len(manifest["visual_elements"]),
                "visible_edges": len(manifest["visual_links"]),
            }
        structural = {}
        for level in ("overview", "block", "operation"):
            figures = output / ("structure-" + level)
            projection = present(first, figures, view="structure", level=level,
                                 format="svg", mode="exploration")
            health = json.loads((figures / "presentation-health.json").read_text())
            manifest = json.loads((figures / "presentation-manifest.json").read_text())
            assert manifest["status"] == health["status"]
            assert manifest["rendered"] == projection["render_allowed"]
            assert all(x.get("ir_node_ids") and x.get("source_refs")
                       for x in manifest["visual_elements"].values())
            assert all(x.get("ir_edge_ids") and x.get("source_refs")
                       for x in manifest["visual_links"].values())
            structural[level] = {
                "status": health["status"],
                "render_allowed": projection["render_allowed"],
                "visible_nodes": len(manifest["visual_elements"]),
                "visible_edges": len(manifest["visual_links"]),
                "semantic_categories": sorted(set(
                    x["semantic_category"]
                    for x in manifest["visual_elements"].values()
                    if x.get("semantic_category"))),
            }
        assert structural["overview"]["visible_nodes"] <= structural["block"]["visible_nodes"]
        assert structural["block"]["visible_nodes"] < structural["operation"]["visible_nodes"]
        overview_manifest = json.loads(
            (output / "structure-overview/presentation-manifest.json").read_text()
        )["visual_elements"]
        operation_manifest = json.loads(
            (output / "structure-operation/presentation-manifest.json").read_text()
        )["visual_elements"]
        root_groups = [n for n in overview_manifest.values() if n["kind"] == "model"]
        assert len(root_groups) == 1
        source_members = set(root_groups[0]["member_ir_node_ids"])
        proven_operation_ids = {
            nid for nid, n in operation_manifest.items()
            if n["kind"] == "operation" and source_members.intersection(n["ir_node_ids"])
        }
        assert proven_operation_ids, "No source-backed operation membership for root class"
    finally:
        builtins.__import__ = original_import

    return {
        "case": case,
        "source_repo": spec["repo"], "source_commit": spec["commit"],
        "entrypoint": spec["entrypoint"], "source_oracle": oracle,
        "source_only": True, "bundle_valid": True,
        "deterministic": True,
        "ir_nodes": len(ir["nodes"]), "ir_edges": len(ir["edges"]),
        "unresolved": len(ir.get("unresolved", [])),
        "instance_forward_equivalence": "not_verified",
        "tensor_shapes": "not_verified",
        "dynamic_control_flow": "not_proven",
        "views": views,
        "structural_views": structural,
        "source_backed_root_operation_sites": len(proven_operation_ids),
        "verdict": "source_index_and_partial_views_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.case, args.source, args.output)
    # Do not upload bundles/snapshots: AWN upstream has noncommercial restrictions.
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
