"""Render v0.21 provenance-preserving instance hierarchy from audited AE bundles.

Input bundles must already have passed scripts/verify_full_vit_tiny_v020.py.
This script only reads AE IR and renders the Presenter (never imports timm).
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from ae.presenter.api import present


def audit_and_render(root: Path) -> dict:
    rows = []
    for attention in ("fused", "unfused"):
        for batch in (1, 2):
            label = f"{attention}-{batch}x3x224x224"
            original_folder = root / label
            bundle = original_folder / "ae-bundle"
            original = json.loads((bundle / "instance-forward.json").read_text())
            original_nodes = {n["id"] for n in original["nodes"]}
            original_edges = {e["id"] for e in original["edges"]}
            for name, level, focus in (
                ("overview", "overview", None),
                ("block", "block", None),
                ("focus-block-0", "operation", "blocks.0"),
            ):
                output = original_folder / "hierarchy" / name
                result = present(bundle, output, view="instance", level=level,
                                 focus=focus, format="all")
                assert result["status"] == "partial" and result["render_allowed"], (
                    label, name, result)
                manifest = json.loads(
                    (output / "presentation-manifest.json").read_text()
                )
                visible_nodes = manifest["visual_elements"]
                visible_edges = manifest["visual_links"]
                if focus is None:
                    expected = 5 if level == "overview" else 16
                    assert len(visible_nodes) == expected, (label, level, len(visible_nodes))
                    represented = {
                        uid for e in visible_nodes.values() for uid in e["ir_node_ids"]
                    }
                    assert original_nodes <= represented
                    edges = {
                        uid for e in visible_edges.values() for uid in e["ir_edge_ids"]
                    }
                    collapsed = set(manifest["collapsed_instance_edge_ids"])
                    assert edges.isdisjoint(collapsed)
                    assert edges | collapsed == original_edges
                else:
                    assert set(visible_nodes) | set(
                        manifest["omitted_instance_node_ids"]) == original_nodes
                    assert set(visible_edges) | set(
                        manifest["omitted_instance_edge_ids"]) == original_edges
                for extension in ("svg", "pdf", "tex"):
                    assert (output / f"architecture.{extension}").stat().st_size > 200
                rows.append({
                    "configuration": label, "view": name,
                    "visual_nodes": len(visible_nodes),
                    "visual_links": len(visible_edges),
                    "original_nodes_accounted_for": len(original_nodes),
                    "original_edges_accounted_for": len(original_edges),
                    "status": "source-backed_grouping_verified",
                })

    overview = [r for r in rows
                if r["configuration"].startswith("fused-1x") or
                r["configuration"].startswith("unfused-1x")]
    cards = []
    for item in overview:
        label = item["configuration"]
        stage = item["view"]
        path = f"{label}/hierarchy/{stage}/architecture.svg"
        cards.append(
            f'<section><header><strong>{html.escape(label)} | '
            f'{html.escape(stage)}</strong> — '
            f'{item["visual_nodes"]} groups, {item["visual_links"]} visible links'
            f'</header><object type="image/svg+xml" data="{html.escape(path)}">'
            f'<a href="{html.escape(path)}">View original AE SVG</a>'
            f'</object><p><a href="{html.escape(path)}">SVG</a> · '
            f'<a href="{html.escape(path[:-3]+"pdf")}">PDF</a> · '
            f'<a href="{html.escape(path[:-3]+"tex")}">TikZ</a></p></section>'
        )
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>AE v0.21 ViT-Tiny hierarchy — original diagrams</title>'
        '<style>body{font-family:Arial,Helvetica,sans-serif;color:#111;'
        'background:#f7f7f7;margin:0 auto;max-width:1140px;padding:24px}'
        'h1{font-size:24px}p{color:#444}section{background:white;'
        'border:1px solid #ddd;margin:24px 0;padding:20px}'
        'section header{font-size:15px;margin-bottom:14px}'
        'object{width:100%;height:640px;border:1px solid #eee}'
        'a{color:#222}section p{font-size:13px}</style></head><body>'
        '<h1>Architecture Extractor — ViT Tiny</h1>'
        '<p>Original AE v0.21 SVG exports. Summarized views preserve '
        'source-backed operation/edge identities in their JSON manifests. '
        'No diagram was recreated manually. '
        'Original operation detail remains in the v0.20 viewer.</p>'
        + "\n".join(cards) + '</body></html>'
    )
    (root / "hierarchy-gallery.html").write_text(page)
    result = {"scope": "pinned timm 1.0.22 ViT Tiny", "status": "passed",
              "cases": rows}
    (root / "hierarchy-results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark-output/vit-tiny-v020"))
    args = parser.parse_args()
    audit_and_render(args.output)
