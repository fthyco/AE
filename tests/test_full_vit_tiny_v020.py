"""v0.20 ViT Tiny source-only fixed-factory and conservative fail-closed tests."""
from __future__ import annotations

import json

import pytest

from ae.sae.api import extract, validate_bundle
from ae.presenter.api import present


@pytest.fixture(scope="module")
def originals():
    pytest.importorskip("timm")
    from scripts.preflight_vit_tiny_v020 import snapshot_sources
    return snapshot_sources


def prepare(tmp_path, originals, shape=(1, 3, 224, 224), attention="fused",
            changed=None, missing=None):
    root = tmp_path / "source"
    originals(root)
    if missing:
        (root / missing).unlink()
    if changed:
        relative, old, new = changed
        target = root / relative
        text = target.read_text()
        assert old in text
        target.write_text(text.replace(old, new, 1))
    folder = tmp_path / "bundle"
    extract(root, folder, entrypoint="timm.models.vision_transformer:vit_tiny_patch16_224",
            input_shape=shape, attention_mode=attention)
    assert validate_bundle(folder)["valid"]
    return json.loads((folder / "instance-forward.json").read_text()), folder


@pytest.mark.parametrize("mode, expected", [
    ("fused", (477, 596)),
    ("unfused", (549, 668)),
])
def test_fixed_factory_graph_and_render(tmp_path, originals, mode, expected):
    data, bundle = prepare(tmp_path, originals, attention=mode)
    assert data["status"] == "expanded_static", data.get("issues")
    assert (len(data["nodes"]), len(data["edges"])) == expected
    assert data["output_shape"] == [1, 1000]
    assert all(x["source_refs"] and x["source_ir_ids"] for x in data["nodes"])
    assert len({x["id"] for x in data["nodes"]}) == expected[0]
    assert sum(x["target"] == "scaled_dot_product_attention" for x in data["nodes"]) == (
        12 if mode == "fused" else 0)
    assert sum(x["target"] == "maybe_add_mask" for x in data["nodes"]) == (
        0 if mode == "fused" else 12)
    assert sum(x["target"] == "layer_norm" for x in data["nodes"]) == 25
    assert sum(x["target"] == "add" for x in data["nodes"]) == 25
    assert sum(x["target"].endswith(".attn.qkv") for x in data["nodes"]) == 12
    result = present(bundle, tmp_path / "figure", view="instance",
                     level="operation", format="all")
    assert result["status"] == "partial" and result["render_allowed"]
    manifest = json.loads((tmp_path / "figure/presentation-manifest.json").read_text())
    assert len(manifest["visual_elements"]) == expected[0]
    assert len(manifest["visual_links"]) == expected[1]


@pytest.mark.parametrize("shape", [
    (1, 3, 192, 192), (1, 3, 224, 256),
    (1, 4, 224, 224), (0, 3, 224, 224),
    (1, 3, -224, 224), (1, 3, 224),
])
def test_unsupported_geometry(tmp_path, originals, shape):
    graph, _ = prepare(tmp_path, originals, shape=shape)
    assert graph["status"] == "unsupported"
    assert not graph["nodes"] and not graph["edges"] and graph["issues"]


@pytest.mark.parametrize("mode", [None, "auto", "invalid"])
def test_implicit_or_unknown_attention_mode_rejected(tmp_path, originals, mode):
    graph, _ = prepare(tmp_path, originals, attention=mode)
    assert graph["status"] == "unsupported" and graph["nodes"] == []


@pytest.mark.parametrize("missing", [
    "timm/models/vision_transformer.py", "timm/layers/attention.py",
    "timm/layers/patch_embed.py", "timm/layers/mlp.py", "timm/layers/norm.py"
])
def test_missing_upstream_source(tmp_path, originals, missing):
    graph, _ = prepare(tmp_path, originals, missing=missing)
    assert graph["status"] == "unsupported" and not graph["nodes"]


@pytest.mark.parametrize("changed", [
    ("timm/layers/attention.py", "self.qkv(x)", "self.qkv(x + 0)"),
    ("timm/layers/patch_embed.py", "x = self.proj(x)", "x = self.norm(x)"),
    ("timm/models/vision_transformer.py", "embed_dim=192", "embed_dim=256"),
])
def test_mutated_original_source_rejected(tmp_path, originals, changed):
    graph, _ = prepare(tmp_path, originals, changed=changed)
    assert graph["status"] == "unsupported"
    assert not graph["nodes"] and not graph["edges"]


def test_batch_two(tmp_path, originals):
    graph, _ = prepare(tmp_path, originals, shape=(2, 3, 224, 224))
    assert graph["status"] == "expanded_static"
    assert graph["output_shape"] == [2, 1000]
