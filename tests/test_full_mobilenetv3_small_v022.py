"""TorchVision 0.25.0 MobileNetV3-Small bounded source-only acceptance tests."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path

import pytest

from ae.sae.api import extract, validate_bundle
from ae.presenter.api import present


@pytest.fixture(scope="module")
def originals():
    pytest.importorskip("torchvision")
    from scripts.preflight_mobilenetv3_small_v022 import SOURCES, snapshot_sources
    version = importlib.metadata.version("torchvision")
    changed = [
        path for path, (module, digest) in SOURCES.items()
        if hashlib.sha256(Path(importlib.import_module(module).__file__)
                          .read_bytes()).hexdigest() != digest
    ]
    if version.split("+")[0] != "0.25.0" or changed:
        message = ("Original torchvision==0.25.0 with matching source bytes required: "
                   f"installed {version}; changed {changed}")
        if os.environ.get("AE_REQUIRE_MNV3_PIN") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return snapshot_sources


def prepare(tmp_path, originals, *, shape=(1, 3, 224, 224),
            entrypoint="torchvision.models.mobilenetv3:mobilenet_v3_small",
            modify=None, missing=None):
    source = tmp_path / "source"
    originals(source)
    if modify:
        rel, old, new = modify
        file = source / rel
        body = file.read_text()
        assert old in body
        file.write_text(body.replace(old, new, 1))
    if missing:
        (source / missing).unlink()
    bundle = tmp_path / "bundle"
    extract(source, bundle, entrypoint=entrypoint, input_shape=shape)
    assert validate_bundle(bundle)["valid"]
    return json.loads((bundle / "instance-forward.json").read_text()), bundle


@pytest.mark.parametrize("shape", [
    (1,3,64,64),(1,3,128,128),(1,3,224,224),(2,3,224,224),
])
def test_full_static_mobile_graph_and_vector_projection(tmp_path, originals, shape):
    graph,bundle=prepare(tmp_path,originals,shape=shape)
    assert graph["status"] == "expanded_static", graph.get("issues")
    assert (len(graph["nodes"]), len(graph["edges"])) == (159,173)
    assert graph["output_shape"] == [shape[0],1000]
    assert sum(n["op"]=="call_module" for n in graph["nodes"]) == 141
    assert sum(n["target"]=="mul" for n in graph["nodes"]) == 9
    assert sum(n["target"]=="add" for n in graph["nodes"]) == 6
    assert sum(n["target"]=="flatten" for n in graph["nodes"]) == 1
    assert len({n["id"] for n in graph["nodes"]}) == 159
    assert all(n["source_refs"] and n["source_ir_ids"] for n in graph["nodes"])
    assert all(e["relation"] == "STATIC_INSTANCE_TENSOR_DEPENDENCY"
               for e in graph["edges"])
    figure=tmp_path/"figure"
    output=present(bundle,figure,view="instance",level="operation",format="all")
    assert output["status"]=="partial" and output["render_allowed"]
    manifest=json.loads((figure/"presentation-manifest.json").read_text())
    assert len(manifest["visual_elements"]) == 159
    assert len(manifest["visual_links"]) == 173
    assert all((figure/f"architecture.{s}").stat().st_size>100
               for s in ("svg","pdf","tex"))


@pytest.mark.parametrize("shape",[
    (1,1,224,224),(1,4,224,224),(1,3,80,80),
    (1,3,224,192),(1,3,0,224),(1,3,-1,224),(1,3,224),
])
def test_non_audited_input_fails_closed(tmp_path,originals,shape):
    graph,_=prepare(tmp_path,originals,shape=shape)
    assert graph["status"]=="unsupported" and not graph["nodes"] and not graph["edges"]


@pytest.mark.parametrize("missing",[
    "torchvision/models/mobilenetv3.py",
    "torchvision/ops/misc.py",
    "torchvision/models/_utils.py",
])
def test_missing_original_source_is_rejected(tmp_path,originals,missing):
    graph,_=prepare(tmp_path,originals,missing=missing)
    assert graph["status"]=="unsupported" and not graph["nodes"]


@pytest.mark.parametrize("change",[
    ("torchvision/models/mobilenetv3.py","self.use_res_connect = cnf.stride == 1",
     "self.use_res_connect = cnf.stride == 2"),
    ("torchvision/ops/misc.py","return scale * input","return scale + input"),
    ("torchvision/models/_utils.py","def _make_divisible(",
     "def _make_divisible_unsafe("),
])
def test_mutation_rejected_before_emitting_graph(tmp_path,originals,change):
    graph,_=prepare(tmp_path,originals,modify=change)
    assert graph["status"]=="unsupported" and not graph["nodes"] and not graph["edges"]


def test_small_factory_entrypoint_variant(tmp_path,originals):
    data,_=prepare(tmp_path,originals,entrypoint="mobilenetv3:mobilenet_v3_small")
    assert data["status"]=="expanded_static" and len(data["nodes"])==159


@pytest.mark.parametrize("level,count", [("overview",6), ("block",16)])
def test_mobile_hierarchy_preserves_source_operation_and_edge_ids(
        tmp_path,originals,level,count):
    graph,bundle=prepare(tmp_path,originals)
    folder=tmp_path/"hierarchy"
    result=present(bundle,folder,view="instance",level=level,format="all")
    assert result["status"]=="partial" and result["render_allowed"]
    manifest=json.loads((folder/"presentation-manifest.json").read_text())
    visual_nodes=manifest["visual_elements"]
    visual_edges=manifest["visual_links"]
    assert len(visual_nodes)==count
    assert all(item["source_refs"] for item in visual_nodes.values())
    represented={uid for entry in visual_nodes.values()
                 for uid in entry["ir_node_ids"]}
    assert {n["id"] for n in graph["nodes"]} <= represented
    visible={eid for entry in visual_edges.values()
             for eid in entry["ir_edge_ids"]}
    collapsed=set(manifest["collapsed_instance_edge_ids"])
    assert visible.isdisjoint(collapsed)
    assert visible|collapsed=={e["id"] for e in graph["edges"]}
    assert not manifest["omitted_instance_node_ids"]
    assert not manifest["omitted_instance_edge_ids"]
    import xml.etree.ElementTree as ET
    dimensions=ET.parse(folder/"architecture.svg").getroot().attrib
    assert int(dimensions["height"])>int(dimensions["width"])


def test_mobile_focus_keeps_immediate_dependency_context(tmp_path,originals):
    graph,bundle=prepare(tmp_path,originals)
    dest=tmp_path/"focused"
    result=present(bundle,dest,view="instance",level="operation",
                   focus="features.4",format="all")
    assert result["status"]=="partial" and result["render_allowed"]
    manifest=json.loads((dest/"presentation-manifest.json").read_text())
    seen=set(manifest["visual_elements"])
    seen_edges=set(manifest["visual_links"])
    assert 0<len(seen)<len(graph["nodes"])
    assert seen|set(manifest["omitted_instance_node_ids"])=={
        node["id"] for node in graph["nodes"]
    }
    assert seen_edges|set(manifest["omitted_instance_edge_ids"])=={
        edge["id"] for edge in graph["edges"]
    }
    assert not (dest/"architecture.pdf").stat().st_size==0


def test_mobile_invalid_focus_blocks_without_stale_diagram(tmp_path,originals):
    _,bundle=prepare(tmp_path,originals)
    dest=tmp_path/"invalid"
    result=present(bundle,dest,view="instance",level="operation",focus="features.99")
    assert result["status"]=="blocked"
    assert not (dest/"architecture.svg").exists()
