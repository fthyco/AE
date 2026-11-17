"""EfficientNet B0 source-bound and conservative negative acceptance."""
from __future__ import annotations
import hashlib,importlib,importlib.metadata,json,os
from pathlib import Path
import pytest
from ae.sae.api import extract,validate_bundle
from ae.presenter.api import present

@pytest.fixture(scope="module")
def snapshot():
    pytest.importorskip("torchvision")
    from scripts.verify_full_efficientnet_b0_v024 import SOURCES,snapshot
    version=importlib.metadata.version("torchvision").split("+")[0]
    changed=[p for p,(module,expected) in SOURCES.items() if
             hashlib.sha256(Path(importlib.import_module(module).__file__)
                            .read_bytes()).hexdigest()!=expected]
    if version!="0.25.0" or changed:
        msg=f"Audited original torchvision 0.25.0 source required; changed: {changed}"
        if os.environ.get("AE_REQUIRE_EFFICIENTNET_PIN")=="1":pytest.fail(msg)
        pytest.skip(msg)
    return snapshot

def run(tmp_path,snapshot,shape=(1,3,224,224),mutation=None):
    root=tmp_path/"source";snapshot(root)
    if mutation:
        rel,old,new=mutation
        file=root/rel;text=file.read_text();assert old in text
        file.write_text(text.replace(old,new,1))
    out=tmp_path/"bundle"
    extract(root,out,entrypoint="torchvision.models.efficientnet:efficientnet_b0",
            input_shape=shape)
    assert validate_bundle(out)["valid"]
    return json.loads((out/"instance-forward.json").read_text()),out

@pytest.mark.parametrize("shape",[(1,3,64,64),(1,3,128,128),(1,3,224,224),(2,3,224,224)])
def test_original_graph_and_operation_view(tmp_path,snapshot,shape):
    graph,bundle=run(tmp_path,snapshot,shape)
    assert graph["status"]=="expanded_static",graph.get("issues")
    assert (len(graph["nodes"]),len(graph["edges"]))==(251,275)
    assert graph["output_shape"]==[shape[0],1000]
    assert sum(n["target"]=="mul" for n in graph["nodes"])==16
    assert sum(n["target"]=="stochastic_depth" for n in graph["nodes"])==9
    assert sum(n["target"]=="add" for n in graph["nodes"])==9
    assert all(n["source_refs"] and n["source_ir_ids"] for n in graph["nodes"])
    fig=tmp_path/"figures"
    result=present(bundle,fig,view="instance",level="operation",format="all")
    assert result["render_allowed"] and result["status"]=="partial"
    manifest=json.loads((fig/"presentation-manifest.json").read_text())
    assert len(manifest["visual_elements"])==251
    assert len(manifest["visual_links"])==275

@pytest.mark.parametrize("shape",[(1,4,224,224),(1,3,80,80),(1,3,224,192),
                                  (1,3,-1,224),(0,3,224,224),(1,3,224)])
def test_non_audited_input_rejected(tmp_path,snapshot,shape):
    graph,_=run(tmp_path,snapshot,shape)
    assert graph["status"]=="unsupported" and not graph["nodes"]

@pytest.mark.parametrize("mutation",[
  ("torchvision/models/efficientnet.py","self.use_res_connect = cnf.stride == 1",
   "self.use_res_connect = cnf.stride == 2"),
  ("torchvision/ops/misc.py","return scale * input","return scale + input"),
  ("torchvision/ops/stochastic_depth.py","class StochasticDepth",
   "class StochasticDepthUnsafe"),
])
def test_original_source_mutation_rejected(tmp_path,snapshot,mutation):
    graph,_=run(tmp_path,snapshot,mutation=mutation)
    assert graph["status"]=="unsupported" and not graph["edges"]


@pytest.mark.parametrize("level,expected", [("overview",6),("block",12)])
def test_efficientnet_grouped_manifest_conserves_ids(tmp_path,snapshot,level,expected):
    graph,bundle=run(tmp_path,snapshot)
    folder=tmp_path/"hierarchy"
    r=present(bundle,folder,level=level,view="instance",format="all")
    assert r["render_allowed"] and r["status"]=="partial"
    manifest=json.loads((folder/"presentation-manifest.json").read_text())
    grouped=manifest["visual_elements"]
    links=manifest["visual_links"]
    assert len(grouped)==expected
    assert {n["id"] for n in graph["nodes"]}<={
        original for summary in grouped.values() for original in summary["ir_node_ids"]}
    visible={x for edge in links.values() for x in edge["ir_edge_ids"]}
    hidden=set(manifest["collapsed_instance_edge_ids"])
    assert visible.isdisjoint(hidden)
    assert visible|hidden=={e["id"] for e in graph["edges"]}
    import xml.etree.ElementTree as ET
    attrs=ET.parse(folder/"architecture.svg").getroot().attrib
    assert int(attrs["height"])>int(attrs["width"])

def test_efficientnet_stage_focus_preserves_omitted_ids(tmp_path,snapshot):
    graph,bundle=run(tmp_path,snapshot)
    folder=tmp_path/"focus"
    r=present(bundle,folder,level="operation",view="instance",focus="features.4")
    assert r["render_allowed"] and r["status"]=="partial"
    m=json.loads((folder/"presentation-manifest.json").read_text())
    assert (set(m["visual_elements"])|set(m["omitted_instance_node_ids"]))=={
        x["id"] for x in graph["nodes"]}
    assert (set(m["visual_links"])|set(m["omitted_instance_edge_ids"]))=={
        x["id"] for x in graph["edges"]}
