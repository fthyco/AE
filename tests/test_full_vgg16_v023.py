"""VGG16 original-source gated static extraction and fail-closed tests."""
from __future__ import annotations
import hashlib, importlib, importlib.metadata, json, os
from pathlib import Path
import pytest
from ae.sae.api import extract,validate_bundle
from ae.presenter.api import present

@pytest.fixture(scope="module")
def snapshot():
    pytest.importorskip("torchvision")
    from scripts.verify_full_vgg16_v023 import SOURCE, snapshot
    rel, module, expected=SOURCE
    version=importlib.metadata.version("torchvision").split("+")[0]
    content=Path(importlib.import_module(module).__file__).read_bytes()
    if version!="0.25.0" or hashlib.sha256(content).hexdigest()!=expected:
        message="VGG16 test requires original SHA256-pinned torchvision 0.25.0"
        if os.environ.get("AE_REQUIRE_VGG_PIN")=="1":pytest.fail(message)
        pytest.skip(message)
    return snapshot


def run(tmp_path,snapshot,shape=(1,3,224,224),modify=None):
    root=tmp_path/"source";snapshot(root)
    if modify:
        rel,old,new=modify;file=root/rel
        body=file.read_text();assert old in body
        file.write_text(body.replace(old,new,1))
    bundle=tmp_path/"bundle"
    extract(root,bundle,entrypoint="torchvision.models.vgg:vgg16",
            input_shape=shape)
    assert validate_bundle(bundle)["valid"]
    return json.loads((bundle/"instance-forward.json").read_text()),bundle


@pytest.mark.parametrize("shape",[(1,3,64,64),(1,3,128,128),(1,3,224,224),(2,3,224,224)])
def test_original_vgg16_graph_and_vector(tmp_path,snapshot,shape):
    graph,bundle=run(tmp_path,snapshot,shape)
    assert graph["status"]=="expanded_static",graph.get("issues")
    assert (len(graph["nodes"]),len(graph["edges"]))==(42,41)
    assert graph["output_shape"]==[shape[0],1000]
    assert len({x["id"] for x in graph["nodes"]})==42
    assert all(x["source_refs"] and x["source_ir_ids"] for x in graph["nodes"])
    r=present(bundle,tmp_path/"figures",view="instance",level="operation",format="all")
    assert r["render_allowed"] and r["status"]=="partial"


@pytest.mark.parametrize("shape",[(1,4,224,224),(1,3,0,224),(1,3,224,192),(1,3,80,80),(1,3,224)])
def test_unsupported_geometry_fails_closed(tmp_path,snapshot,shape):
    graph,_=run(tmp_path,snapshot,shape)
    assert graph["status"]=="unsupported" and not graph["nodes"] and not graph["edges"]


@pytest.mark.parametrize("mutation",[
("torchvision/models/vgg.py","return _vgg(\"D\", False, weights, progress, **kwargs)","return _vgg(\"D\", True, weights, progress, **kwargs)"),
("torchvision/models/vgg.py","self.avgpool = nn.AdaptiveAvgPool2d((7, 7))","self.avgpool = nn.AdaptiveAvgPool2d((8, 8))"),
])
def test_unsupported_mutation_fails_closed(tmp_path,snapshot,mutation):
    graph,_=run(tmp_path,snapshot,modify=mutation)
    assert graph["status"]=="unsupported" and not graph["nodes"]
