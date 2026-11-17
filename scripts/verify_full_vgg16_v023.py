"""Independent VGG16 original FX, ShapeProp, runtime hooks vs AE source-only IR."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import torch
import torchvision
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from ae.sae.api import extract, validate_bundle
from ae.presenter.api import present


SOURCE = ("torchvision/models/vgg.py", "torchvision.models.vgg",
          "214a5a421d7e5693315bb91c5087b47431660f39c905c97a806cc54840dfb04f")
SHAPES = ((1,3,64,64),(1,3,128,128),(1,3,224,224),(2,3,224,224))


def snapshot(root: Path):
    path, module, expected = SOURCE
    raw=Path(importlib.import_module(module).__file__).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==expected, "Original VGG source drifted"
    target=root/path;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(raw)
    return {path:expected}


def verify(base: Path):
    assert torch.__version__.split("+")[0]=="2.10.0"
    assert torchvision.__version__.split("+")[0]=="0.25.0"
    torch.set_num_threads(1)
    base.mkdir(parents=True,exist_ok=True)
    source=base/"source"
    hashes=snapshot(source)
    model=torchvision.models.vgg16(weights=None).eval()
    assert sum(p.numel() for p in model.parameters())==138357544
    traced=symbolic_trace(model)
    entries=[]
    for shape in SHAPES:
        value=torch.zeros(*shape)
        ShapeProp(traced).propagate(value)
        ordered=[]
        originals=[]
        for n in traced.graph.nodes:
            meta=n.meta.get("tensor_meta")
            name=(str(n.target) if n.op in ("placeholder","call_module","output")
                  else getattr(n.target,"__name__",str(n.target)))
            originals.append({"id":n.name,"op":n.op,"target":name,
                              "shape":list(meta.shape) if hasattr(meta,"shape") else None,
                              "inputs":[v.name for v in n.all_input_nodes]})
            ordered.extend((v.name,n.name,i) for i,v in enumerate(n.all_input_nodes))
        hooks=[]
        handles=[]
        for name,module in model.named_modules():
            if name and not tuple(module.children()):
                handles.append(module.register_forward_hook(
                    lambda mod,inp,out,name=name: hooks.append(
                        {"target":name,"shape":list(out.shape)})))
        try:
            with torch.inference_mode():
                output=model(value)
        finally:
            for h in handles:h.remove()
        assert (len(originals),len(ordered),len(hooks))==(42,41,39)
        assert hooks==[{"target":n["target"],"shape":n["shape"]}
                       for n in originals if n["op"]=="call_module"]
        assert list(output.shape)==[shape[0],1000]
        label="x".join(map(str,shape));folder=base/label;folder.mkdir(exist_ok=True)
        (folder/"fx.json").write_text(json.dumps({"nodes":originals,"edges":ordered},indent=2))
        bundle=folder/"bundle"
        extract(source,bundle,entrypoint="torchvision.models.vgg:vgg16",
                input_shape=shape)
        assert validate_bundle(bundle)["valid"]
        found=json.loads((bundle/"instance-forward.json").read_text())
        assert found["status"]=="expanded_static",found.get("issues")
        nodes=found["nodes"]
        assert (len(nodes),len(found["edges"]))==(42,41)
        id_to_name={a["id"]:b["id"] for a,b in zip(nodes,originals)}
        for i,(a,b) in enumerate(zip(nodes,originals)):
            actual=(a["op"],a["target"],a["shape"],[id_to_name[x] for x in a["input_ids"]])
            expected=(b["op"],b["target"],b["shape"],b["inputs"])
            assert actual==expected,(shape,i,actual,expected)
            assert a["source_refs"] and a["source_ir_ids"]
        ae_edges={(id_to_name[x["source"]],id_to_name[x["target"]],x["input_slot"])
                  for x in found["edges"]}
        assert ae_edges==set(ordered)
        result=present(bundle,folder/"figures",view="instance",level="operation",
                       format="all")
        assert result["status"]=="partial" and result["render_allowed"]
        manifest=json.loads((folder/"figures/presentation-manifest.json").read_text())
        assert set(manifest["visual_elements"])=={n["id"] for n in nodes}
        assert set(manifest["visual_links"])=={e["id"] for e in found["edges"]}
        entries.append({"shape":list(shape),"fx_operations":42,"ordered_edges":41,
                        "runtime_hooks":39,"matching":True,"rendered":True})
    result={"model":"torchvision.models.vgg16","status":"source_and_fx_verified",
            "sources":hashes,"cases":entries}
    (base/"summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("benchmark-output/vgg16-v023"))
    verify(parser.parse_args().output)
