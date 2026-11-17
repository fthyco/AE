"""Original-only TorchVision EfficientNet-B0 vs source-only AE full acceptance."""
from __future__ import annotations
import argparse,hashlib,importlib,json
from pathlib import Path
import torch,torchvision
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from ae.sae.api import extract,validate_bundle
from ae.presenter.api import present

SOURCES={
 "torchvision/models/efficientnet.py":("torchvision.models.efficientnet","82d058e1989ef84e4ef7ce43e4a49dc10f1b95dcd003303c2c052438de823b17"),
 "torchvision/ops/misc.py":("torchvision.ops.misc","a2bb1f3af1d3b854761ea4bc35d0b9deb5e49771f0dc919b390643f081a1ab7e"),
 "torchvision/ops/stochastic_depth.py":("torchvision.ops.stochastic_depth","21267d9e827264bc694c6fb06b65663ecebaaa3855b0fed9c561efba65923f75"),
 "torchvision/models/_utils.py":("torchvision.models._utils","b00ff45460ea07fa08e38495259f5954dfd2050587eb4c9398d0f6945df5f7b8"),
}
SHAPES=((1,3,64,64),(1,3,128,128),(1,3,224,224),(2,3,224,224))


def snapshot(root):
    hashes={}
    for path,(module,expected) in SOURCES.items():
        raw=Path(importlib.import_module(module).__file__).read_bytes()
        sha=hashlib.sha256(raw).hexdigest()
        assert sha==expected,(path,sha)
        dst=root/path;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw)
        hashes[path]=sha
    return hashes


def verify(base):
    assert torch.__version__.split("+")[0]=="2.10.0"
    assert torchvision.__version__.split("+")[0]=="0.25.0"
    torch.set_num_threads(1);base.mkdir(parents=True,exist_ok=True)
    hashes=snapshot(base/"source")
    model=torchvision.models.efficientnet_b0(weights=None).eval()
    assert sum(p.numel() for p in model.parameters())==5288548
    assert len(model.features)==9
    traced=symbolic_trace(model)
    summary=[]
    gallery=[]
    for shape in SHAPES:
        value=torch.zeros(*shape)
        ShapeProp(traced).propagate(value)
        fx=[]
        for n in traced.graph.nodes:
            meta=n.meta.get("tensor_meta")
            fx.append({
                "name":n.name,"op":n.op,
                "target": str(n.target) if n.op in ("call_module","placeholder","output")
                        else getattr(n.target,"__name__",str(n.target)),
                "shape":list(meta.shape) if hasattr(meta,"shape") else None,
                "inputs":[x.name for x in n.all_input_nodes],
            })
        expected_edges={(n["name"],to["name"],i)
                        for to in fx for i,x in enumerate(to["inputs"])
                        for n in fx if n["name"]==x}
        assert (len(fx),len(expected_edges))==(251,275)
        assert sum(n["op"]=="call_module" for n in fx)==214
        assert sum(n["target"]=="mul" for n in fx)==16
        assert sum(n["target"]=="add" for n in fx)==9
        assert sum(n["target"]=="stochastic_depth" for n in fx)==9
        hooks=[];sd_hooks=[];handles=[]
        for name,mod in model.named_modules():
            if not name or tuple(mod.children()):continue
            if isinstance(mod,torchvision.ops.StochasticDepth):
                handles.append(mod.register_forward_hook(
                    lambda m,inp,out,name=name:sd_hooks.append(
                        {"name":name,"shape":list(out.shape)})))
            else:
                handles.append(mod.register_forward_hook(
                    lambda m,inp,out,name=name:hooks.append(
                        {"name":name,"shape":list(out.shape)})))
        try:
            with torch.inference_mode(): y=model(value)
        finally:
            for h in handles:h.remove()
        assert hooks==[{"name":n["target"],"shape":n["shape"]}
                       for n in fx if n["op"]=="call_module"],"FX leaf calls differ from runtime"
        assert len(sd_hooks)==9
        assert list(y.shape)==[shape[0],1000]
        label="x".join(map(str,shape));folder=base/label;folder.mkdir(exist_ok=True)
        (folder/"original-fx.json").write_text(json.dumps(fx,indent=2))
        (folder/"runtime-hooks.json").write_text(json.dumps(
            {"modules":hooks,"stochastic_depth":sd_hooks},indent=2))
        bundle=folder/"bundle"
        extract(base/"source",bundle,
                entrypoint="torchvision.models.efficientnet:efficientnet_b0",
                input_shape=shape)
        assert validate_bundle(bundle)["valid"]
        data=json.loads((bundle/"instance-forward.json").read_text())
        assert data["status"]=="expanded_static",data.get("issues")
        found=data["nodes"]
        assert (len(found),len(data["edges"]))==(251,275)
        id_to_name={a["id"]:b["name"] for a,b in zip(found,fx)}
        for i,(a,b) in enumerate(zip(found,fx)):
            actual=(a["op"],a["target"],a["shape"],
                    [id_to_name[x] for x in a["input_ids"]])
            expected=(b["op"],b["target"],b["shape"],b["inputs"])
            assert actual==expected,(label,i,actual,expected)
            assert a["source_refs"] and a["source_ir_ids"]
        actual_edges={(id_to_name[e["source"]],id_to_name[e["target"]],e["input_slot"])
                      for e in data["edges"]}
        assert actual_edges==expected_edges,(label,sorted(expected_edges-actual_edges)[:12],
                                              sorted(actual_edges-expected_edges)[:12])
        assert data["output_shape"]==[shape[0],1000]
        view=present(bundle,folder/"operation",level="operation",view="instance",
                     format="all")
        assert view["status"]=="partial" and view["render_allowed"]
        manifest=json.loads((folder/"operation/presentation-manifest.json").read_text())
        assert set(manifest["visual_elements"])==set(id_to_name)
        assert set(manifest["visual_links"])=={e["id"] for e in data["edges"]}
        for level,count in (("overview",6),("block",12)):
            out=folder / ("figures-"+level)
            proj=present(bundle,out,view="instance",level=level,format="all")
            assert proj["status"]=="partial" and proj["render_allowed"]
            evidence=json.loads((out/"presentation-manifest.json").read_text())
            assert len(evidence["visual_elements"])==count
            assert set(id_to_name)<={key for node in evidence["visual_elements"].values()
                                     for key in node["ir_node_ids"]}
            visible={key for edge in evidence["visual_links"].values()
                     for key in edge["ir_edge_ids"]}
            hidden=set(evidence["collapsed_instance_edge_ids"])
            assert visible.isdisjoint(hidden)
            assert visible|hidden=={edge["id"] for edge in data["edges"]}
            gallery.append((label,level,f"{label}/figures-{level}/architecture.svg"))
        summary.append({"shape":list(shape),"nodes":251,"ordered_edges":275,
                        "leaf_modules":214,"residuals":9,"SE_multiplications":16,
                        "original_sd_calls":9,"matched":True})
    report={"model":"torchvision.models.efficientnet_b0",
            "status":"source-only_mirrors_original_fx_and_runtime",
            "source_sha256":hashes,"cases":summary}
    (base/"summary.json").write_text(json.dumps(report,indent=2))
    cards=["<section><h2>"+name+" / "+level+"</h2>"
           '<object type="image/svg+xml" data="'+path+'"></object>'
           '<p><a href="'+path+'">Original source-backed SVG</a></p></section>'
           for name,level,path in gallery]
    (base/"hierarchy-gallery.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<title>AE — EfficientNet B0 gallery</title>'
        '<style>body{font:14px Arial,sans-serif;max-width:900px;margin:25px auto}'
        'section{padding:15px;margin:20px;border:1px solid #ddd}'
        'object{width:100%;height:700px}</style></head><body>'
        '<h1>EfficientNet-B0 — Original Architecture Extractor output</h1>'
        +''.join(cards)+'</body></html>')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--output",type=Path,
                        default=Path("benchmark-output/efficientnet-b0-v024"))
    verify(a.parse_args().output)
