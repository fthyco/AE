"""Independent SMP U-Net FX and runtime reference inspection. Not used by AE extraction."""
from __future__ import annotations
import hashlib
import inspect
import json
import operator
from collections import Counter
from pathlib import Path

import torch
import torchvision
import segmentation_models_pytorch as smp
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp


def shapes(v):
    if isinstance(v, torch.Tensor):
        return list(v.shape)
    if isinstance(v,(tuple,list)):
        return [shapes(x) for x in v]
    return type(v).__name__


def main():
    out=Path("benchmark-output/unet-reference")
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    model=smp.Unet(encoder_name="resnet34",encoder_weights=None,
                   in_channels=3,classes=1).eval()
    fx=symbolic_trace(model)
    rows=[]
    for n in fx.graph.nodes:
        target = (str(n.target) if n.op in ("placeholder","output","call_module")
                  else getattr(n.target,"__name__",str(n.target)))
        rows.append({
            "name":n.name,"op":n.op,"target":target,
            "input_names":[m.name for m in n.all_input_nodes],
            "args_repr":repr(n.args)[:500],
            "kwargs_repr":repr(n.kwargs)[:250]})
    (out/"fx-events.json").write_text(json.dumps(rows,indent=2))
    mods=[]
    for name,child in model.named_modules():
        if not name or list(child.children()):continue
        mods.append({"name":name,"class":type(child).__name__,"repr":repr(child)[:500]})
    (out/"module-leaves.json").write_text(json.dumps(mods,indent=2))
    ref_sides=(64,96,224)
    summaries=[]
    for side in ref_sides:
        x=torch.zeros(1,3,side,side)
        ShapeProp(fx).propagate(x)
        visits=[];hooks=[]
        for name,child in model.named_modules():
            if name and not list(child.children()):
                hooks.append(child.register_forward_hook(
                    lambda obj,inp,output,name=name: visits.append(
                        {"module":name,"shapes":shapes(output)})))
        try:
            with torch.inference_mode():
                y=model(x)
        finally:
            for h in hooks:h.remove()
        metadata={}
        for n in fx.graph.nodes:
            t=n.meta.get("tensor_meta")
            metadata[n.name]={"tensor_shape": list(t.shape) if hasattr(t,"shape") else None,
                              "meta_type":str(n.meta.get("type_expr",""))}
        (out/f"shapes-{side}.json").write_text(json.dumps(metadata,indent=2))
        (out/f"hooks-{side}.json").write_text(json.dumps(visits,indent=2))
        summaries.append({"side":side,"output":shapes(y),"hooks":len(visits),
                          "tensor_nodes":sum(v["tensor_shape"] is not None for v in metadata.values())})
    source={}
    filenames=[
        "segmentation_models_pytorch.base.model",
        "segmentation_models_pytorch.base.modules",
        "segmentation_models_pytorch.base.heads",
        "segmentation_models_pytorch.decoders.unet.model",
        "segmentation_models_pytorch.decoders.unet.decoder",
        "segmentation_models_pytorch.encoders.resnet",
        "torchvision.models.resnet",
    ]
    import importlib
    for name in filenames:
        module=importlib.import_module(name)
        path=Path(module.__file__)
        source[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    report={
        "torch":torch.__version__, "torchvision":torchvision.__version__,
        "smp":smp.__version__,
        "source_sha256":source,
        "counts":{"nodes":len(rows),"edges":sum(len(r["input_names"]) for r in rows),
                  "modules":dict(Counter(r["op"] for r in rows)),
                  "targets":dict(Counter(r["target"] for r in rows if r["op"]=="call_function"))},
        "summary":summaries}
    (out/"summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
