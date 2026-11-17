"""Conservative original-source-only TorchVision 0.25.0 VGG16 expansion.

No execution or import of the analyzed project. Source factory parameters,
VGG configuration, forward method and candidate operation sites are derived
from pinned original bytes and AST.
"""
from __future__ import annotations

import ast
import hashlib

from ..ir.canonical import stable_id
from ..linking.project import Unit, span
from .instance_forward import StaticPathUnsupported, _check, _method

PINNED = {"torchvision/models/vgg.py":
          "214a5a421d7e5693315bb91c5087b47431660f39c905c97a806cc54840dfb04f"}
ENTRYPOINTS = {"torchvision.models.vgg:vgg16", "vgg:vgg16"}


class _VGG:
    def __init__(self, units, ir_nodes, shape):
        self.shape = shape
        matches = [u for u in units if u.path in PINNED or
                   u.path.endswith("/torchvision/models/vgg.py")]
        _check(len(matches) == 1, "Original VGG module not uniquely available")
        self.unit = matches[0]
        _check(hashlib.sha256(self.unit.source.raw).hexdigest() ==
               PINNED["torchvision/models/vgg.py"], "Original VGG source has changed")
        self.ids = {}
        for n in ir_nodes:
            for r in n.get("source_refs", []):
                if r.get("precision") == "exact":
                    self.ids.setdefault((r["path"], r["start_byte"], r["end_byte"]),
                                        set()).add(n["id"])
        self.sites = {}
        self._prove()
        self.nodes, self.edges, self.ordinals = [], [], {}

    def _fn(self, name):
        matches=[n for n in self.unit.tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == name]
        _check(len(matches)==1, "Unknown VGG function "+name)
        return matches[0]

    def _class(self, name):
        matches=[n for n in self.unit.tree.body
                 if isinstance(n, ast.ClassDef) and n.name == name]
        _check(len(matches)==1, "Unknown VGG class "+name)
        return matches[0]

    def _prove(self):
        _check(len(self.shape)==4 and all(type(v) is int and v>0 for v in self.shape),
               "Require positive 4-dimensional input shape")
        b,c,h,w=self.shape
        _check(c==3 and (h,w) in {(64,64),(128,128),(224,224)},
               "Only original RGB 64/128/224 square inputs supported")
        factory=self._fn("vgg16")
        calls=[n for n in ast.walk(factory) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Name) and n.func.id=="_vgg"]
        _check(len(calls)==1 and len(calls[0].args)==4 and
               ast.literal_eval(calls[0].args[0])=="D" and
               ast.literal_eval(calls[0].args[1]) is False,
               "Unproven VGG16 default factory")
        cfg=[n.value for n in self.unit.tree.body
             if ((isinstance(n,ast.AnnAssign)
                  and isinstance(n.target,ast.Name) and n.target.id=="cfgs")
                 or (isinstance(n,ast.Assign) and any(
                     isinstance(t,ast.Name) and t.id=="cfgs"
                     for t in n.targets)))]
        _check(len(cfg)==1 and isinstance(cfg[0],ast.Dict),
               "Original VGG configuration map missing")
        conf=ast.literal_eval(cfg[0])
        self.blueprint=conf["D"]
        _check(len(self.blueprint)==18 and self.blueprint.count("M")==5,
               "VGG-D blueprint differs from original")
        make=self._fn("make_layers")
        model=self._class("VGG")
        ctor=_method(model,"__init__")
        forward=_method(model,"forward")
        _check([type(n).__name__ for n in forward.body]==
               ["Assign","Assign","Assign","Assign","Return"],
               "VGG forward statements have changed")
        def find(root, predicate, name, *, expected_count=1, selected=0):
            found=sorted((n for n in ast.walk(root) if predicate(n)),
                         key=lambda n:(n.lineno,n.col_offset))
            _check(len(found)==expected_count,"Original VGG operation not uniquely proven: "+name)
            self.sites[name]=found[selected]
        self.sites["input"]=forward
        self.sites["output"]=forward.body[-1]
        find(make,lambda n:isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
             and n.func.attr=="Conv2d","conv")
        find(make,lambda n:isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
             and n.func.attr=="ReLU","relu",expected_count=2,selected=1)
        find(make,lambda n:isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
             and n.func.attr=="MaxPool2d","pool")
        find(ctor,lambda n:isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
             and n.func.attr=="AdaptiveAvgPool2d","avgpool")
        find(forward,lambda n:isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
             and n.func.attr=="flatten","flatten")
        self.sites["classifier"]=ctor
        _check(len([n for n in ast.walk(ctor) if isinstance(n,ast.Call)
                    and isinstance(n.func,ast.Attribute)
                    and n.func.attr=="Linear"])==3,
               "Original three Linear layers not proven")

    def _source(self, key):
        ref=span(self.unit,self.sites[key])
        ids=sorted(self.ids.get((ref["path"],ref["start_byte"],ref["end_byte"]), []))
        return ref, ids or [self.unit.source_id]

    def _add(self, op, target, parents, shape, site):
        ref, ids=self._source(site)
        fp=(ref["path"],ref["start_byte"],ref["end_byte"],op,target)
        idx=self.ordinals.get(fp,0);self.ordinals[fp]=idx+1
        uid=stable_id("ae-vgg16-v023",*fp,idx)
        self.nodes.append({"id":uid,"op":op,"target":target,"call_ordinal":idx,
                           "module_instance":target if op=="call_module" else None,
                           "source_refs":[ref],"source_ir_ids":ids,"shape":list(shape),
                           "value_kind":"tensor","input_ids":[p for p in parents]})
        for slot, p in enumerate(parents):
            self.edges.append({"id":stable_id("ae-vgg16-edge-v023",p,uid,slot),
                               "source":p,"target":uid,"input_slot":slot,
                               "relation":"STATIC_INSTANCE_TENSOR_DEPENDENCY"})
        return uid

    def run(self):
        b,c,h,w=self.shape
        current=self._add("placeholder","x",[],self.shape,"input")
        idx=0
        for val in self.blueprint:
            if val=="M":
                h,w=h//2,w//2
                current=self._add("call_module",f"features.{idx}",[current],
                                  (b,c,h,w),"pool");idx+=1
            else:
                c=val
                current=self._add("call_module",f"features.{idx}",[current],
                                  (b,c,h,w),"conv");idx+=1
                current=self._add("call_module",f"features.{idx}",[current],
                                  (b,c,h,w),"relu");idx+=1
        _check(idx==31 and c==512,"Unexpected VGG features layout")
        current=self._add("call_module","avgpool",[current],(b,512,7,7),"avgpool")
        current=self._add("call_function","flatten",[current],(b,25088),"flatten")
        for k,channels in enumerate((4096,4096,4096,4096,4096,4096,1000)):
            current=self._add("call_module",f"classifier.{k}",[current],
                              (b,channels),"classifier")
        current=self._add("output","output",[current],(b,1000),"output")
        _check((len(self.nodes),len(self.edges))==(42,41),
               "Unexpected VGG16 operation count")
        return {"schema_version":"1.0","status":"expanded_static",
                "scope":"pinned_torchvision_0_25_0_vgg16",
                "source_path":self.unit.path,
                "input_conditions":{"input_shape":list(self.shape),"weights":None,
                                    "batch_norm":False,"eval":True},
                "output_shape":[b,1000],"nodes":self.nodes,"edges":self.edges,"unresolved":[]}


def expand_vgg16_forward(units: list[Unit], entrypoint: str | None,
                         input_shape: tuple[int,...] | None, ir_nodes: list[dict]) -> dict | None:
    if entrypoint not in ENTRYPOINTS or input_shape is None:
        return None
    try:
        return _VGG(units,ir_nodes,input_shape).run()
    except (StaticPathUnsupported, ValueError,TypeError,KeyError,IndexError,
            AttributeError,RecursionError,OverflowError,SyntaxError) as exc:
        return {"schema_version":"1.0","status":"unsupported",
                "issues":[f"{type(exc).__name__}: {exc}"],"nodes":[],"edges":[]}
