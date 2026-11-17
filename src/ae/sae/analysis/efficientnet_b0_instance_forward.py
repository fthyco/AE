"""Audited original TorchVision 0.25.0 EfficientNet-B0 source-only expansion.

A bounded source AST specialization, not a general PyTorch execution engine.
No imports or execution of the target source during extraction.
"""
from __future__ import annotations

import ast
import hashlib

from ..ir.canonical import stable_id
from ..linking.project import Unit, span
from .instance_forward import StaticPathUnsupported, _check, _method


PINNED = {
    "torchvision/models/efficientnet.py":
        "82d058e1989ef84e4ef7ce43e4a49dc10f1b95dcd003303c2c052438de823b17",
    "torchvision/ops/misc.py":
        "a2bb1f3af1d3b854761ea4bc35d0b9deb5e49771f0dc919b390643f081a1ab7e",
    "torchvision/ops/stochastic_depth.py":
        "21267d9e827264bc694c6fb06b65663ecebaaa3855b0fed9c561efba65923f75",
    "torchvision/models/_utils.py":
        "b00ff45460ea07fa08e38495259f5954dfd2050587eb4c9398d0f6945df5f7b8",
}
ENTRYPOINTS = {
    "torchvision.models.efficientnet:efficientnet_b0", "efficientnet:efficientnet_b0",
}
BLUEPRINT = ((1,3,1,32,16,1),(6,3,2,16,24,2),(6,5,2,24,40,2),
             (6,3,2,40,80,3),(6,5,1,80,112,3),(6,5,2,112,192,4),
             (6,3,1,192,320,1))


class _Original:
    def __init__(self, units: list[Unit], ir_nodes: list[dict]):
        self.units = {}
        for path,digest in PINNED.items():
            found=[unit for unit in units if unit.path==path or
                   unit.path.endswith("/"+path)]
            _check(len(found)==1,"Original source missing/ambiguous: "+path)
            unit=found[0]
            _check(hashlib.sha256(unit.source.raw).hexdigest()==digest,
                   "Original TorchVision source changed: "+path)
            self.units[path]=unit
        self.ir_sites={}
        for node in ir_nodes:
            for ref in node.get("source_refs",[]):
                if ref.get("precision")=="exact":
                    key=(ref["path"],ref["start_byte"],ref["end_byte"])
                    self.ir_sites.setdefault(key,set()).add(node["id"])
        self.sites={}
        self.prove()

    def cls(self,path,name):
        found=[n for n in self.units[path].tree.body
               if isinstance(n,ast.ClassDef) and n.name==name]
        _check(len(found)==1,"Unresolved model class "+name)
        return found[0]

    def fn(self,path,name):
        found=[n for n in self.units[path].tree.body
               if isinstance(n,ast.FunctionDef) and n.name==name]
        _check(len(found)==1,"Unresolved factory "+name)
        return found[0]

    def site(self, key, path, root, predicate):
        found=sorted((n for n in ast.walk(root) if predicate(n)),
                     key=lambda n:(n.lineno,n.col_offset))
        _check(len(found)==1,"Unproven source operation "+key)
        self.sites[key]=(self.units[path],found[0])

    def prove(self):
        m="torchvision/models/efficientnet.py"
        o="torchvision/ops/misc.py"
        s="torchvision/ops/stochastic_depth.py"
        fac=self.fn(m,"efficientnet_b0")
        calls=[n for n in ast.walk(fac) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Name) and n.func.id=="_efficientnet_conf"]
        _check(len(calls)==1 and len(calls[0].args)==1 and
               ast.literal_eval(calls[0].args[0])=="efficientnet_b0",
               "B0 factory no longer selects the audited configuration")
        kwargs={k.arg:ast.literal_eval(k.value)
                for k in calls[0].keywords if k.arg in {"width_mult","depth_mult"}}
        _check(kwargs=={"width_mult":1.0,"depth_mult":1.0},
               "Nondefault width/depth multiplier")
        config=self.fn(m,"_efficientnet_conf")
        branches=[n for n in ast.walk(config) if isinstance(n,ast.If)
                  and isinstance(n.test,ast.Call) and
                  isinstance(n.test.func,ast.Attribute)
                  and n.test.func.attr=="startswith"
                  and len(n.test.args)==1
                  and isinstance(n.test.args[0],ast.Constant)
                  and n.test.args[0].value=="efficientnet_b"]
        _check(len(branches)==1,"B-family builder selection changed")
        cfgs=[node.value for node in branches[0].body
              if isinstance(node,ast.Assign) and any(
                  isinstance(t,ast.Name) and t.id=="inverted_residual_setting"
                  for t in node.targets)]
        _check(len(cfgs)==1 and isinstance(cfgs[0],ast.List),
               "Static EfficientNet 7-stage config unavailable")
        rows=[]
        for node in cfgs[0].elts:
            _check(isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                   and node.func.id=="bneck_conf" and len(node.args)==6
                   and not node.keywords, "Unsupported B0 stage config")
            rows.append(tuple(ast.literal_eval(a) for a in node.args))
        _check(tuple(rows)==BLUEPRINT,"B0 seven-stage blueprint differs from original")
        self.config=rows
        model=self.cls(m,"EfficientNet")
        block=self.cls(m,"MBConv")
        conv=self.cls(o,"ConvNormActivation")
        se=self.cls(o,"SqueezeExcitation")
        depth=self.cls(s,"StochasticDepth")
        forward=_method(model,"forward")
        impl=_method(model,"_forward_impl")
        bforward=_method(block,"forward")
        self.sites["input"]=(self.units[m],forward)
        ret=[n for n in impl.body if isinstance(n,ast.Return)]
        _check(len(ret)==1,"Unproven model output")
        self.sites["output"]=(self.units[m],ret[0])
        self.site("flatten",m,impl,lambda n:isinstance(n,ast.Call)
                  and isinstance(n.func,ast.Attribute) and n.func.attr=="flatten")
        init=_method(model,"__init__")
        self.sites["head"]=(self.units[m],init)
        conv_init=_method(conv,"__init__")
        for name in ("conv_layer","norm_layer","activation_layer"):
            self.site("conv:"+name,o,conv_init,lambda n,name=name:
                      isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                      and n.func.id==name)
        se_scale=_method(se,"_scale")
        for name in ("avgpool","fc1","activation","fc2","scale_activation"):
            self.site("se:"+name,o,se_scale,lambda n,name=name:
                      isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                      and isinstance(n.func.value,ast.Name)
                      and n.func.value.id=="self" and n.func.attr==name)
        se_forward=_method(se,"forward")
        self.site("se:mul",o,se_forward,lambda n:
                  isinstance(n,ast.BinOp) and isinstance(n.op,ast.Mult))
        self.site("residual:add",m,bforward,lambda n:
                  isinstance(n,ast.AugAssign) and isinstance(n.op,ast.Add))
        depth_forward=_method(depth,"forward")
        self.site("stochastic_depth",s,depth_forward,lambda n:
                  isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                  and n.func.id=="stochastic_depth")
        guards=[n for n in bforward.body if isinstance(n,ast.If)
                and isinstance(n.test,ast.Attribute)
                and n.test.attr=="use_res_connect"]
        _check(len(guards)==1 and len(guards[0].body)==2,
               "Original MBConv optional stochastic residual changed")
        _check(len([n for n in ast.walk(init) if isinstance(n,ast.Call)
                    and isinstance(n.func,ast.Attribute) and
                    n.func.attr=="AdaptiveAvgPool2d"])==1,
               "Global average pool was changed")

    def ref(self,key):
        unit,node=self.sites[key]
        ref=span(unit,node)
        ids=sorted(self.ir_sites.get(
            (ref["path"],ref["start_byte"],ref["end_byte"]),()))
        return ref,ids or [unit.source_id]


class _Graph:
    def __init__(self,original,shape):
        self.original=original;self.shape=shape
        self.nodes=[];self.edges=[];self.ordinals={}

    def emit(self,op,target,parents,shape,site):
        ref,ids=self.original.ref(site)
        fp=(ref["path"],ref["start_byte"],ref["end_byte"],op,target)
        ordinal=self.ordinals.get(fp,0);self.ordinals[fp]=ordinal+1
        ident=stable_id("ae-efficientnet-b0-v024",*fp,ordinal)
        self.nodes.append({"id":ident,"op":op,"target":target,
                           "call_ordinal":ordinal,
                           "module_instance":target if op=="call_module" else None,
                           "source_refs":[ref],"source_ir_ids":ids,
                           "shape":list(shape),"value_kind":"tensor",
                           "input_ids":parents[:]})
        for slot,source in enumerate(parents):
            self.edges.append({"id":stable_id("ae-efficientnet-b0-edge",source,ident,slot),
                               "source":source,"target":ident,
                               "input_slot":slot,
                               "relation":"STATIC_INSTANCE_TENSOR_DEPENDENCY"})
        return ident

    def seq(self,prefix,previous,b,c,h,w,stride,act):
        _check(stride in (1,2),"Unproven convolution stride")
        h=(h+stride-1)//stride;w=(w+stride-1)//stride
        for index,site in enumerate(("conv:conv_layer","conv:norm_layer")):
            previous=self.emit("call_module",f"{prefix}.{index}",[previous],
                               (b,c,h,w),site)
        if act:
            previous=self.emit("call_module",prefix+".2",[previous],
                               (b,c,h,w),"conv:activation_layer")
        return previous,h,w

    def se(self,prefix,x,b,channels,squeezed,h,w):
        first=x
        for name,ch,hh,ww in (
            ("avgpool",channels,1,1),
            ("fc1",squeezed,1,1),
            ("activation",squeezed,1,1),
            ("fc2",channels,1,1),
            ("scale_activation",channels,1,1),
        ):
            first=self.emit("call_module",prefix+"."+name,[first],
                            (b,ch,hh,ww),"se:"+name)
        return self.emit("call_function","mul",[first,x],
                         (b,channels,h,w),"se:mul")

    def run(self):
        _check(len(self.shape)==4 and all(type(x) is int and x>0 for x in self.shape),
               "Require four positive concrete dimensions")
        b,c,h,w=self.shape
        _check(c==3 and (h,w) in {(64,64),(128,128),(224,224)},
               "Only original RGB square 64/128/224 geometry supported")
        x=self.emit("placeholder","x",[],self.shape,"input")
        x,h,w=self.seq("features.0",x,b,32,h,w,2,True)
        c=32;residual=0;blocks=0;se_count=0
        for stage,(ratio,kernel,stride,inc,out,count) in enumerate(self.original.config,1):
            for repetition in range(count):
                ic=inc if repetition==0 else out
                oc=out
                s=stride if repetition==0 else 1
                _check(c==ic,"Source blueprint stage-channel continuity failure")
                skip=x;old_h,old_w=h,w
                expanded=ic*ratio
                prefix=f"features.{stage}.{repetition}.block"
                index=0
                if expanded!=ic:
                    x,h,w=self.seq(f"{prefix}.{index}",x,b,expanded,h,w,1,True)
                    index+=1
                x,h,w=self.seq(f"{prefix}.{index}",x,b,expanded,h,w,s,True)
                index+=1
                x=self.se(f"{prefix}.{index}",x,b,expanded,max(1,ic//4),h,w)
                index+=1;se_count+=1
                x,h,w=self.seq(f"{prefix}.{index}",x,b,oc,h,w,1,False)
                if s==1 and ic==oc:
                    _check((old_h,old_w)==(h,w),"Residual shape mismatch")
                    val=self.emit("call_function","stochastic_depth",[x],
                                  (b,oc,h,w),"stochastic_depth")
                    x=self.emit("call_function","add",[val,skip],
                                (b,oc,h,w),"residual:add")
                    residual+=1
                c=oc;blocks+=1
        _check((blocks,se_count,residual)==(16,16,9),
               "Unexpected EfficientNet stage/SE/residual count")
        x,h,w=self.seq("features.8",x,b,1280,h,w,1,True)
        x=self.emit("call_module","avgpool",[x],(b,1280,1,1),"head")
        x=self.emit("call_function","flatten",[x],(b,1280),"flatten")
        x=self.emit("call_module","classifier.0",[x],(b,1280),"head")
        x=self.emit("call_module","classifier.1",[x],(b,1000),"head")
        x=self.emit("output","output",[x],(b,1000),"output")
        _check((len(self.nodes),len(self.edges))==(251,275),
               "Static EfficientNet FX counts differ from audited original")
        return {"schema_version":"1.0","status":"expanded_static",
                "scope":"pinned_torchvision_0_25_0_efficientnet_b0",
                "source_path":self.original.units["torchvision/models/efficientnet.py"].path,
                "input_conditions":{"input_shape":list(self.shape),"weights":None,
                                    "width_mult":1.0,"depth_mult":1.0,
                                    "dropout":0.2,"eval":True},
                "output_shape":[b,1000],"nodes":self.nodes,
                "edges":self.edges,"unresolved":[]}


def expand_efficientnet_b0_forward(units:list[Unit],entrypoint:str|None,
                                   input_shape:tuple[int,...]|None,
                                   ir_nodes:list[dict])->dict|None:
    if entrypoint not in ENTRYPOINTS or input_shape is None:
        return None
    try:
        return _Graph(_Original(units,ir_nodes),input_shape).run()
    except (StaticPathUnsupported, ValueError, TypeError, KeyError, IndexError,
            AttributeError, OverflowError, RecursionError, SyntaxError) as exc:
        return {"schema_version":"1.0","status":"unsupported",
                "issues":[f"{type(exc).__name__}: {exc}"],"nodes":[],"edges":[]}
