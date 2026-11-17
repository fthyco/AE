"""Evidence-preserving two-level presentation for audited EfficientNet-B0."""
from __future__ import annotations
from .instance_hierarchy import _focus,_summary,_unsupported


def project_efficientnet_hierarchy(graph:dict, *, level:str,focus:str|None,
                                   source_scope:str)->dict:
    if source_scope!="pinned_torchvision_0_25_0_efficientnet_b0":
        return _unsupported(level,"Requires pinned TorchVision 0.25.0 EfficientNet-B0")
    stages=[];current="input"
    for n in graph["nodes"]:
        target=n["target"]
        if n["op"]=="placeholder":
            current="input"
        elif target.startswith("features."):
            pieces=target.split(".")
            if len(pieces)>2 and pieces[1].isdigit():
                index=int(pieces[1])
                if not 0<=index<=8:
                    return _unsupported(level,"Unexpected EfficientNet feature target")
                current=f"features.{index}"
        elif target=="avgpool" or target=="flatten" or target.startswith("classifier."):
            current="classifier"
        elif n["op"]=="output":
            current="output"
        # Functional SE, StochasticDepth, and residual add remain within their
        # previously source-proven feature stage.
        stages.append(current)
    expected={"input","classifier","output"}|{f"features.{i}" for i in range(9)}
    if set(stages)!=expected:
        return _unsupported(level,"Incomplete source-evidenced EfficientNet stages")
    if focus:
        if level!="operation":
            return _unsupported(level,"Only operation focus is supported")
        return _focus(graph,stages,focus)
    if level=="operation":return graph
    if level not in ("overview","block"):
        return _unsupported(level,"Unknown EfficientNet display level")
    transformed=["backbone" if level=="overview" and s in {
                 f"features.{i}" for i in range(1,8)} else s for s in stages]
    labels={"input":"RGB Input","features.0":"Stem Conv-BN-SiLU",
            "backbone":"EfficientNet-B0 MBConv Backbone",
            "features.8":"Final Conv-BN-SiLU",
            "classifier":"Global Pool / Classifier","output":"Output",
            **{f"features.{i}":f"MBConv Stage {i}" for i in range(1,8)}}
    result=_summary(graph,transformed,level,model_key="torchvision-efficientnet-b0",
                    labels_override=labels,fallback_prefix="EfficientNet ")
    if len(result["nodes"])!=(6 if level=="overview" else 12):
        return _unsupported(level,"Unexpected EfficientNet grouping")
    return result
