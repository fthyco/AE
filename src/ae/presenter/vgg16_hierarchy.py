"""Source-evidence-conserving grouping of the exact audited VGG16 forward."""
from __future__ import annotations

from .instance_hierarchy import _focus, _summary, _unsupported


def project_vgg16_hierarchy(graph: dict, *, level: str,
                            focus: str | None, source_scope: str) -> dict:
    if source_scope != "pinned_torchvision_0_25_0_vgg16":
        return _unsupported(level, "Only source-pinned TorchVision 0.25.0 VGG16 is supported")
    stages=[]
    current="input"
    for node in graph["nodes"]:
        target=node["target"]
        if node["op"]=="placeholder":
            current="input"
        elif target.startswith("features."):
            suffix=target.split(".")[1]
            if not suffix.isdigit():
                return _unsupported(level,"Unknown VGG feature target")
            i=int(suffix)
            if not 0<=i<=30:
                return _unsupported(level,"Unexpected VGG feature position")
            current=(
                "stage.0" if i<=4 else "stage.1" if i<=9
                else "stage.2" if i<=16 else "stage.3" if i<=23
                else "stage.4"
            )
        elif target=="avgpool" or target=="flatten" or target.startswith("classifier."):
            current="classifier"
        elif node["op"]=="output":
            current="output"
        stages.append(current)
    expected={"input","classifier","output"}|{f"stage.{i}" for i in range(5)}
    if set(stages)!=expected:
        return _unsupported(level,"Audited VGG five pooling stages not fully present")
    if focus:
        if level!="operation":
            return _unsupported(level,"Focus requires operation detail")
        return _focus(graph, stages, focus)
    if level=="operation":return graph
    if level not in ("overview","block"):
        return _unsupported(level,"Unsupported VGG view")
    mapped=["feature_extractor" if level=="overview" and s.startswith("stage.") else s
            for s in stages]
    labels={"input":"RGB Input","feature_extractor":"VGG16 Feature Extractor",
            "classifier":"Adaptive Pool + 3 Linear Layers","output":"Output",
            **{f"stage.{i}":f"VGG Conv Stage {i+1}" for i in range(5)}}
    result=_summary(graph,mapped,level,model_key="torchvision-vgg16",
                    labels_override=labels,fallback_prefix="VGG ")
    if len(result["nodes"])!=(4 if level=="overview" else 8):
        return _unsupported(level,"Unexpected VGG grouping")
    return result
