"""Audited MobileNetV3-Small instance hierarchy with lossless ID accounting.

This groups an already source-validated graph. It does not infer additional
conv, SE, residual or runtime operations.
"""
from __future__ import annotations

from .instance_hierarchy import _focus, _summary, _unsupported


_SCOPE = "pinned_torchvision_0_25_0_mobilenet_v3_small"


def _stages(nodes: list[dict]) -> list[str]:
    current = "input"
    result = []
    for n in nodes:
        target = n["target"]
        if n["op"] == "placeholder":
            stage = "input"
        elif n["op"] == "output":
            stage = "output"
        elif target.startswith("features."):
            parts = target.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                stage = f"features.{parts[1]}"
            else:
                stage = current
        elif target == "avgpool" or target == "flatten" or target.startswith("classifier."):
            stage = "classifier"
        else:
            # FX-inlined SqueezeExcitation.mul and residual.add are in the
            # immediately preceding inverted residual block, not in a new
            # fake feature layer.
            stage = current
        current = stage
        result.append(stage)
    return result


def project_mobilenet_hierarchy(graph: dict, *,
                                level: str, focus: str | None,
                                source_scope: str) -> dict:
    if source_scope != _SCOPE:
        return _unsupported(level, "MobileNet hierarchy requires the audited original source")
    stages = _stages(graph["nodes"])
    required = {"input", "output", "classifier", "features.0",
                "features.12"} | {f"features.{i}" for i in range(1, 12)}
    if set(stages) != required:
        return _unsupported(level, "Original 11 blocks, stem and final convolution not fully observed")
    if focus is not None:
        if level != "operation":
            return _unsupported(level, "Focus requires operation level")
        return _focus(graph, stages, focus)
    if level == "operation":
        return graph
    if level not in ("overview", "block"):
        return _unsupported(level, "Unsupported MobileNet projection level")
    transformed = [
        "backbone" if level == "overview" and s in {
            f"features.{i}" for i in range(1, 12)
        } else s
        for s in stages
    ]
    labels = {
        "input": "RGB Input",
        "features.0": "Stem Conv-BN-HSwish",
        "backbone": "11 Inverted Residual Blocks",
        "features.12": "Final Conv-BN-HSwish",
        "classifier": "Global Pool + Classifier",
        "output": "Output",
        **{f"features.{i}": f"Inverted Residual {i}" for i in range(1, 12)},
    }
    expected = 6 if level == "overview" else 16
    result = _summary(
        graph, transformed, level, model_key="torchvision-mobilenetv3-small",
        labels_override=labels, fallback_prefix="MobileNet "
    )
    if len(result["nodes"]) != expected:
        return _unsupported(level, "Unexpected MobileNet visual hierarchy")
    return result
