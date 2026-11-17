"""Minimal semantic operation identity for audited, source-backed instance graphs.

Color is a secondary cue. Text labels, source refs and exact IR IDs remain
the authoritative identity; unknown module kinds stay neutral.
"""
from __future__ import annotations

# Muted scientific accents; never a rainbow per node.
COLORS = {
    "linear": "#527996",
    "normalization": "#4F8F83",
    "activation": "#BC8A4B",
    "attention": "#826EA4",
    "elementwise": "#68798A",
    "pool_or_reshape": "#898D93",
    "regularization": "#979197",
    "matrix_multiplication": "#536D7A",
    "unknown": "#B0B0B0",
}
LABELS = {
    "linear": "Conv / Linear",
    "normalization": "Normalization",
    "activation": "Activation",
    "attention": "Attention",
    "elementwise": "Elementwise / Residual",
    "pool_or_reshape": "Pool / Reshape",
    "regularization": "Regularization",
    "matrix_multiplication": "Matrix Multiplication",
    "unknown": "Other / Unclassified",
}

_VGG_CONV = {0,2,5,7,10,12,14,17,19,21,24,26,28}
_VGG_RELU = {1,3,6,8,11,13,15,18,20,22,25,27,29}
_VGG_POOL = {4,9,16,23,30}


def semantic_category(op: str, target: str, scope: str) -> str:
    """Classify only where the validated original source gives enough evidence."""
    if op in ("placeholder","output"):
        return "unknown"
    if op=="call_function":
        if target in {"add","mul","sub","div","operator.add","operator.mul"}:
            return "elementwise"
        if target=="scaled_dot_product_attention":
            return "attention"
        if target=="softmax":
            return "activation"
        if target in {"matmul","bmm"}:
            return "matrix_multiplication"
        if target in {"flatten","reshape","view","permute","transpose","cat","concat"}:
            return "pool_or_reshape"
        if target=="stochastic_depth":
            return "regularization"
        return "unknown"
    if op!="call_module":
        return "unknown"
    if scope=="pinned_torchvision_0_25_0_vgg16":
        if target.startswith("features."):
            index=target.split(".")[1]
            if index.isdigit():
                i=int(index)
                if i in _VGG_CONV:return "linear"
                if i in _VGG_RELU:return "activation"
                if i in _VGG_POOL:return "pool_or_reshape"
        if target=="avgpool":return "pool_or_reshape"
        if target.startswith("classifier."):
            index=target.split(".")[1]
            if index in {"0","3","6"}:return "linear"
            if index in {"1","4"}:return "activation"
            if index in {"2","5"}:return "regularization"
        return "unknown"
    if scope in {
        "pinned_torchvision_0_25_0_mobilenet_v3_small",
        "pinned_torchvision_0_25_0_efficientnet_b0",
    }:
        leaf=target.split(".")[-1]
        if leaf in {"avgpool"}:return "pool_or_reshape"
        if leaf in {"fc1","fc2"}:return "linear"
        if leaf in {"activation","scale_activation"}:return "activation"
        if target.startswith("classifier."):
            index=target.split(".")[-1]
            if scope.endswith("efficientnet_b0"):
                return "linear" if index=="1" else "regularization"
            return ("linear" if index in {"0","3"} else
                    "activation" if index=="1" else "regularization")
        parts=target.split(".")
        if "block" in parts or target.startswith(("features.0.","features.8.","features.12.")):
            # Audited ConvNormActivation leaves: .0 Conv, .1 BN, .2 activation.
            if leaf=="0":return "linear"
            if leaf=="1":return "normalization"
            if leaf=="2":return "activation"
        if target=="avgpool":return "pool_or_reshape"
        return "unknown"
    if scope=="pinned_timm_1_0_22_vit_tiny_patch16_224":
        if "norm" in target or "layer_norm" in target:return "normalization"
        if ".attn." in target or target.startswith("attn."):return "attention"
        if any(label in target for label in (".fc1",".fc2",".proj",".qkv","patch_embed.proj")):
            return "linear"
        if "act" in target or "gelu" in target:return "activation"
        if "pool" in target:return "pool_or_reshape"
    return "unknown"

# Conservative classification for source-level component declarations.
# A constructor type is a lexical declaration, not proof of runtime execution.
_DECLARED_TYPES = {
    "linear": frozenset({"Linear", "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "Embedding", "Bilinear"}),
    "normalization": frozenset({"BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm", "GroupNorm", "InstanceNorm1d", "InstanceNorm2d", "RMSNorm"}),
    "activation": frozenset({"ReLU", "ReLU6", "GELU", "SiLU", "Sigmoid", "Tanh", "Softmax", "LeakyReLU", "ELU", "Hardswish", "Hardsigmoid"}),
    "attention": frozenset({"MultiheadAttention"}),
    "pool_or_reshape": frozenset({"MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d", "AvgPool2d", "AvgPool3d", "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveMaxPool2d", "Flatten", "Unflatten", "Upsample"}),
    "regularization": frozenset({"Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "AlphaDropout"}),
}
_STATIC_OPERATORS = {
    "elementwise": frozenset({"Add", "Sub", "Mult", "Div", "Pow", "Mod"}),
    "matrix_multiplication": frozenset({"MatMult"}),
    "pool_or_reshape": frozenset({"Reshape", "View", "Flatten", "Permute", "Transpose", "Cat", "Concat"}),
}


def source_category(kind: str, label: str) -> str:
    """Classify only explicit source constructor/operator spellings."""
    if kind not in {"component", "nested-constructor", "operation"}:
        return "unknown"
    part = label.rsplit(":", 1)[-1].strip().split("(")[0].split(".")[-1]
    groups = _STATIC_OPERATORS if kind == "operation" else _DECLARED_TYPES
    for name, identifiers in groups.items():
        if part in identifiers:
            return name
    return "unknown"
