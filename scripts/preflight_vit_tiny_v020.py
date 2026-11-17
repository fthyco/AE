"""Independent timm ViT-Tiny reference inventory for AE v0.20.

This is a *preflight reference*, not a proof of AE extraction coverage.
It runs the original timm model only in this verifier, never inside SAE.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path

import torch
import timm
from timm.layers.config import set_fused_attn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp


MODEL_NAME = "vit_tiny_patch16_224"
TIMM_VERSION = "1.0.22"
SOURCE_MODULES = {
    "timm/models/vision_transformer.py": "timm.models.vision_transformer",
    "timm/layers/attention.py": "timm.layers.attention",
    "timm/layers/patch_embed.py": "timm.layers.patch_embed",
    "timm/layers/mlp.py": "timm.layers.mlp",
    "timm/layers/norm.py": "timm.layers.norm",
}


EXPECTED_SOURCE_SHA256 = {
    "timm/models/vision_transformer.py": "fbbfd91ae654ec3d32ccea05002177efee21281c6db73493c4ea79acc3d7b212",
    "timm/layers/attention.py": "5fcb14cb16e7382a4845d7aa97328253745827a6f1399e60632ad527fd8c1e07",
    "timm/layers/patch_embed.py": "7b7e670f8e6a2828752a65297025fb903fc2fbe0298617a3cd29d804be1455aa",
    "timm/layers/mlp.py": "c024128a1c7e2a55ebc5e33a7c94c585ae1e960cd032b39b08445ea30f05c5a5",
    "timm/layers/norm.py": "afe3abf01a4379065c1c11bf6b6f947a1a743aff7f5e51849b353697b0db557d",
}
EXPECTED_COUNTS = {
    True: (477, 596, 405, 225, 200, 25),
    False: (549, 668, 477, 237, 212, 25),
}

def snapshot_sources(destination: Path) -> dict[str, str]:
    result = {}
    for rel, module in SOURCE_MODULES.items():
        original = Path(importlib.import_module(module).__file__)
        raw = original.read_bytes()
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        assert target.read_bytes() == raw
        result[rel] = hashlib.sha256(raw).hexdigest()
    return result


def fx_record(node) -> dict:
    meta = node.meta.get("tensor_meta")
    shape = list(meta.shape) if hasattr(meta, "shape") else None
    return {
        "name": node.name,
        "op": node.op,
        "target": (
            str(node.target) if node.op in ("placeholder", "output", "call_module")
            else getattr(node.target, "__name__", str(node.target))
        ),
        "input_nodes": [dep.name for dep in node.all_input_nodes],
        "tensor_shape": shape,
    }


def hook_trace(model: torch.nn.Module, value: torch.Tensor) -> tuple[list[dict], list[int]]:
    calls = []
    handles = []
    for path, module in model.named_modules():
        if path and not tuple(module.children()):
            def record(mod, inputs, output, name=path):
                calls.append({
                    "target": name,
                    "tensor_shape": list(output.shape) if isinstance(output, torch.Tensor) else None,
                })
            handles.append(module.register_forward_hook(record))
    try:
        with torch.inference_mode():
            result = model(value)
    finally:
        for handle in handles:
            handle.remove()
    assert isinstance(result, torch.Tensor)
    return calls, list(result.shape)


def audit(output: Path, shapes=((1, 3, 224, 224), (2, 3, 224, 224))):
    assert importlib.metadata.version("timm") == TIMM_VERSION, "Unexpected timm version"
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    source_sha256 = snapshot_sources(output / "source")
    assert source_sha256 == EXPECTED_SOURCE_SHA256, "Audited timm source bytes changed"
    all_cases = []
    for fused in (True, False):
        set_fused_attn(fused)
        torch.manual_seed(2026)
        model = timm.create_model(MODEL_NAME, pretrained=False).eval()
        assert len(model.blocks) == 12
        assert model.embed_dim == 192
        assert model.patch_embed.proj.kernel_size == (16, 16)
        assert model.patch_embed.proj.out_channels == 192
        assert all(block.attn.num_heads == 3 for block in model.blocks)
        assert all(block.attn.fused_attn is fused for block in model.blocks)
        assert model.num_classes == 1000
        reference = symbolic_trace(model)
        for shape in shapes:
            value = torch.zeros(*shape)
            ShapeProp(reference).propagate(value)
            nodes = [fx_record(node) for node in reference.graph.nodes]
            edges = [
                {"source": dep.name, "target": node.name, "slot": index}
                for node in reference.graph.nodes
                for index, dep in enumerate(node.all_input_nodes)
            ]
            hooks, result_shape = hook_trace(model, value)
            fx_calls = [
                {"target": node["target"], "tensor_shape": node["tensor_shape"]}
                for node in nodes if node["op"] == "call_module"
            ]
            # timm.LayerNorm is an inlined Python module for torch.fx:
            # FX records its functional layer_norm, hooks see the leaf module.
            # Check all FX leaf calls against runtime, and *only* allow the
            # explicit 12 norm1 + 12 norm2 + final norm runtime-only events.
            fx_targets = {item["target"] for item in fx_calls}
            inlined_hooks = [h for h in hooks if h["target"] not in fx_targets]
            expected_inlined = {f"blocks.{i}.{name}" for i in range(12)
                                for name in ("norm1", "norm2")} | {"norm"}
            inlined_targets = [h["target"] for h in inlined_hooks]
            assert len(inlined_hooks) == 25 and set(inlined_targets) == expected_inlined, (
                "Unexpected non-FX leaf invocations: " + repr(inlined_targets))
            assert all(
                type(model.get_submodule(h["target"])).__name__ == "LayerNorm"
                and type(model.get_submodule(h["target"])).__module__.startswith("timm.layers")
                for h in inlined_hooks
            ), "Non-LayerNorm runtime call missing from FX"
            matched_hooks = [h for h in hooks if h["target"] in fx_targets]
            if matched_hooks != fx_calls:
                first = next(
                    (idx for idx, (a, b) in enumerate(zip(matched_hooks, fx_calls)) if a != b),
                    min(len(matched_hooks), len(fx_calls)),
                )
                discrepancy = {
                    "first_mismatch_index": first,
                    "hook_count": len(hooks),
                    "filtered_hook_count": len(matched_hooks),
                    "fx_call_count": len(fx_calls),
                    "hooks_near": matched_hooks[max(0, first - 3):first + 5],
                    "fx_near": fx_calls[max(0, first - 3):first + 5],
                }
                (output / "reference-discrepancy.json").write_text(json.dumps(discrepancy, indent=2))
                raise AssertionError(json.dumps(discrepancy, indent=2))
            assert result_shape == [shape[0], 1000], result_shape
            counts = (len(nodes), len(edges),
                      sum(node["tensor_shape"] is not None for node in nodes),
                      len(hooks), len(fx_calls), len(inlined_hooks))
            assert counts == EXPECTED_COUNTS[fused], ("Reference topology drift", fused, counts)
            assert len({node["name"] for node in nodes}) == len(nodes)
            assert len({(edge["source"], edge["target"], edge["slot"]) for edge in edges}) == len(edges)
            label = ("fused" if fused else "unfused") + "-" + "x".join(map(str, shape))
            folder = output / label
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "fx.json").write_text(json.dumps({"nodes": nodes, "edges": edges}, indent=2))
            (folder / "hooks.json").write_text(json.dumps(hooks, indent=2))
            row = {
                "configuration": label,
                "fused_attention": fused,
                "shape": list(shape),
                "fx_nodes": len(nodes),
                "fx_edges": len(edges),
                "fx_tensor_shapes": sum(node["tensor_shape"] is not None for node in nodes),
                "leaf_runtime_invocations": len(hooks),
                "fx_call_module_invocations": len(fx_calls),
                "inlined_timm_layernorm_invocations": len(inlined_hooks),
                "output_shape": result_shape,
                "status": "reference_only_not_ae_verified",
            }
            (folder / "result.json").write_text(json.dumps(row, indent=2))
            all_cases.append(row)
    summary = {
        "status": "reference_preflight_only",
        "model": MODEL_NAME,
        "timm": timm.__version__,
        "torch": torch.__version__,
        "source_sha256": source_sha256,
        "cases": all_cases,
        "ae_static_equivalence_verified": False,
        "note": "Do not merge as a completed instance-forward gate until AE IR matches every reference case.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark-output/vit-tiny-v020"))
    args = parser.parse_args()
    audit(args.output)
