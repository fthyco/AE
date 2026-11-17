"""Original-only TorchVision 0.25.0 MobileNetV3-Small FX/hook ground truth.

This module MUST NOT be imported by the SAE extractor.
"""
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


SOURCES = {
    "torchvision/models/mobilenetv3.py": (
        "torchvision.models.mobilenetv3",
        "5444909362734ab54f07ba64cd2d54e90854965d4d1ce6b2b337e055d0fcbb3e"),
    "torchvision/ops/misc.py": (
        "torchvision.ops.misc",
        "a2bb1f3af1d3b854761ea4bc35d0b9deb5e49771f0dc919b390643f081a1ab7e"),
    "torchvision/models/_utils.py": (
        "torchvision.models._utils",
        "b00ff45460ea07fa08e38495259f5954dfd2050587eb4c9398d0f6945df5f7b8"),
}
SHAPES = ((1, 3, 64, 64), (1, 3, 128, 128),
          (1, 3, 224, 224), (2, 3, 224, 224))


def snapshot_sources(destination: Path) -> dict[str, str]:
    sha256 = {}
    for rel, (module, expected) in SOURCES.items():
        original = Path(importlib.import_module(module).__file__).read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        assert digest == expected, f"Unpinned TorchVision source drift: {rel} {digest}"
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original)
        assert target.read_bytes() == original
        sha256[rel] = digest
    return sha256


def record_node(node) -> dict:
    meta = node.meta.get("tensor_meta")
    return {
        "name": node.name,
        "op": node.op,
        "target": (
            str(node.target) if node.op in ("placeholder", "call_module", "output")
            else getattr(node.target, "__name__", str(node.target))
        ),
        "inputs": [n.name for n in node.all_input_nodes],
        "shape": list(meta.shape) if hasattr(meta, "shape") else None,
    }


def runtime_hooks(model: torch.nn.Module, value: torch.Tensor):
    sequence = []
    handles = []
    for path, mod in model.named_modules():
        if path and not tuple(mod.children()):
            def record(_module, _inputs, output, name=path):
                assert isinstance(output, torch.Tensor), (name, type(output))
                sequence.append({"target": name, "shape": list(output.shape)})
            handles.append(mod.register_forward_hook(record))
    try:
        with torch.inference_mode():
            output = model(value)
    finally:
        for hook in handles:
            hook.remove()
    return sequence, list(output.shape)


def audit(output: Path) -> dict:
    assert torch.__version__.split("+")[0] == "2.10.0", torch.__version__
    assert torchvision.__version__.split("+")[0] == "0.25.0", torchvision.__version__
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    output.mkdir(parents=True, exist_ok=True)
    hashes = snapshot_sources(output / "source")
    cases = []
    model = torchvision.models.mobilenet_v3_small(weights=None).eval()
    assert len(model.features) == 13
    assert sum(isinstance(m, torchvision.models.mobilenetv3.InvertedResidual)
               for m in model.features) == 11
    assert sum(isinstance(m, torchvision.ops.SqueezeExcitation)
               for m in model.modules()) == 9
    assert sum(b.use_res_connect for b in model.features
               if isinstance(b, torchvision.models.mobilenetv3.InvertedResidual)) == 6
    assert sum(p.numel() for p in model.parameters()) == 2542856
    traced = symbolic_trace(model)
    for shape in SHAPES:
        value = torch.zeros(*shape)
        ShapeProp(traced).propagate(value)
        nodes = [record_node(n) for n in traced.graph.nodes]
        edges = [
            {"source": dep.name, "target": n.name, "slot": slot}
            for n in traced.graph.nodes
            for slot, dep in enumerate(n.all_input_nodes)
        ]
        hooks, output_shape = runtime_hooks(model, value)
        expected_hooks = [{"target": n["target"], "shape": n["shape"]}
                          for n in nodes if n["op"] == "call_module"]
        assert hooks == expected_hooks, "FX leaf module sequence != actual runtime"
        assert output_shape == [shape[0], 1000]
        assert (len(nodes), len(edges), len(hooks)) == (159, 173, 141)
        assert sum(n["target"] == "mul" for n in nodes) == 9
        assert sum(n["target"] == "add" for n in nodes) == 6
        assert sum(n["target"] == "flatten" for n in nodes) == 1
        label = "x".join(map(str, shape))
        folder = output / label
        folder.mkdir(exist_ok=True)
        (folder / "fx.json").write_text(
            json.dumps({"nodes": nodes, "edges": edges}, indent=2))
        (folder / "hooks.json").write_text(json.dumps(hooks, indent=2))
        cases.append({
            "configuration": label, "shape": list(shape),
            "fx_nodes": len(nodes), "fx_ordered_edges": len(edges),
            "fx_tensor_shapes": sum(n["shape"] is not None for n in nodes),
            "fx_leaf_module_calls": len(expected_hooks),
            "runtime_leaf_module_calls": len(hooks),
            "squeeze_excitation_multiplications": 9,
            "residual_additions": 6,
            "output_shape": output_shape,
            "status": "independent_original_reference",
        })
    summary = {
        "status": "reference_only_not_AE_verified",
        "model": "torchvision.models.mobilenet_v3_small",
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "source_sha256": hashes, "cases": cases,
    }
    (output / "reference-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=Path("benchmark-output/mobilenetv3-small-v022"))
    audit(ap.parse_args().output)
