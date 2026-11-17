"""Independent source/FX/hooks benchmark for unmodified open-source models."""
from __future__ import annotations

import collections
import hashlib
import importlib
import inspect
import json
import resource
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from ae.sae.api import extract, validate_bundle
from ae.presenter.api import present

CASES = {
    "resnet18": ("torchvision", "resnet18", 64),
    "resnet50": ("torchvision", "resnet50", 64),
    "densenet121": ("torchvision", "densenet121", 64),
    "vgg16": ("torchvision", "vgg16", 64),
    "mobilenet_v3_small": ("torchvision", "mobilenet_v3_small", 64),
    "efficientnet_b0": ("torchvision", "efficientnet_b0", 64),
    "unet_resnet34": ("segmentation_models_pytorch", "Unet", 64),
    "vit_tiny_patch16_224": ("timm", "vit_tiny_patch16_224", 224),
}


def shape(value):
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        return [shape(x) for x in value]
    if isinstance(value, dict):
        return {str(k): shape(v) for k, v in value.items()}
    return type(value).__name__


def create_model(group, name):
    if group == "torchvision":
        from torchvision import models
        return getattr(models, name)(weights=None)
    if group == "timm":
        import timm
        return timm.create_model(name, pretrained=False)
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name="resnet34", encoder_weights=None,
                    in_channels=3, classes=1)


def source_snapshot(model, group, destination):
    base = Path(importlib.import_module(group).__file__).parent.resolve()
    copied = {}
    classes = {type(model)}
    for sub in model.modules():
        if type(sub).__module__.startswith(group + "."):
            classes.add(type(sub))
    for cls in sorted(classes, key=lambda c: (c.__module__, c.__name__)):
        path = inspect.getsourcefile(cls)
        if not path:
            continue
        original = Path(path).resolve()
        try:
            rel = original.relative_to(base)
        except ValueError:
            continue
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        data = original.read_bytes()
        target.write_bytes(data)
        assert target.read_bytes() == data
        copied[rel.as_posix()] = hashlib.sha256(data).hexdigest()
    model_file = Path(inspect.getsourcefile(type(model))).resolve().relative_to(base)
    entrypoint = model_file.with_suffix("").as_posix().replace("/", ".") + ":" + type(model).__name__
    return entrypoint, copied


def runtime_reference(model, size):
    visits = []
    hooks = []
    names = collections.defaultdict(list)
    for name, module in model.named_modules(remove_duplicate=False):
        names[id(module)].append(name)
    aliases = [values for values in names.values() if len(values) > 1]
    for name, module in model.named_modules():
        if name:
            hooks.append(module.register_forward_hook(
                lambda obj, args, out, name=name: visits.append(
                    {"module": name, "type": type(obj).__name__,
                     "input": shape(args), "output": shape(out)})))
    observations = {}
    start = time.perf_counter()
    try:
        for side in sorted({size, 96 if size == 64 else size}):
            visits.clear()
            with torch.inference_mode():
                output = model(torch.zeros(1, 3, side, side))
            observations[str(side)] = {
                "output_shape": shape(output),
                "invocation_count": len(visits),
                "modules_reached": len({v["module"] for v in visits}),
                "examples": [v for v in visits if v["type"] in
                             {"BasicBlock", "Bottleneck", "_DenseBlock", "_DenseLayer",
                              "InvertedResidual", "MBConv", "Block", "Attention",
                              "UnetDecoder", "UnetDecoderBlock"}][:60],
            }
    finally:
        for hook in hooks:
            hook.remove()
    return {
        "samples": observations,
        "runtime_seconds": round(time.perf_counter() - start, 4),
        "named_module_count": len(list(model.named_modules())),
        "direct_children": list(dict(model.named_children())),
        "module_types": dict(collections.Counter(type(m).__name__ for m in model.modules())),
        "shared_module_aliases": aliases,
    }


def fx_reference(model, size):
    start = time.perf_counter()
    try:
        traced = symbolic_trace(model)
        nodes = list(traced.graph.nodes)
        by_op = dict(collections.Counter(n.op for n in nodes))
        functions = dict(collections.Counter(
            getattr(n.target, "__name__", str(n.target))
            for n in nodes if n.op == "call_function"))
        record = {
            "status": "traced", "node_count": len(nodes),
            "op_counts": by_op, "function_counts": functions,
            "edges": sum(len(n.all_input_nodes) for n in nodes),
        }
        try:
            ShapeProp(traced).propagate(torch.zeros(1, 3, size, size))
            record["shape_prop"] = "passed"
        except Exception as exc:
            record["shape_prop"] = "failed: " + repr(exc)
    except Exception as exc:
        record = {"status": "unsupported", "reason": repr(exc)}
    record["seconds"] = round(time.perf_counter() - start, 4)
    return record


def main(name):
    group, factory, side = CASES[name]
    root = Path("benchmark-output") / name
    sources = root / "source"
    sources.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    model = create_model(group, factory)
    model.eval()
    init_sec = round(time.perf_counter() - started, 4)
    entry, hashes = source_snapshot(model, group, sources)
    before = time.perf_counter()
    extract(sources, root / "sae", entrypoint=entry)
    extract_sec = round(time.perf_counter() - before, 4)
    validated = validate_bundle(root / "sae")["valid"]
    present(root / "sae", root / "figure", view="flow", level="operation",
            format="svg", mode="exploration")
    ir = json.loads((root / "sae/architecture.ir.json").read_text())
    index = json.loads((root / "sae/model-index.json").read_text())
    health = json.loads((root / "figure/presentation-health.json").read_text())
    manifest = json.loads((root / "figure/presentation-manifest.json").read_text())
    runtime = runtime_reference(model, side)
    fx = fx_reference(model, side)
    components = {c["owner_expression"].removeprefix("self.")
                  for m in index["modules"] for c in m["components"]}
    direct_children = set(runtime["direct_children"])
    result = {
        "model": name, "package": group,
        "package_version": importlib.metadata.version(group.replace("_", "-")) if False else
                           getattr(importlib.import_module(group), "__version__", "unknown"),
        "torch": torch.__version__, "entrypoint": entry,
        "source_hashes": hashes, "source_unmodified": True,
        "random_initialization": True, "extract_sec": extract_sec,
        "init_sec": init_sec, "peak_process_rss_mb":
            round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        "validated": validated,
        "ir_nodes": len(ir["nodes"]), "ir_edges": len(ir["edges"]),
        "unresolved": len(ir["unresolved"]),
        "symbolic_shapes": sum(n["kind"] == "symbolic_shape_fact" for n in ir["nodes"]),
        "indexed_modules": len(index["modules"]),
        "indexed_forward_calls": sum(len(m["forward_calls"]) for m in index["modules"]),
        "indexed_components": len(components),
        "direct_children_covered": len(components & direct_children),
        "direct_children_total": len(direct_children),
        "missing_direct_children": sorted(direct_children - components),
        "flow_status": health["status"],
        "flow_issues": [i["code"] for i in health["issues"]],
        "visual_elements": len(manifest["visual_elements"]),
        "visual_links": len(manifest["visual_links"]),
        "runtime": runtime, "fx": fx,
        "full_runtime_dependency_equivalence": "not_verified",
        "shared_module_identity_equivalence": "not_verified",
        "source_hash_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
    }
    (root / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    brief = {k: result[k] for k in [
        "model", "package_version", "extract_sec", "peak_process_rss_mb",
        "ir_nodes", "indexed_modules", "direct_children_covered",
        "direct_children_total", "flow_status", "flow_issues",
        "visual_elements", "visual_links", "unresolved"]}
    brief["fx_status"] = fx["status"]
    brief["fx_nodes"] = fx.get("node_count")
    brief["fx_functions"] = {k: fx.get("function_counts", {}).get(k, 0)
                             for k in ["add", "cat"] }
    brief["output_shapes"] = {x: v["output_shape"]
                               for x, v in runtime["samples"].items()}
    print("BENCH_RESULT " + json.dumps(brief, sort_keys=True), flush=True)
    assert validated
    if fx.get("node_count", 0) >= 30 and len(manifest["visual_elements"]) < 15:
        assert health["status"] != "complete", "Incomplete graph reported complete"


if __name__ == "__main__":
    main(sys.argv[1])
