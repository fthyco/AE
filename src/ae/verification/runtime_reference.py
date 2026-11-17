"""Opt-in independent PyTorch FX + runtime-hooks reference for a trusted model.

IMPORTANT: Unlike AE static extraction, this command explicitly imports and
EXECUTES target Python. It is never invoked by AE source-only extraction or the
offline viewer. It records a reference, NOT validated AE-to-runtime equivalence.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def _source_import_path(source: Path):
    source = source.resolve()
    old = list(sys.path)
    try:
        sys.path.insert(0, str(source))
        yield
    finally:
        sys.path[:] = old


def _shapes(value: Any) -> Any:
    """Summarize without storing tensors, weights or customer samples."""
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (tuple, list)):
        return [_shapes(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _shapes(v) for k, v in sorted(value.items(),
                                                      key=lambda x: str(x[0]))}
    return {"type": type(value).__name__}


def _shape_meta(meta: Any) -> Any:
    if hasattr(meta, "shape") and hasattr(meta, "dtype"):
        return {"shape": list(meta.shape), "dtype": str(meta.dtype)}
    if isinstance(meta, (tuple, list)):
        return [_shape_meta(v) for v in meta]
    if isinstance(meta, dict):
        return {str(k): _shape_meta(v) for k, v in sorted(meta.items())}
    return None


def _parse_input(spec: tuple[int, ...], dtype: str, torch: Any) -> Any:
    if not spec or any(type(dim) is not int or dim < 1 for dim in spec):
        raise ValueError("Each input shape needs positive integer dimensions")
    n = 1
    for dim in spec:
        n *= dim
        if n > 8_000_000:
            raise ValueError("Refusing oversized synthetic input")
    dtypes = {
        "float32": torch.float32, "float64": torch.float64,
        "float16": torch.float16, "int64": torch.int64,
        "int32": torch.int32, "bool": torch.bool,
    }
    if dtype not in dtypes:
        raise ValueError("Unsupported input dtype")
    return torch.zeros(*spec, dtype=dtypes[dtype], device="cpu")


def runtime_audit(*, source_root: str | Path, factory: str,
                  input_shapes: list[tuple[int, ...]],
                  input_dtypes: list[str] | None = None,
                  output: str | Path, allow_execute: bool = False,
                  max_nodes: int = 20_000) -> dict:
    """Run only with deliberate consent, on trusted source and synthetic inputs.

    A factory is a zero-argument callable that returns an nn.Module.
    This runner has no sandbox and is inappropriate for untrusted code.
    """
    if not allow_execute:
        raise ValueError("Target code execution is disabled. Explicitly pass --allow-execute for TRUSTED code.")
    if not input_shapes or max_nodes < 1:
        raise ValueError("Require at least one shape and a positive node limit")
    dtypes = input_dtypes or ["float32"] * len(input_shapes)
    if len(dtypes) != len(input_shapes):
        raise ValueError("Supply one dtype per input tensor")
    source = Path(source_root).resolve()
    if not source.is_dir() or not factory or factory.count(":") != 1:
        raise ValueError("Pass an existing source directory and a module:factory")
    module_name, factory_name = factory.split(":")
    if not all(piece.isidentifier() for piece in module_name.split(".")) or not factory_name.isidentifier():
        raise ValueError("Factory must be a Python module:callable identifier")
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": "1.0", "reference_kind": "fx_and_runtime_hooks",
        "factory": factory, "input_shapes": [list(x) for x in input_shapes],
        "input_dtypes": dtypes, "target_source_executed": True,
        "input_data": "synthetic_zeros", "weights": "random_or_factory_initialized",
        "ae_static_equivalence": "not_verified",
        "dynamic_control_flow_completeness": "not_verified",
        "status": "unsupported",
    }

    try:
        # Lazy import: default AE source-only installation NEVER needs torch.
        import torch
        from torch.fx import symbolic_trace
        from torch.fx.passes.shape_prop import ShapeProp

        inputs = [_parse_input(s, dtype, torch) for s, dtype in zip(input_shapes, dtypes)]
        with _source_import_path(source):
            module = importlib.import_module(module_name)
            creator = getattr(module, factory_name)
            if not callable(creator):
                raise ValueError("Selected factory is not callable")
            model = creator()
        if not isinstance(model, torch.nn.Module):
            raise ValueError("Factory did not return torch.nn.Module")
        model.eval().to("cpu")
        result["parameter_count"] = sum(p.numel() for p in model.parameters())
        result["torch_version"] = torch.__version__
        result["named_modules"] = len(list(model.named_modules()))

        with torch.no_grad():
            traced = symbolic_trace(model)
        nodes = list(traced.graph.nodes)
        if len(nodes) > max_nodes:
            raise ValueError("FX graph exceeds configured safety limit")
        ShapeProp(traced).propagate(*inputs)

        fx_records = []
        for node in nodes:
            fx_records.append({
                "id": node.name, "op": node.op,
                "target": str(node.target),
                "operand_ids": [n.name for n in node.all_input_nodes],
                "shape": _shape_meta(node.meta.get("tensor_meta")),
            })
        hooks = []
        visits = []
        for name, submodule in model.named_modules():
            if not name or any(True for _ in submodule.children()):
                continue

            def observer(_mod, args, value, *, path=name):
                visits.append({"module": path,
                               "inputs": _shapes(args),
                               "output": _shapes(value)})
            hooks.append(submodule.register_forward_hook(observer))

        try:
            with torch.inference_mode():
                final = model(*inputs)
        finally:
            for hook in hooks:
                hook.remove()

        reference = {
            "schema_version": "1.0", "status": "traced",
            "inputs": [{"shape": list(s), "dtype": d}
                       for s, d in zip(input_shapes, dtypes)],
            "output": _shapes(final),
            "nodes": fx_records,
            "leaf_module_invocations": visits,
        }
        raw = (json.dumps(reference, sort_keys=True, indent=2,
                          ensure_ascii=True) + "\n").encode()
        (destination / "fx-runtime-reference.json").write_bytes(raw)
        result.update({
            "status": "traced", "fx_nodes": len(nodes),
            "fx_operand_edges": sum(len(node.all_input_nodes) for node in nodes),
            "shape_prop": "passed",
            "runtime_leaf_invocations": len(visits),
            "runtime_output": _shapes(final),
            "reference_sha256": hashlib.sha256(raw).hexdigest(),
            "reference_file": "fx-runtime-reference.json",
        })
    except Exception as exc:
        # Bounded diagnostic, never report a made-up graph.
        result["status"] = "unsupported"
        result["reason"] = type(exc).__name__ + ": " + str(exc)[:400]
        (destination / "fx-runtime-reference.json").unlink(missing_ok=True)
    (destination / "runtime-audit-summary.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return result
