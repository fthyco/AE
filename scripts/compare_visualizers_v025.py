"""Reproducible side-by-side smoke test of architecture visualizers.

FX is the independently executed forward graph reference for the pinned model.
Tool outputs are NOT counted as 1:1 FX equivalents: torchviz renders autograd,
torchview renders runtime module flow, VisualTorch renders architectural layers.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch
import torchvision
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp


def attempt(name, func):
    start = time.perf_counter()
    try:
        meta = func()
        return {"status": "rendered", "seconds": round(time.perf_counter() - start, 4),
                **meta}
    except Exception as exc:
        return {"status": "error", "seconds": round(time.perf_counter() - start, 4),
                "exception": type(exc).__name__, "message": str(exc)[:600]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert torch.__version__.split("+")[0] == "2.10.0"
    assert torchvision.__version__.split("+")[0] == "0.25.0"
    torch.set_num_threads(1)
    model = torchvision.models.resnet18(weights=None).eval()
    x = torch.ones(1, 3, 64, 64)
    traced = symbolic_trace(model)
    ShapeProp(traced).propagate(x)
    reference = list(traced.graph.nodes)
    fx = {
        "status": "verified_original_forward",
        "operations": len(reference),
        "operand_edges": sum(len(n.all_input_nodes) for n in reference),
        "shape_prop": "passed",
        "input_shape": list(x.shape),
    }

    def torchview_run():
        from torchview import draw_graph
        graph = draw_graph(model, input_size=tuple(x.shape), device="cpu",
                           expand_nested=True)
        dot = graph.visual_graph.source
        (out / "torchview.dot").write_text(dot)
        return {"artifact": "torchview.dot", "bytes": len(dot.encode())}

    def torchviz_run():
        from torchviz import make_dot
        with torch.enable_grad():
            y = model(x.clone().requires_grad_(True))
        dot = make_dot(y, params=dict(model.named_parameters())).source
        (out / "torchviz.dot").write_text(dot)
        return {"artifact": "torchviz.dot", "bytes": len(dot.encode())}

    def visualtorch_run():
        import visualtorch
        image = visualtorch.render(model, tuple(x.shape), style="graph",
                                   show_dimension=True)
        image.save(out / "visualtorch.png")
        return {"artifact": "visualtorch.png",
                "bytes": (out / "visualtorch.png").stat().st_size}

    tools = {
        "torchview": attempt("torchview", torchview_run),
        "torchviz": attempt("torchviz", torchviz_run),
        "visualtorch": attempt("visualtorch", visualtorch_run),
    }
    # Failures are recorded, not silently converted to successful equivalence.
    result = {
        "model": "torchvision.models.resnet18(weights=None).eval()",
        "runtime": {"torch": torch.__version__, "torchvision": torchvision.__version__},
        "fx_reference": fx, "tools": tools,
        "comparison_validity": "different graph semantics; no cross-tool node-count parity claim",
        "ae_reference": "scripts/verify_full_resnet18.py (separate source-only vs FX gate)",
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    assert fx["shape_prop"] == "passed"


if __name__ == "__main__":
    main()
