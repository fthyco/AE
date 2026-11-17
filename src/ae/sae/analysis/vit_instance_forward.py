"""Bounded source-only timm 1.0.22 ViT-Tiny instance-forward expansion.

This is a source-audited specialization, NOT an arbitrary Python interpreter.
No target model imports, torch operations, or reference FX traces occur here.
Unsupported or modified sources fail closed with zero speculative nodes.
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from typing import Any

from ..ir.canonical import stable_id
from ..linking.project import Unit, span
from .instance_forward import StaticPathUnsupported, _check, _method

_PINNED = {
    "timm/models/vision_transformer.py": "fbbfd91ae654ec3d32ccea05002177efee21281c6db73493c4ea79acc3d7b212",
    "timm/layers/attention.py": "5fcb14cb16e7382a4845d7aa97328253745827a6f1399e60632ad527fd8c1e07",
    "timm/layers/patch_embed.py": "7b7e670f8e6a2828752a65297025fb903fc2fbe0298617a3cd29d804be1455aa",
    "timm/layers/mlp.py": "c024128a1c7e2a55ebc5e33a7c94c585ae1e960cd032b39b08445ea30f05c5a5",
    "timm/layers/norm.py": "afe3abf01a4379065c1c11bf6b6f947a1a743aff7f5e51849b353697b0db557d",
}
ENTRYPOINTS = {
    "timm.models.vision_transformer:vit_tiny_patch16_224",
    "vision_transformer:vit_tiny_patch16_224",
}


@dataclass(frozen=True)
class _Value:
    id: str
    shape: tuple[int, ...] | None


class _OriginalSources:
    def __init__(self, units: list[Unit], graph_nodes: list[dict]):
        self.units: dict[str, Unit] = {}
        for name, expected_digest in _PINNED.items():
            found = [u for u in units if u.path == name or u.path.endswith("/" + name)]
            _check(len(found) == 1, "Unresolved/duplicated timm source: " + name)
            source = found[0]
            _check(hashlib.sha256(source.source.raw).hexdigest() == expected_digest,
                   "Audited timm source changed: " + name)
            self.units[name] = source
        self.sites: dict[str, tuple[Unit, ast.AST]] = {}
        self.ir_ids: dict[tuple, set[str]] = {}
        for item in graph_nodes:
            for ref in item.get("source_refs", []):
                if ref.get("precision") == "exact":
                    key = (ref["path"], ref["start_line"], ref["start_column"],
                           ref["end_line"], ref["end_column"])
                    self.ir_ids.setdefault(key, set()).add(item["id"])
        self._register_methods()
        self._check_factory()

    def _method(self, source: str, cls: str, name: str) -> tuple[Unit, ast.FunctionDef]:
        unit = self.units[source]
        classes = [x for x in unit.tree.body if isinstance(x, ast.ClassDef) and x.name == cls]
        _check(len(classes) == 1, "Missing timm class " + cls)
        return unit, _method(classes[0], name)

    def _register_methods(self):
        methods = {
            "vit": ("timm/models/vision_transformer.py", "VisionTransformer", "forward"),
            "features": ("timm/models/vision_transformer.py", "VisionTransformer", "forward_features"),
            "pos": ("timm/models/vision_transformer.py", "VisionTransformer", "_pos_embed"),
            "head": ("timm/models/vision_transformer.py", "VisionTransformer", "forward_head"),
            "pool": ("timm/models/vision_transformer.py", "VisionTransformer", "pool"),
            "global_pool": ("timm/models/vision_transformer.py", "VisionTransformer", "__init__"),
            "block": ("timm/models/vision_transformer.py", "Block", "forward"),
            "attention": ("timm/layers/attention.py", "Attention", "forward"),
            "patch": ("timm/layers/patch_embed.py", "PatchEmbed", "forward"),
            "mlp": ("timm/layers/mlp.py", "Mlp", "forward"),
            "norm": ("timm/layers/norm.py", "LayerNorm", "forward"),
        }
        self.sites = {key: self._method(*spec) for key, spec in methods.items()}
        # Select the closest source-backed AST operation, not a fabricated line.
        for scope, (unit, function) in list(self.sites.items()):
            options = sorted(
                (x for x in ast.walk(function) if isinstance(
                    x, (ast.Call, ast.Attribute, ast.Subscript, ast.BinOp, ast.Compare))),
                key=lambda x: (getattr(x, "lineno", 0), getattr(x, "col_offset", 0)))
            for action in ("getattr", "getitem", "add", "eq", "reshape", "transpose",
                           "flatten", "permute", "unbind", "matmul", "mul", "softmax",
                           "cat", "expand", "layer_norm", "scaled_dot_product_attention",
                           "maybe_add_mask", "_assert", "qkv", "q_norm", "k_norm",
                           "attn_drop", "proj", "norm1", "norm2", "norm", "fc1", "fc2",
                           "act", "drop1", "drop2", "ls1", "ls2", "drop_path1",
                           "drop_path2", "patch_drop", "norm_pre", "pos_drop", "head_drop",
                           "head", "fc_norm"):
                matched = []
                for n in options:
                    if action == "add" and isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
                        matched.append(n)
                    elif action == "eq" and isinstance(n, ast.Compare) and any(
                            isinstance(op, ast.Eq) for op in n.ops):
                        matched.append(n)
                    elif action == "getitem" and isinstance(n, ast.Subscript):
                        matched.append(n)
                    elif action == "getattr" and isinstance(n, ast.Attribute) and n.attr == "shape":
                        matched.append(n)
                    elif isinstance(n, ast.Call):
                        fun = n.func
                        if ((isinstance(fun, ast.Attribute) and fun.attr == action) or
                            (isinstance(fun, ast.Name) and fun.id == action)):
                            matched.append(n)
                if matched:
                    self.sites[scope + ":" + action] = (unit, matched[0])

    def _check_factory(self):
        unit = self.units["timm/models/vision_transformer.py"]
        functions = [x for x in unit.tree.body if isinstance(x, ast.FunctionDef)
                     and x.name == "vit_tiny_patch16_224"]
        _check(len(functions) == 1, "Missing ViT-Tiny factory")
        factory = functions[0]
        definitions = [x.value for x in ast.walk(factory) if isinstance(x, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "model_args"
                               for t in x.targets)]
        _check(len(definitions) == 1 and isinstance(definitions[0], ast.Call)
               and isinstance(definitions[0].func, ast.Name)
               and definitions[0].func.id == "dict", "Factory is not a fixed dictionary")
        keywords = {x.arg: x.value for x in definitions[0].keywords}
        expected = {"patch_size": 16, "embed_dim": 192, "depth": 12, "num_heads": 3}
        _check(set(expected).issubset(keywords) and all(
            ast.literal_eval(keywords[k]) == v for k, v in expected.items()
        ), "Factory ViT-Tiny parameters drifted")
        self.sites["factory"] = (unit, factory)

    def source(self, scope: str, action: str | None = None) -> tuple[dict, list[str]]:
        site = self.sites.get(f"{scope}:{action}") if action else None
        if site is None:
            site = self.sites[scope]
        unit, syntax = site
        ref = span(unit, syntax)
        key = (ref["path"], ref["start_line"], ref["start_column"],
               ref["end_line"], ref["end_column"])
        ids = sorted(self.ir_ids.get(key, []))
        if not ids:
            ids = [stable_id(unit.path, "source-file")]
        return ref, ids


class _ViT:
    def __init__(self, sources: _OriginalSources, shape: tuple[int, ...], fused: bool):
        self.src = sources
        self.input = shape
        self.fused = fused
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self.ordinals: dict[tuple, int] = {}

    def emit(self, op: str, target: str, inputs: list[_Value],
             shape: tuple[int, ...] | None, scope: str,
             action: str | None = None) -> _Value:
        ref, src_ids = self.src.source(scope, action or target)
        key = (ref["path"], op, target, ref["start_byte"], ref["end_byte"])
        ordinal = self.ordinals.get(key, 0)
        self.ordinals[key] = ordinal + 1
        uid = stable_id("vit-tiny-instance/v1", *key, ordinal)
        self.nodes.append({
            "id": uid, "op": op, "target": target, "call_ordinal": ordinal,
            "module_instance": target if op == "call_module" else None,
            "source_refs": [ref], "source_ir_ids": src_ids,
            "shape": list(shape) if shape is not None else None,
            "value_kind": "metadata" if shape is None else "tensor",
            "input_ids": [v.id for v in inputs],
        })
        for slot, val in enumerate(inputs):
            kind = ("STATIC_INSTANCE_SHAPE_METADATA_DEPENDENCY"
                    if shape is None or val.shape is None
                    else "STATIC_INSTANCE_TENSOR_DEPENDENCY")
            self.edges.append({
                "id": stable_id("vit-tiny-edge/v1", val.id, uid, slot),
                "source": val.id, "target": uid, "input_slot": slot, "relation": kind,
            })
        return _Value(uid, shape)

    def mod(self, target: str, x: _Value, shape: tuple[int, ...], scope: str,
            action: str | None = None) -> _Value:
        return self.emit("call_module", target, [x], shape, scope, action or target.split(".")[-1])

    def attr(self, target: str, shape: tuple[int, ...], scope: str,
             action: str | None = None) -> _Value:
        return self.emit("get_attr", target, [], shape, scope, action)

    def fn(self, target: str, inputs: list[_Value], shape: tuple[int, ...] | None,
           scope: str, action: str | None = None) -> _Value:
        return self.emit("call_function", target, inputs, shape, scope, action)

    def method(self, target: str, inputs: list[_Value], shape: tuple[int, ...] | None,
               scope: str, action: str | None = None) -> _Value:
        return self.emit("call_method", target, inputs, shape, scope, action)

    def _layer_norm(self, x: _Value, prefix: str, scope: str = "norm") -> _Value:
        weight = self.attr(prefix + ".weight", (192,), scope, "layer_norm")
        bias = self.attr(prefix + ".bias", (192,), scope, "layer_norm")
        return self.fn("layer_norm", [x, weight, bias], x.shape, "norm", "layer_norm")

    def run(self) -> dict:
        shape = self.input
        _check(len(shape) == 4 and all(type(v) is int and v > 0 for v in shape),
               "Four positive NCHW axes required")
        batch, channels, height, width = shape
        _check(channels == 3 and height == width == 224,
               "Unmodified ViT-Tiny uses fixed RGB 224x224 image geometry")
        _check(self.fused in (True, False), "Unresolved attention branch")
        pixels = 196
        tokens = 197
        dim = 192
        hidden = 768
        heads = 3
        head_dim = 64
        nlc = (batch, tokens, dim)
        q_shape = (batch, heads, tokens, head_dim)

        x = self.emit("placeholder", "x", [], shape, "vit")
        mask = self.emit("placeholder", "attn_mask", [], None, "vit")
        shape_of_x = self.fn("getattr", [x], None, "patch", "getattr")
        indices = [self.fn("getitem", [shape_of_x], None, "patch", "getitem") for _ in range(4)]
        for size in indices[2:]:
            eq = self.fn("eq", [size], None, "patch", "eq")
            self.fn("_assert", [eq], None, "patch", "_assert")
        x = self.mod("patch_embed.proj", x, (batch, dim, 14, 14), "patch", "proj")
        x = self.method("flatten", [x], (batch, dim, pixels), "patch")
        x = self.method("transpose", [x], (batch, pixels, dim), "patch")
        x = self.mod("patch_embed.norm", x, (batch, pixels, dim), "patch", "norm")
        pos = self.attr("pos_embed", (1, tokens, dim), "pos")
        cls = self.attr("cls_token", (1, 1, dim), "pos")
        pshape = self.fn("getattr", [x], None, "pos", "getattr")
        pbatch = self.fn("getitem", [pshape], None, "pos", "getitem")
        cls = self.method("expand", [cls, pbatch], (batch, 1, dim), "pos")
        x = self.fn("cat", [cls, x], nlc, "pos")
        x = self.fn("add", [x, pos], nlc, "pos")
        x = self.mod("pos_drop", x, nlc, "pos")
        x = self.mod("patch_drop", x, nlc, "features")
        x = self.mod("norm_pre", x, nlc, "features")

        for i in range(12):
            base = f"blocks.{i}"
            skip = x
            ln = self._layer_norm(x, base + ".norm1", "block")
            attn_shape = self.fn("getattr", [ln], None, "attention", "getattr")
            b, n, c = [self.fn("getitem", [attn_shape], None, "attention", "getitem")
                       for _ in range(3)]
            y = self.mod(base + ".attn.qkv", ln, (batch, tokens, dim * 3), "attention", "qkv")
            y = self.method("reshape", [y, b, n],
                            (batch, tokens, 3, heads, head_dim), "attention")
            y = self.method("permute", [y], (3, batch, heads, tokens, head_dim), "attention")
            y = self.method("unbind", [y], None, "attention")
            q, k, v = [self.fn("getitem", [y], q_shape, "attention", "getitem")
                       for _ in range(3)]
            q = self.mod(base + ".attn.q_norm", q, q_shape, "attention")
            k = self.mod(base + ".attn.k_norm", k, q_shape, "attention")
            if self.fused:
                y = self.fn("scaled_dot_product_attention",
                            [q, k, v, mask], q_shape, "attention")
            else:
                q = self.fn("mul", [q], q_shape, "attention")
                kt = self.method("transpose", [k], (batch, heads, head_dim, tokens), "attention")
                attn = self.fn("matmul", [q, kt],
                               (batch, heads, tokens, tokens), "attention")
                attn = self.fn("maybe_add_mask", [attn, mask],
                               (batch, heads, tokens, tokens), "attention")
                attn = self.method("softmax", [attn],
                                   (batch, heads, tokens, tokens), "attention")
                attn = self.mod(base + ".attn.attn_drop", attn,
                                (batch, heads, tokens, tokens), "attention")
                y = self.fn("matmul", [attn, v], q_shape, "attention")
            y = self.method("transpose", [y], (batch, tokens, heads, head_dim), "attention")
            y = self.method("reshape", [y, b, n, c], nlc, "attention")
            for suffix in ("norm", "proj", "proj_drop"):
                y = self.mod(base + ".attn." + suffix, y, nlc, "attention", suffix)
            for suffix in ("ls1", "drop_path1"):
                y = self.mod(base + "." + suffix, y, nlc, "block", suffix)
            x = self.fn("add", [skip, y], nlc, "block")
            skip = x
            y = self._layer_norm(x, base + ".norm2", "block")
            for suffix in ("fc1", "act", "drop1", "norm", "fc2", "drop2"):
                hshape = (batch, tokens, hidden) if suffix in (
                    "fc1", "act", "drop1", "norm") else nlc
                y = self.mod(base + ".mlp." + suffix, y, hshape, "mlp", suffix)
            for suffix in ("ls2", "drop_path2"):
                y = self.mod(base + "." + suffix, y, nlc, "block", suffix)
            x = self.fn("add", [skip, y], nlc, "block")

        x = self._layer_norm(x, "norm", "features")
        x = self.fn("getitem", [x], (batch, dim), "pool")
        x = self.mod("fc_norm", x, (batch, dim), "head")
        x = self.mod("head_drop", x, (batch, dim), "head")
        x = self.mod("head", x, (batch, 1000), "head")
        x = self.emit("output", "output", [x], x.shape, "vit")
        expect = (477, 596) if self.fused else (549, 668)
        _check((len(self.nodes), len(self.edges)) == expect,
               "Source-backed operation/dependency inventory differs from audited topology")
        return {
            "schema_version": "1.0",
            "status": "expanded_static",
            "scope": "pinned_timm_1_0_22_vit_tiny_patch16_224",
            "entrypoint_factory": "timm.models.vision_transformer:vit_tiny_patch16_224",
            "source_path": self.src.units["timm/models/vision_transformer.py"].path,
            "source_paths": {name: u.path for name, u in self.src.units.items()},
            "input_conditions": {
                "input_shape": list(shape),
                "attention_mode": "fused" if self.fused else "unfused",
                "patch_size": 16, "embed_dim": 192, "depth": 12, "num_heads": 3,
                "pretrained": False,
            },
            "nodes": self.nodes, "edges": self.edges, "unresolved": [],
            "output_shape": list(x.shape),
        }


def expand_vit_forward(units: list[Unit], entrypoint: str | None,
                       input_shape: tuple[int, ...] | None,
                       ir_nodes: list[dict], attention_mode: str | None) -> dict | None:
    if entrypoint not in ENTRYPOINTS or input_shape is None:
        return None
    try:
        _check(attention_mode in ("fused", "unfused"),
               "ViT attention mode must be explicitly selected (fused or unfused)")
        original = _OriginalSources(units, ir_nodes)
        return _ViT(original, input_shape, attention_mode == "fused").run()
    except (StaticPathUnsupported, KeyError, IndexError, TypeError, ValueError,
            AttributeError, RecursionError, OverflowError, SyntaxError) as exc:
        return {
            "schema_version": "1.0", "status": "unsupported",
            "issues": [f"{type(exc).__name__}: {exc}"], "nodes": [], "edges": [],
        }
