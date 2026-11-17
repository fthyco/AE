"""Bounded, non-executing TorchVision MobileNetV3-Small instance-forward expansion.

Only the original torchvision 0.25.0 source bytes, default factory, and
explicit RGB input geometry are supported. No imports of analyzed packages.
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass

from ..ir.canonical import stable_id
from ..linking.project import Unit, span
from .instance_forward import StaticPathUnsupported, _check, _method


_PINNED = {
    "torchvision/models/mobilenetv3.py": "5444909362734ab54f07ba64cd2d54e90854965d4d1ce6b2b337e055d0fcbb3e",
    "torchvision/ops/misc.py": "a2bb1f3af1d3b854761ea4bc35d0b9deb5e49771f0dc919b390643f081a1ab7e",
    "torchvision/models/_utils.py": "b00ff45460ea07fa08e38495259f5954dfd2050587eb4c9398d0f6945df5f7b8",
}
_ENTRYPOINTS = {
    "torchvision.models.mobilenetv3:mobilenet_v3_small",
    "mobilenetv3:mobilenet_v3_small",
}


@dataclass(frozen=True)
class _Val:
    ident: str
    shape: tuple[int, ...]


def _literal(node: ast.AST, env: dict[str, int] | None = None):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and env is not None and node.id in env:
        return env[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv):
        a, b = _literal(node.left, env), _literal(node.right, env)
        _check(type(a) is type(b) is int and b != 0, "Unknown configuration division")
        return a // b
    raise StaticPathUnsupported("Unproven MobileNet factory argument")


class _Original:
    def __init__(self, units: list[Unit], ir_nodes: list[dict]):
        self.units = {}
        for name, digest in _PINNED.items():
            found = [u for u in units if u.path == name or u.path.endswith("/" + name)]
            _check(len(found) == 1, "Missing or duplicated original source: " + name)
            unit = found[0]
            _check(hashlib.sha256(unit.source.raw).hexdigest() == digest,
                   "Audited torchvision source changed: " + name)
            self.units[name] = unit
        self.ir_site_ids = {}
        for item in ir_nodes:
            for ref in item.get("source_refs", []):
                if ref.get("precision") == "exact":
                    key = (ref["path"], ref["start_line"], ref["start_column"],
                           ref["end_line"], ref["end_column"])
                    self.ir_site_ids.setdefault(key, set()).add(item["id"])
        self.sites = {}
        self._prove_and_select_sites()
        self.config = self._factory_configs()

    def _class(self, module: str, name: str) -> ast.ClassDef:
        matches = [n for n in self.units[module].tree.body
                   if isinstance(n, ast.ClassDef) and n.name == name]
        _check(len(matches) == 1, "Unresolved original class: " + name)
        return matches[0]

    def _func(self, module: str, name: str) -> ast.FunctionDef:
        matches = [n for n in self.units[module].tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == name]
        _check(len(matches) == 1, "Unresolved original function: " + name)
        return matches[0]

    def _prove_and_select_sites(self):
        model = "torchvision/models/mobilenetv3.py"
        misc = "torchvision/ops/misc.py"
        definitions = {
            "input": (model, _method(self._class(model, "MobileNetV3"), "forward")),
            "output": (model, _method(self._class(model, "MobileNetV3"), "_forward_impl")),
            "flatten": (model, _method(self._class(model, "MobileNetV3"), "_forward_impl")),
            "residual": (model, _method(self._class(model, "InvertedResidual"), "forward")),
            "se_mul": (misc, _method(self._class(misc, "SqueezeExcitation"), "forward")),
            "se_ops": (misc, _method(self._class(misc, "SqueezeExcitation"), "_scale")),
            "se_ctor": (misc, _method(self._class(misc, "SqueezeExcitation"), "__init__")),
            "conv_ctor": (misc, _method(self._class(misc, "ConvNormActivation"), "__init__")),
            "classifier": (model, _method(self._class(model, "MobileNetV3"), "__init__")),
        }
        for key, (module, func) in definitions.items():
            self.sites[key] = (self.units[module], func)

        def require(key: str, predicate, root: ast.AST):
            found = sorted((n for n in ast.walk(root) if predicate(n)),
                           key=lambda n: (n.lineno, n.col_offset))
            _check(bool(found), "Source operation missing: " + key)
            unit = self.sites[key.split(":")[0]][0]
            self.sites[key] = unit, found[0]

        require("conv_ctor:conv", lambda n: isinstance(n, ast.Call) and
                isinstance(n.func, ast.Name) and n.func.id == "conv_layer",
                self.sites["conv_ctor"][1])
        require("conv_ctor:norm", lambda n: isinstance(n, ast.Call) and
                isinstance(n.func, ast.Name) and n.func.id == "norm_layer",
                self.sites["conv_ctor"][1])
        require("conv_ctor:act", lambda n: isinstance(n, ast.Call) and
                isinstance(n.func, ast.Name) and n.func.id == "activation_layer",
                self.sites["conv_ctor"][1])
        for suffix in ("avgpool", "fc1", "activation", "fc2", "scale_activation"):
            require("se_ops:" + suffix,
                    lambda n, name=suffix: isinstance(n, ast.Call) and
                    isinstance(n.func, ast.Attribute) and
                    isinstance(n.func.value, ast.Name) and
                    n.func.value.id == "self" and n.func.attr == name,
                    self.sites["se_ops"][1])
        require("se_mul:mul", lambda n: isinstance(n, ast.BinOp) and
                isinstance(n.op, ast.Mult), self.sites["se_mul"][1])
        require("residual:add", lambda n: isinstance(n, ast.AugAssign) and
                isinstance(n.op, ast.Add), self.sites["residual"][1])
        require("flatten:flatten", lambda n: isinstance(n, ast.Call) and
                isinstance(n.func, ast.Attribute) and n.func.attr == "flatten",
                self.sites["flatten"][1])
        # The Squeeze-Excitation and conditional residual paths must still be
        # present in original forward AST; file hashes alone are not enough to
        # explain which source sites justify the branch semantics.
        residual = self.sites["residual"][1]
        guards = [n for n in residual.body if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Attribute)
                  and n.test.attr == "use_res_connect"]
        _check(len(guards) == 1, "Residual guard is not source-proven")
        _check(len([n for n in ast.walk(residual) if isinstance(n, ast.AugAssign)
                    and isinstance(n.op, ast.Add)]) == 1,
               "Unexpected residual additions")

    def _factory_configs(self) -> list[tuple[int, int, int, int, bool, str, int, int]]:
        src = "torchvision/models/mobilenetv3.py"
        factory = self._func(src, "mobilenet_v3_small")
        conf = self._func(src, "_mobilenet_v3_conf")
        defaults = dict(zip([a.arg for a in conf.args.args][-len(conf.args.defaults):],
                            conf.args.defaults))
        for name, value in (("width_mult", 1.0),
                            ("reduced_tail", False), ("dilated", False)):
            _check(name in defaults and ast.literal_eval(defaults[name]) == value,
                   "Nondefault mobile factory configuration: " + name)
        _check(any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
                   and x.func.id == "_mobilenet_v3_conf" and x.args
                   and isinstance(x.args[0], ast.Constant)
                   and x.args[0].value == "mobilenet_v3_small"
                   for x in ast.walk(factory)),
               "Default small factory does not construct the selected architecture")
        conditions = [x for x in ast.walk(conf) if isinstance(x, ast.If)
                      and isinstance(x.test, ast.Compare)
                      and isinstance(x.test.left, ast.Name)
                      and x.test.left.id == "arch"
                      and x.test.comparators
                      and isinstance(x.test.comparators[0], ast.Constant)
                      and x.test.comparators[0].value == "mobilenet_v3_small"]
        _check(len(conditions) == 1, "Default Small configuration is not unique")
        assignments = [s.value for s in conditions[0].body
                       if isinstance(s, ast.Assign) and any(
                           isinstance(t, ast.Name) and t.id == "inverted_residual_setting"
                           for t in s.targets)]
        _check(len(assignments) == 1 and isinstance(assignments[0], ast.List),
               "Static eleven-block blueprint missing")
        result = []
        for call in assignments[0].elts:
            _check(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                   and call.func.id == "bneck_conf" and len(call.args) == 8
                   and not call.keywords, "Unproven InvertedResidual config")
            row = tuple(_literal(n, {"reduce_divider": 1, "dilation": 1})
                        for n in call.args)
            _check(all(type(v) is int and v > 0 for v in (row[0],row[1],row[2],row[3],row[6],row[7])),
                   "Invalid channel, kernel, stride or dilation")
            _check(type(row[4]) is bool and row[5] in ("RE", "HS")
                   and row[6] in (1, 2) and row[7] == 1,
                   "Nondefault Small residual path")
            result.append(row)
        _check(len(result) == 11, "MobileNetV3 Small is not 11 blocks")
        _check(result[0][0] == 16 and result[-1][3] == 96,
               "Factory stem/last stage channel mismatch")
        return result

    def ref(self, key: str) -> tuple[dict, list[str]]:
        unit, node = self.sites[key]
        source_ref = span(unit, node)
        k = (source_ref["path"], source_ref["start_line"], source_ref["start_column"],
             source_ref["end_line"], source_ref["end_column"])
        original_ids = sorted(self.ir_site_ids.get(k, []))
        if not original_ids:
            original_ids = [unit.source_id]
        return source_ref, original_ids


class _Expand:
    def __init__(self, original: _Original, shape: tuple[int, ...]):
        self.original = original
        self.shape = shape
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self.ordinals: dict[tuple, int] = {}

    def emit(self, op: str, target: str, parents: list[_Val],
             shape: tuple[int, ...], site: str) -> _Val:
        source_ref, ids = self.original.ref(site)
        fingerprint = (source_ref["path"], source_ref["start_byte"],
                       source_ref["end_byte"], op, target)
        ordinal = self.ordinals.get(fingerprint, 0)
        self.ordinals[fingerprint] = ordinal + 1
        uid = stable_id("ae-mnv3-v022", *fingerprint, ordinal)
        self.nodes.append({
            "id": uid, "op": op, "target": target,
            "call_ordinal": ordinal, "module_instance": target if op == "call_module" else None,
            "source_refs": [source_ref], "source_ir_ids": ids,
            "shape": list(shape), "value_kind": "tensor",
            "input_ids": [v.ident for v in parents],
        })
        for slot, p in enumerate(parents):
            self.edges.append({
                "id": stable_id("ae-mnv3-edge-v022", p.ident, uid, slot),
                "source": p.ident, "target": uid, "input_slot": slot,
                "relation": "STATIC_INSTANCE_TENSOR_DEPENDENCY",
            })
        return _Val(uid, shape)

    def mod(self, target: str, parent: _Val, channels: int,
            height: int, width: int, source: str) -> _Val:
        return self.emit("call_module", target, [parent],
                         (parent.shape[0], channels, height, width), source)

    def conv_seq(self, prefix: str, x: _Val, channels: int, stride: int,
                 activation: bool) -> _Val:
        b, _, h, w = x.shape
        _check(stride in (1, 2), "Unsupported convolution stride")
        h, w = ((h + stride - 1) // stride, (w + stride - 1) // stride)
        for sub, site in ((0, "conv_ctor:conv"), (1, "conv_ctor:norm")):
            x = self.mod(prefix + f".{sub}", x, channels, h, w, site)
        if activation:
            x = self.mod(prefix + ".2", x, channels, h, w, "conv_ctor:act")
        return x

    def se(self, prefix: str, x: _Val) -> _Val:
        b, c, h, w = x.shape
        squeezed = (c // 4 + 7) // 8 * 8
        squeezed = max(8, squeezed)
        avg = self.mod(prefix + ".avgpool", x, c, 1, 1, "se_ops:avgpool")
        fc1 = self.mod(prefix + ".fc1", avg, squeezed, 1, 1, "se_ops:fc1")
        a = self.mod(prefix + ".activation", fc1, squeezed, 1, 1, "se_ops:activation")
        fc2 = self.mod(prefix + ".fc2", a, c, 1, 1, "se_ops:fc2")
        scale = self.mod(prefix + ".scale_activation", fc2, c, 1, 1,
                         "se_ops:scale_activation")
        return self.emit("call_function", "mul", [scale, x], (b, c, h, w),
                         "se_mul:mul")

    def run(self) -> dict:
        _check(len(self.shape) == 4 and all(type(v) is int and v > 0 for v in self.shape),
               "Four concrete positive NCHW dimensions required")
        b, c, h, w = self.shape
        _check(c == 3 and (h, w) in {(64, 64), (128, 128), (224, 224)},
               "Audited RGB geometry is limited to 64, 128 or 224 square inputs")
        x = self.emit("placeholder", "x", [], (b, c, h, w), "input")
        x = self.conv_seq("features.0", x, 16, 2, True)
        for i, (inp, kernel, expanded, outc, use_se, activation, stride, dilation) in (
                enumerate(self.original.config, 1)):
            _check(x.shape[1] == inp and dilation == 1,
                   "Factory channel continuity or dilation is not proven")
            skip = x
            prefix = f"features.{i}.block"
            section = 0
            if expanded != inp:
                x = self.conv_seq(prefix + f".{section}", x, expanded, 1, True)
                section += 1
            x = self.conv_seq(prefix + f".{section}", x, expanded, stride, True)
            section += 1
            if use_se:
                x = self.se(prefix + f".{section}", x)
                section += 1
            x = self.conv_seq(prefix + f".{section}", x, outc, 1, False)
            if stride == 1 and inp == outc:
                _check(x.shape == skip.shape, "Residual dimensions do not match")
                x = self.emit("call_function", "add", [x, skip],
                              x.shape, "residual:add")
        x = self.conv_seq("features.12", x, 576, 1, True)
        x = self.mod("avgpool", x, 576, 1, 1, "classifier")
        x = self.emit("call_function", "flatten", [x], (b, 576), "flatten:flatten")
        for i, channels in enumerate((1024, 1024, 1024, 1000)):
            x = self.emit("call_module", f"classifier.{i}", [x],
                          (b, channels), "classifier")
        x = self.emit("output", "output", [x], x.shape, "output")
        _check((len(self.nodes), len(self.edges)) == (159, 173),
               "Audited operation/dependency counts drifted")
        return {
            "schema_version": "1.0", "status": "expanded_static",
            "scope": "pinned_torchvision_0_25_0_mobilenet_v3_small",
            "entrypoint_factory": "torchvision.models.mobilenetv3:mobilenet_v3_small",
            "source_path": self.original.units["torchvision/models/mobilenetv3.py"].path,
            "source_paths": {k: u.path for k, u in self.original.units.items()},
            "input_conditions": {
                "input_shape": list(self.shape), "weights": None,
                "width_mult": 1.0, "reduced_tail": False, "dilated": False,
                "eval": True,
            },
            "nodes": self.nodes, "edges": self.edges, "output_shape": list(x.shape),
            "unresolved": [],
        }


def expand_mobilenetv3_forward(units: list[Unit], entrypoint: str | None,
                               input_shape: tuple[int, ...] | None,
                               ir_nodes: list[dict]) -> dict | None:
    if entrypoint not in _ENTRYPOINTS or input_shape is None:
        return None
    try:
        return _Expand(_Original(units, ir_nodes), input_shape).run()
    except (StaticPathUnsupported, KeyError, IndexError, TypeError,
            ValueError, AttributeError, RecursionError, OverflowError,
            SyntaxError) as exc:
        return {
            "schema_version": "1.0", "status": "unsupported",
            "issues": [f"{type(exc).__name__}: {exc}"],
            "nodes": [], "edges": [],
        }
