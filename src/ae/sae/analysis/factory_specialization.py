"""Static factory arguments and fixed-count module construction."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from ..ir.canonical import stable_id
from ..ir.models import Graph
from ..linking.project import Unit, derived_edge, span, syntax_ids

RULE = 'static-factory-construction/v1'
_MAX_STAGES = 32
_MAX_BLOCKS = 256


@dataclass(frozen=True)
class _Symbol:
    name: str
    ident: str
    kind: str


_MISSING = object()


def _simple_value(node: ast.AST, env: dict[str, Any], symbols: dict[str, _Symbol]) -> Any:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id, symbols.get(node.id, _MISSING))
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_simple_value(x, env, symbols) for x in node.elts]
        if any(v is _MISSING or type(v) is not int for v in values):
            return _MISSING
        return tuple(values)
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and type(node.slice.value) is int:
        values = _simple_value(node.value, env, symbols)
        index = node.slice.value
        if isinstance(values, tuple) and 0 <= index < len(values):
            return values[index]
    return _MISSING


def _bind(fn: ast.FunctionDef, call: ast.Call, env: dict[str, Any], symbols: dict[str, _Symbol], *, skip_self: bool = False) -> dict[str, Any] | None:
    arguments = [*fn.args.posonlyargs, *fn.args.args]
    if skip_self and arguments and arguments[0].arg == "self":
        arguments = arguments[1:]
    if any(isinstance(arg, ast.Starred) for arg in call.args):
        return None
    if len(call.args) > len(arguments) and fn.args.vararg is None:
        return None
    mapping: dict[str, Any] = {}
    defaults = fn.args.defaults
    for index, arg in enumerate(arguments):
        default_index = index - (len(arguments) - len(defaults))
        default = defaults[default_index] if default_index >= 0 else None
        mapping[arg.arg] = (_simple_value(default, env, symbols) if default is not None else _MISSING)
    for arg in fn.args.kwonlyargs:
        mapping[arg.arg] = _MISSING
    for index, value in enumerate(call.args[:len(arguments)]):
        mapping[arguments[index].arg] = _simple_value(value, env, symbols)
    for kw in call.keywords:
        if kw.arg is None:
            continue  # unknown **kwargs never overrides a proven explicit positional argument
        if kw.arg not in mapping:
            if fn.args.kwarg is None:
                return None
            continue
        if kw.arg in {a.arg for a in arguments[:len(call.args)]}:
            return None  # duplicate argument, not a valid instantiation
        mapping[kw.arg] = _simple_value(kw.value, env, symbols)
    return mapping


def _selected_constructor(fn: ast.FunctionDef) -> ast.Call | None:
    returns = [s for s in fn.body if isinstance(s, ast.Return)]
    if len(returns) != 1 or returns[0].value is None:
        return None
    returned = returns[0].value
    if isinstance(returned, ast.Call):
        return returned
    if not isinstance(returned, ast.Name):
        return None
    assignments = [s for s in ast.walk(fn) if isinstance(s, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
                   and any(isinstance(t, ast.Name) and t.id == returned.id
                           for t in (s.targets if isinstance(s, ast.Assign) else [s.target]))]
    if len(assignments) != 1 or assignments[0] not in fn.body:
        return None
    value = assignments[0].value
    return value if isinstance(value, ast.Call) else None


def _append_block(expr: ast.stmt, block_arg: str, collection: str) -> bool:
    if not isinstance(expr, ast.Expr) or not isinstance(expr.value, ast.Call):
        return False
    outer = expr.value
    if not isinstance(outer.func, ast.Attribute) or outer.func.attr != 'append':
        return False
    if not isinstance(outer.func.value, ast.Name) or outer.func.value.id != collection:
        return False
    if len(outer.args) != 1 or not isinstance(outer.args[0], ast.Call):
        return False
    inner = outer.args[0]
    return isinstance(inner.func, ast.Name) and inner.func.id == block_arg


def _verify_repeat_template(cls: ast.ClassDef) -> tuple[ast.FunctionDef, ast.stmt, ast.For] | None:
    methods = [m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name == '_make_layer']
    if len(methods) != 1:
        return None
    fn = methods[0]
    names = [a.arg for a in fn.args.args]
    if len(names) < 4 or names[:4] != ['self', 'block', 'planes', 'blocks']:
        return None
    first = [s for s in fn.body if _append_block(s, 'block', 'layers')]
    loops = [s for s in fn.body if isinstance(s, ast.For)]
    if len(first) != 1 or len(loops) != 1:
        return None
    loop = loops[0]
    if fn.body.index(first[0]) >= fn.body.index(loop):
        return None
    initial_layers = [s for s in fn.body if isinstance(s, ast.Assign)
                      and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)
                      and s.targets[0].id == 'layers' and isinstance(s.value, ast.List)
                      and not s.value.elts]
    if len(initial_layers) != 1 or fn.body.index(initial_layers[0]) >= fn.body.index(first[0]):
        return None
    all_appends = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                   and n.func.value.id == 'layers' and n.func.attr in {'append', 'extend', 'insert', 'pop', 'clear'}]
    if len(all_appends) != 2 or any(n.func.attr != 'append' for n in all_appends):
        return None
    if not (isinstance(loop.iter, ast.Call) and isinstance(loop.iter.func, ast.Name)
            and loop.iter.func.id == 'range' and len(loop.iter.args) == 2
            and isinstance(loop.iter.args[0], ast.Constant) and loop.iter.args[0].value == 1
            and isinstance(loop.iter.args[1], ast.Name) and loop.iter.args[1].id == 'blocks'):
        return None
    if len(loop.body) != 1 or not _append_block(loop.body[0], 'block', 'layers') or loop.orelse:
        return None
    returns = [s for s in fn.body if isinstance(s, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Call):
        return None
    out = returns[0].value
    if not (isinstance(out.func, ast.Attribute) and isinstance(out.func.value, ast.Name)
            and out.func.value.id in {'nn', 'torch'} and out.func.attr == 'Sequential'
            and len(out.args) == 1 and isinstance(out.args[0], ast.Starred)
            and isinstance(out.args[0].value, ast.Name) and out.args[0].value.id == 'layers'):
        return None
    return fn, first[0], loop


def _stage_calls(cls: ast.ClassDef, env: dict[str, Any], symbols: dict[str, _Symbol]):
    initializers = [f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == '__init__']
    if len(initializers) != 1:
        return None
    init = initializers[0]
    stages = []
    for stmt in init.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                and target.value.id == 'self' and target.attr.startswith('layer')):
            continue
        call = stmt.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name) and call.func.value.id == 'self'
                and call.func.attr == '_make_layer' and len(call.args) >= 3):
            return None
        block = _simple_value(call.args[0], env, symbols)
        count = _simple_value(call.args[2], env, symbols)
        if not isinstance(block, _Symbol) or block.kind != 'class_definition':
            return None
        if type(count) is not int or not (1 <= count <= _MAX_BLOCKS):
            return None
        stages.append((stmt, target.attr, block, count))
    return stages if 1 <= len(stages) <= _MAX_STAGES else None


def specialize_factory(graph: Graph, units: list[Unit], requested: str | None) -> tuple[str | None, dict | None, dict | None]:
    """Return (model class ID, specialization, failure) for a function entrypoint."""
    if not requested or ':' not in requested:
        return None, None, None
    module, symbol_name = requested.split(':', 1)
    matching = [(u, f) for u in units if u.module == module
                for f in u.tree.body if isinstance(f, ast.FunctionDef) and f.name == symbol_name]
    if not matching:
        return None, None, None
    if len(matching) != 1:
        return None, None, {'kind': 'factory_specialization', 'reason': 'Ambiguous factory definition',
                            'path': '', 'source_refs': []}
    unit, factory = matching[0]
    funcs = {f.name: f for f in unit.tree.body if isinstance(f, ast.FunctionDef)}
    classes = {c.name: c for c in unit.tree.body if isinstance(c, ast.ClassDef)}
    symbols = {name: _Symbol(name, stable_id(unit.path, 'semantic', 'class_definition', syntax_ids(unit)[cls], ''),
                             'class_definition') for name, cls in classes.items()}
    funcs_unique = len(funcs) == sum(isinstance(x, ast.FunctionDef) for x in unit.tree.body)
    if not funcs_unique:
        return None, None, {'kind': 'factory_specialization', 'path': unit.path,
                            'reason': 'Ambiguous local function definitions', 'source_refs': [span(unit, factory)]}
    current = factory
    env: dict[str, Any] = {}
    visited = set()
    chain = []
    source_factory_id = stable_id(unit.path, 'semantic', 'function_definition', syntax_ids(unit)[factory], '')
    target_class = None
    for _ in range(5):
        if current.name in visited:
            break
        visited.add(current.name)
        call = _selected_constructor(current)
        if call is None or not isinstance(call.func, ast.Name):
            break
        chain.append(call)
        name = call.func.id
        if name in classes:
            target_class = classes[name]
            init = next((m for m in target_class.body if isinstance(m, ast.FunctionDef) and m.name == '__init__'), None)
            if init is None:
                break
            bound = _bind(init, call, env, symbols, skip_self=True)
            if bound is None:
                break
            env = bound
            break
        fn = funcs.get(name)
        if fn is None:
            break
        bound = _bind(fn, call, env, symbols)
        if bound is None:
            break
        current, env = fn, bound
    if target_class is None:
        return None, None, {'kind': 'factory_specialization', 'path': unit.path,
                            'reason': 'No uniquely resolved local constructor chain',
                            'source_refs': [span(unit, factory)]}
    template = _verify_repeat_template(target_class)
    stages = _stage_calls(target_class, env, symbols) if template else None
    if not template or not stages or any(s[2] != stages[0][2] for s in stages):
        return None, None, {'kind': 'factory_specialization', 'path': unit.path,
                            'reason': 'Unsupported or unresolved fixed stage construction',
                            'source_refs': [span(unit, target_class)]}
    count_total = sum(s[3] for s in stages)
    if count_total > _MAX_BLOCKS:
        return None, None, {'kind': 'factory_specialization', 'path': unit.path,
                            'reason': 'Factory exceeds static block limit',
                            'source_refs': [span(unit, target_class)]}
    block = stages[0][2]
    cls_id = stable_id(unit.path, 'semantic', 'class_definition', syntax_ids(unit)[target_class], '')
    config_id = stable_id(unit.path, 'factory-configuration', source_factory_id, cls_id, block.ident,
                          tuple(n for _, _, _, n in stages))
    graph.add_node({'id': config_id, 'kind': 'factory_configuration',
                    'attributes': {'factory': symbol_name, 'model': target_class.name, 'block': block.name,
                                   'stage_block_counts': [s[3] for s in stages], 'total_blocks': count_total,
                                   'scope': 'static_constructor_blueprint'},
                    'source_refs': [span(unit, factory), span(unit, chain[0])],
                    'evidence_class': 'derived', 'resolution': 'conditional', 'analysis_rule': RULE})
    derived_edge(graph, unit, factory, source_factory_id, config_id, 'CONFIGURES_MODEL_FACTORY', RULE)
    derived_edge(graph, unit, target_class, config_id, cls_id, 'FACTORY_CONSTRUCTS_CLASS', RULE)
    derived_edge(graph, unit, target_class, config_id, block.ident, 'FACTORY_SELECTS_BLOCK', RULE)
    stage_records = []
    previous = None
    _, first_append, repeat_loop = template
    for index, (stmt, name, _, num) in enumerate(stages):
        stage_id = stable_id(unit.path, 'factory-stage', config_id, index, name, num)
        graph.add_node({'id': stage_id, 'kind': 'factory_stage',
                        'attributes': {'name': name, 'index': index, 'block_type': block.name,
                                       'count': num, 'scope': 'static_constructor_blueprint'},
                        'source_refs': [span(unit, stmt), span(unit, first_append), span(unit, repeat_loop)],
                        'evidence_class': 'derived', 'resolution': 'conditional', 'analysis_rule': RULE})
        derived_edge(graph, unit, stmt, config_id, stage_id, 'FACTORY_HAS_STAGE', RULE)
        if previous is not None:
            derived_edge(graph, unit, stmt, previous, stage_id, 'STATIC_STAGE_ORDER', RULE)
        previous = stage_id
        block_ids = []
        for pos in range(num):
            inst_id = stable_id(unit.path, 'factory-block', stage_id, pos, block.ident)
            site = first_append if pos == 0 else repeat_loop
            graph.add_node({'id': inst_id, 'kind': 'factory_block_instance',
                            'attributes': {'stage': name, 'position': pos, 'block_type': block.name,
                                           'scope': 'static_constructor_blueprint'},
                            'source_refs': [span(unit, stmt), span(unit, site)],
                            'evidence_class': 'derived', 'resolution': 'conditional', 'analysis_rule': RULE})
            derived_edge(graph, unit, site, stage_id, inst_id, 'STAGE_HAS_BLOCK_INSTANCE', RULE)
            derived_edge(graph, unit, site, inst_id, block.ident, 'BLOCK_INSTANCE_OF_CLASS', RULE)
            if block_ids:
                derived_edge(graph, unit, site, block_ids[-1], inst_id, 'STATIC_BLOCK_ORDER', RULE)
            block_ids.append(inst_id)
        stage_records.append({'stage_id': stage_id, 'name': name, 'position': index,
                              'block_type': block.name, 'block_definition_id': block.ident,
                              'block_instance_ids': block_ids, 'count': num, 'source_refs': [span(unit, stmt)]})
    spec = {'factory_definition_id': source_factory_id, 'factory_configuration_id': config_id,
            'model_definition_id': cls_id, 'block_definition_id': block.ident,
            'block_type': block.name, 'stage_counts': [s['count'] for s in stage_records],
            'total_blocks': count_total, 'stages': stage_records,
            'scope': 'static_constructor_blueprint', 'resolution': 'conditional'}
    return cls_id, spec, None
