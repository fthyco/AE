"""Concrete shape facts for supported PyTorch tensor operations."""
from __future__ import annotations

import ast
from collections import defaultdict
from math import prod

from ..ir.canonical import stable_id
from ..ir.models import Graph
from ..linking.project import Unit, span, syntax_ids
from .scopes import alias_is_shadowed, parent_links

RULE = 'concrete-torch-operations/v1'
_CREATOR = {'zeros', 'ones', 'empty', 'rand', 'randn'}
_OPERATIONS = {'cat', 'matmul', 'reshape'}


def _int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _int(node.operand)
        return -value if value is not None else None
    return None


def _literal_dims(nodes: list[ast.AST]) -> tuple[int, ...] | None:
    if len(nodes) == 1 and isinstance(nodes[0], (ast.Tuple, ast.List)):
        nodes = nodes[0].elts
    dims = [_int(node) for node in nodes]
    return tuple(dims) if all(dim is not None for dim in dims) else None


def _shape_map(graph: Graph) -> dict[str, tuple[int, ...]]:
    result = {}
    for node in graph.nodes:
        if node['kind'] != 'symbolic_shape_fact':
            continue
        dims = node['attributes']['dimensions']
        if dims and all(isinstance(d, str) and d.isdecimal() and int(d) > 0 for d in dims):
            result[node['attributes']['value_id']] = tuple(int(d) for d in dims)
    return result


def _batch_broadcast(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...] | None:
    result = []
    for a, b in zip(reversed(left), reversed(right)):
        if a != b and a != 1 and b != 1:
            return None
        result.append(max(a, b))
    rest = left if len(left) > len(right) else right
    result.extend(reversed(rest[:abs(len(left) - len(right))]))
    return tuple(reversed(result))


def _matmul(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...] | None:
    if not left or not right:
        return None
    a = (1, left[0]) if len(left) == 1 else left[-2:]
    b = (right[0], 1) if len(right) == 1 else right[-2:]
    if a[-1] != b[-2]:
        return None
    batch = _batch_broadcast(left[:-2] if len(left) > 1 else (),
                             right[:-2] if len(right) > 1 else ())
    if batch is None:
        return None
    tail = ([] if len(left) == 1 else [a[-2]]) + ([] if len(right) == 1 else [b[-1]])
    return (*batch, *tail)


def _reshape(source: tuple[int, ...], target: tuple[int, ...]) -> tuple[int, ...] | None:
    if any(x == 0 or x < -1 for x in target) or target.count(-1) > 1:
        return None
    total = prod(source)
    fixed = prod(x for x in target if x != -1)
    if -1 in target:
        if fixed <= 0 or total % fixed:
            return None
        return tuple(total // fixed if x == -1 else x for x in target)
    return target if total == fixed else None


def _cat(parts: list[tuple[int, ...]], dim: int) -> tuple[int, ...] | None:
    if not parts or not parts[0]:
        return None
    rank = len(parts[0])
    if any(len(p) != rank for p in parts) or dim < -rank or dim >= rank:
        return None
    axis = dim % rank
    if any(any(a != b for i, (a, b) in enumerate(zip(parts[0], p)) if i != axis)
           for p in parts[1:]):
        return None
    shape = list(parts[0])
    shape[axis] = sum(p[axis] for p in parts)
    return tuple(shape)


def analyze_composite_shapes(graph: Graph, units: list[Unit], model_index: dict) -> None:
    index_fids = {m.get('forward_method_id') for m in model_index.get('modules', [])}
    if not index_fids or index_fids == {None}:
        return
    known = _shape_map(graph)
    templates = {n['attributes']['value_id']:tuple(n['attributes']['dimensions'])
                 for n in graph.nodes if n['kind'] == 'symbolic_shape_fact'}
    template_causes = {n['attributes']['value_id']:n['attributes'].get('reason')
                       for n in graph.nodes if n['kind'] == 'symbolic_shape_fact'}
    all_nodes = {n['id']: n for n in graph.nodes}
    incoming = defaultdict(list)
    members = defaultdict(list)
    collections = defaultdict(list)
    for edge in graph.edges:
        incoming[edge['target']].append(edge)
        if edge['relation'] == 'STATIC_COLLECTION_MEMBER':
            members[edge['target']].append(edge['source'])
        elif edge['relation'] == 'STATIC_COLLECTION_ELEMENT':
            collections[edge['target']].append(edge['source'])

    candidates = []
    for unit in sorted(units, key=lambda u: u.path):
        ids = syntax_ids(unit)
        parents = parent_links(unit.tree)
        imports = {}
        for stmt in unit.tree.body:
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if alias.name == 'torch':
                        imports[alias.asname or 'torch'] = True
            elif isinstance(stmt, ast.ImportFrom) and stmt.module == 'torch' and stmt.level == 0:
                for alias in stmt.names:
                    if alias.name in _CREATOR | _OPERATIONS:
                        imports[alias.asname or alias.name] = alias.name
        rebound = set()
        for statement in unit.tree.body:
            targets = (statement.targets if isinstance(statement, ast.Assign) else
                       [statement.target] if isinstance(statement, (ast.AnnAssign, ast.AugAssign)) else [])
            for target in targets:
                root = target
                while isinstance(root, (ast.Attribute, ast.Subscript)):
                    root = root.value
                if isinstance(root, ast.Name):
                    rebound.add(root.id)
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                rebound.add(statement.name)
        for alias in rebound:
            imports.pop(alias, None)

        def qualified(node: ast.AST) -> str | None:
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if imports.get(node.value.id) is True and not alias_is_shadowed(node, node.value.id, parents):
                    return node.attr
            if isinstance(node, ast.Name) and imports.get(node.id) and imports[node.id] is not True:
                if not alias_is_shadowed(node, node.id, parents):
                    return imports[node.id]
            return None
        for module in model_index.get('modules', []):
            fid = module.get('forward_method_id')
            if not fid:
                continue
            for stmt in unit.tree.body:
                if not isinstance(stmt, ast.ClassDef):
                    continue
                for fn in stmt.body:
                    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    fnid = stable_id(unit.path, 'semantic', 'method_definition', ids[fn], '')
                    if fnid != fid:
                        continue
                    for node in ast.walk(fn):
                        if not isinstance(node, (ast.Call, ast.BinOp)):
                            continue
                        if isinstance(node, ast.Call):
                            mode = qualified(node.func)
                            if mode not in _CREATOR | _OPERATIONS:
                                if isinstance(node.func, ast.Attribute) and node.func.attr in {'reshape','view'}:
                                    mode = 'tensor_reshape'
                                else:
                                    continue
                            ident = stable_id(unit.path, 'semantic', 'call_site', ids[node], '')
                        else:
                            if not isinstance(node.op, ast.MatMult):
                                continue
                            mode = 'matmul_operator'
                            ident = stable_id(unit.path, 'semantic', 'operation_site', ids[node], '')
                        if ident in all_nodes:
                            candidates.append((unit, node, mode, ident))
    def dependency(unit: Unit, owner: str, arg: ast.AST) -> str | None:
        ref = span(unit, arg)
        matches = [e['source'] for e in incoming.get(owner, [])
                   if e['relation'] == 'STATIC_VALUE_DEPENDENCY' and
                   any(r['path'] == ref['path'] and r['start_byte'] == ref['start_byte'] and
                       r['end_byte'] == ref['end_byte'] for r in e['source_refs'])]
        values = sorted(set(matches))
        return values[0] if len(values) == 1 else None

    def infer(unit: Unit, node: ast.AST, mode: str, ident: str) -> tuple[int, ...] | None:
        if isinstance(node, ast.Call) and mode in _CREATOR:
            dims = _literal_dims(node.args)
            return dims if dims and all(x > 0 for x in dims) and not node.keywords else None
        if mode in {'matmul', 'matmul_operator'}:
            args = ([node.left, node.right] if isinstance(node, ast.BinOp) else node.args)
            if len(args) != 2 or (isinstance(node, ast.Call) and node.keywords):
                return None
            sources = [dependency(unit, ident, arg) for arg in args]
            if any(v not in known for v in sources):
                return None
            return _matmul(known[sources[0]], known[sources[1]])
        if mode in {'reshape', 'tensor_reshape'}:
            if not isinstance(node, ast.Call) or node.keywords:
                return None
            if mode == 'reshape' and len(node.args) == 2:
                source, dims = node.args
                if not isinstance(dims, (ast.Tuple, ast.List)):
                    return None
                target = _literal_dims([dims])
            elif mode == 'tensor_reshape' and isinstance(node.func, ast.Attribute):
                source, target = node.func.value, _literal_dims(node.args)
            else:
                return None
            original = dependency(unit, ident, source)
            return _reshape(known[original], target) if original in known and target is not None else None
        if mode == 'cat' and isinstance(node, ast.Call):
            if not node.args or len(node.args) > 2 or any(k.arg != 'dim' for k in node.keywords):
                return None
            dimension = next((k.value for k in node.keywords if k.arg == 'dim'),
                             node.args[1] if len(node.args) == 2 else ast.Constant(value=0))
            axis = _int(dimension)
            if axis is None:
                return None
            collection = dependency(unit, ident, node.args[0])
            if collection is None:
                return None
            element_ids = members.get(collection, [])
            if not element_ids:
                return None
            indexed = [(all_nodes[e]['attributes'].get('element_index'), e) for e in element_ids
                       if e in all_nodes]
            if len(indexed) != len(element_ids) or any(type(i) is not int for i, _ in indexed):
                return None
            indexed.sort()
            if [i for i, _ in indexed] != list(range(len(indexed))):
                return None
            parts = []
            for _, el in indexed:
                if el in known:
                    parts.append(known[el])
                else:
                    operands = collections[el]
                    if len(set(operands)) != 1 or operands[0] not in known:
                        return None
                    parts.append(known[operands[0]])
            return _cat(parts, axis)
        return None

    def cat_shared_linear_prefix(unit: Unit, node: ast.AST, ident: str) -> tuple[str, ...] | None:
        if not isinstance(node, ast.Call) or not node.args or len(node.args) > 2:
            return None
        if any(k.arg != 'dim' for k in node.keywords):
            return None
        axis_node = next((k.value for k in node.keywords if k.arg == 'dim'),
                         node.args[1] if len(node.args) == 2 else ast.Constant(value=0))
        if _int(axis_node) != -1:
            return None
        aggregate = dependency(unit, ident, node.args[0])
        if aggregate is None:
            return None
        indexed = sorted((all_nodes[e]['attributes'].get('element_index'), e)
                         for e in members.get(aggregate, ()) if e in all_nodes)
        if len(indexed) < 2 or [i for i, _ in indexed] != list(range(len(indexed))):
            return None
        bases, widths = [], []
        for _, member in indexed:
            sources = sorted(set(collections[member]))
            if len(sources) != 1 or template_causes.get(sources[0]) != 'linear-output-template':
                return None
            origin = sources[0]
            dims = templates.get(origin)
            if dims is None or len(dims) != 2 or dims[0] != '...' or not dims[1].isdecimal():
                return None
            deps = [e['source'] for e in incoming.get(origin, [])
                    if e['relation'] == 'STATIC_VALUE_DEPENDENCY']
            if len(set(deps)) != 1:
                return None
            bases.append(deps[0])
            widths.append(int(dims[1]))
        return ('...', str(sum(widths))) if len(set(bases)) == 1 else None

    # Revisit derived nodes until their static operands become available.
    for _ in range(len(candidates) + 1):
        changed = False
        for unit, node, mode, ident in candidates:
            result = infer(unit, node, mode, ident)
            if result is None and mode == 'cat':
                result = cat_shared_linear_prefix(unit, node, ident)
            if result is None or not result:
                continue
            if any(type(v) is int and v <= 0 for v in result):
                continue
            normalized = tuple(str(v) for v in result)
            if ident in templates:
                if templates[ident] != normalized:
                    graph.unresolved.append({'kind':'composite_shape_conflict', 'path':unit.path,
                                             'site_id':ident, 'reason':'Conflicting shape facts',
                                             'source_refs':[span(unit,node)]})
                continue
            if all(type(v) is int for v in result):
                known[ident] = result
            templates[ident] = normalized
            changed = True
            fact_id = stable_id('ae-composite-shape/v1', ident, result)
            graph.add_node({'id':fact_id, 'kind':'symbolic_shape_fact',
                            'attributes':{'value_id':ident, 'dimensions':[str(x) for x in result],
                                          'scope':'conditional_type_level', 'reason':mode},
                            'source_refs':[span(unit,node)], 'evidence_class':'derived',
                            'analysis_rule':RULE, 'resolution':'conditional'})
            graph.add_edge({'id':stable_id('ae-composite-shape-edge/v1', ident,fact_id),
                            'source':ident,'target':fact_id,'relation':'HAS_SYMBOLIC_SHAPE',
                            'source_refs':[span(unit,node)],'evidence_class':'derived',
                            'analysis_rule':RULE})
        if not changed:
            break
