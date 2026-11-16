"""Soundly scoped, non-executing call argument/return bindings.

Only uniquely linked, single-use, nonrecursive, statically bound local calls
are summarized. No runtime call graph or general Python binding is claimed.
"""
from __future__ import annotations

import ast
from collections import Counter

from ..ir.canonical import stable_id
from ..ir.models import Graph
from ..linking.project import Unit, derived_edge, span, syntax_ids

RULE = 'unique-static-call-summary/v1'


def _function_id(unit: Unit, node: ast.AST, kind: str, ids: dict) -> str:
    return stable_id(unit.path, 'semantic', kind, ids[node], '')


def analyze_interprocedural(graph: Graph, units: list[Unit], model_index: dict) -> None:
    nodes = {node['id']: node for node in graph.nodes}
    edges = list(graph.edges)
    linked = {}
    for edge in edges:
        if edge['relation'] == 'CALLS_LOCAL_DEFINITION':
            linked.setdefault(edge['source'], set()).add(edge['target'])

    functions = {}
    call_sites = []
    for unit in sorted(units, key=lambda u: u.path):
        ids = syntax_ids(unit)
        for declaration in unit.tree.body:
            declarations = ([(declaration, 'function_definition')] if
                            isinstance(declaration, (ast.FunctionDef, ast.AsyncFunctionDef)) else
                            [(m, 'method_definition') for m in declaration.body
                             if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
                            if isinstance(declaration, ast.ClassDef) else [])
            parent_class_id = (_function_id(unit, declaration, 'class_definition', ids)
                               if isinstance(declaration, ast.ClassDef) else None)
            for fn, kind in declarations:
                ident = _function_id(unit, fn, kind, ids)
                functions[ident] = (unit, fn, parent_class_id, ids)
                if not any(n['kind'] == 'forward_input' and n['attributes'].get('forward_id') == ident
                           for n in graph.nodes):
                    continue
                # Do not follow nested function/class bodies. Call expressions
                # in a supported outer function are source-backed call sites.
                def calls_in(root):
                    for child in ast.iter_child_nodes(root):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                            continue
                        if isinstance(child, ast.Call):
                            yield child
                        yield from calls_in(child)
                for call in calls_in(fn):
                    site = _function_id(unit, call, 'call_site', ids)
                    call_sites.append((site, ident, parent_class_id, unit, call, ids))

    modules = {m['definition_id']: m for m in model_index.get('modules', [])}
    proposals = []
    for site, caller, parent_class, unit, call, ids in call_sites:
        candidates = set(linked.get(site, ()))
        if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == 'self':
            owner = f'self.{call.func.attr}'
            components = [c for c in modules.get(parent_class, {}).get('components', [])
                          if c['owner_expression'] == owner and c.get('target_definition_id')]
            if len(components) == 1:
                child = modules.get(components[0]['target_definition_id'])
                if child and child.get('forward_method_id'):
                    candidates.add(child['forward_method_id'])
            elif len(components) > 1:
                candidates.update(c['target_definition_id'] for c in components)
        if len(candidates) == 1:
            target = next(iter(candidates))
            if target in functions:
                proposals.append((site, caller, target, unit, call, ids))
        elif len(candidates) > 1:
            graph.unresolved.append({'kind':'interprocedural_flow_incomplete', 'path':unit.path,
                                     'function_id':caller, 'site_id':site,
                                     'reason':'Ambiguous call target; no interprocedural value binding',
                                     'candidate_ids':sorted(c for c in candidates if c in nodes),
                                     'source_refs':[span(unit,call)]})

    use_counts = Counter(target for _,_,target,*_ in proposals)
    adjacency = {}
    for _, caller, callee, *_ in proposals:
        adjacency.setdefault(caller, set()).add(callee)

    def recursive(caller: str, callee: str) -> bool:
        pending, seen = [callee], set()
        while pending:
            current = pending.pop()
            if current == caller:
                return True
            if current not in seen:
                seen.add(current)
                pending.extend(sorted(adjacency.get(current, ())))
        return False

    def record(unit, call, caller, site, reason):
        graph.unresolved.append({'kind':'interprocedural_flow_incomplete', 'path':unit.path,
                                 'function_id':caller, 'site_id':site, 'reason':reason,
                                 'source_refs':[span(unit,call)]})

    for site, caller, callee, unit, call, ids in proposals:
        if use_counts[callee] != 1 or recursive(caller, callee):
            record(unit, call, caller, site, 'Repeated or recursive callee; context-specific summary not proven')
            continue
        _, target_fn, _, target_ids = functions[callee]
        positional = [*target_fn.args.posonlyargs, *target_fn.args.args]
        if positional and positional[0].arg in ('self', 'cls'):
            positional = positional[1:]
        formal = [*positional, *target_fn.args.kwonlyargs]
        if (target_fn.args.vararg or target_fn.args.kwarg or target_fn.args.defaults
                or any(v is not None for v in target_fn.args.kw_defaults)
                or any(isinstance(x, ast.Starred) for x in call.args)
                or any(k.arg is None for k in call.keywords)
                or len(call.args) > len(positional)):
            record(unit, call, caller, site, 'Variadic, defaulted or dynamic argument binding not supported')
            continue
        param_map = {}
        for index, arg in enumerate(call.args):
            param_map[positional[index].arg] = arg
        for keyword in call.keywords:
            if (keyword.arg in param_map or keyword.arg not in {p.arg for p in formal}
                    or keyword.arg in {p.arg for p in target_fn.args.posonlyargs}):
                param_map = {}
                break
            param_map[keyword.arg] = keyword.value
        if set(param_map) != {p.arg for p in formal}:
            record(unit, call, caller, site, 'Actual and formal arguments not unambiguously matched')
            continue
        inputs = {n['attributes']['name']:n['id'] for n in graph.nodes
                  if n['kind'] == 'forward_input' and n['attributes'].get('forward_id') == callee}
        outputs = [n['id'] for n in graph.nodes
                   if n['kind'] == 'forward_output' and n['attributes'].get('forward_id') == callee]
        if len(outputs) != 1 or set(inputs) != set(param_map):
            record(unit, call, caller, site, 'Callee has no unique analyzable input/output summary')
            continue
        if not any(e['target'] == outputs[0] and e['relation'] == 'STATIC_FORWARD_RETURN' for e in edges):
            record(unit, call, caller, site, 'Callee return value is not statically supported')
            continue
        binding = []
        for name, expr in sorted(param_map.items()):
            ref = span(unit, expr)
            sources = {e['source'] for e in edges if e['target'] == site
                       and e['relation'] == 'STATIC_VALUE_DEPENDENCY'
                       and any(s['path'] == ref['path'] and s['start_byte'] == ref['start_byte']
                               and s['end_byte'] == ref['end_byte'] for s in e['source_refs'])}
            if len(sources) != 1:
                binding = []
                break
            binding.append((next(iter(sources)), inputs[name], expr, name))
        if len(binding) != len(formal):
            record(unit, call, caller, site, 'Actual argument provenance ambiguous or unavailable')
            continue
        for value, formal_id, at, name in binding:
            derived_edge(graph, unit, at, value, formal_id, 'STATIC_CALL_ARGUMENT_BINDING', RULE, label=name)
        derived_edge(graph, unit, call, outputs[0], site, 'STATIC_CALL_RETURN', RULE)
