"""Source-backed nested module construction templates."""
from __future__ import annotations

import ast
from collections import defaultdict

from ..ir.canonical import stable_id
from ..ir.models import Graph
from ..linking.project import Unit, derived_edge, span, syntax_ids

RULE = 'source-nested-module-template/v1'


def collect_nested_templates(graph: Graph, definitions: dict[str, tuple[Unit, ast.ClassDef]],
                             definition_id: str) -> list[dict]:
    """Find local module variables passed to append/add_module sites."""
    unit, klass = definitions[definition_id]
    ids = syntax_ids(unit)
    by_name: dict[str, list[str]] = defaultdict(list)
    for ident, (u, candidate) in definitions.items():
        if u.path == unit.path:
            by_name[candidate.name].append(ident)
    for name in by_name:
        by_name[name].sort()
    constructors = [m for m in klass.body if isinstance(m, ast.FunctionDef)
                    and m.name == '__init__']
    if len(constructors) != 1:
        return []
    observed_containers: set[str] = set()
    for stmt in ast.walk(constructors[0]):
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign)) or not isinstance(stmt.value, ast.Call):
            continue
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        for target in targets:
            if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                    and target.value.id == 'self' and
                    (ast.unparse(stmt.value.func).split('.')[-1] in
                     {'ModuleList', 'ModuleDict', 'Sequential'})):
                observed_containers.add(ast.unparse(target))
    if any(ast.unparse(base).split('.')[-1] in {'Module', 'ModuleDict', 'ModuleList', 'Sequential'}
           for base in klass.bases):
        observed_containers.add('self')
    templates: list[dict] = []
    emitted: set[tuple[str, str]] = set()

    def walk(statements: list[ast.stmt], variables: dict[str, tuple[str, ast.Call]],
             loops: tuple[ast.For, ...], guards: tuple[ast.If, ...]):
        env = dict(variables)
        for stmt in statements:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        env.pop(target.id, None)
                        call = stmt.value
                        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                            options = by_name.get(call.func.id, [])
                            if len(options) == 1:
                                env[target.id] = (options[0], call)
            elif isinstance(stmt, ast.For):
                walk(stmt.body, {}, (*loops, stmt), guards)
                walk(stmt.orelse, {}, loops, guards)
                # A loop might modify earlier locals.
                for child in stmt.body:
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name):
                                env.pop(target.id, None)
            elif isinstance(stmt, ast.If):
                walk(stmt.body, env, loops, (*guards, stmt))
                walk(stmt.orelse, env, loops, (*guards, stmt))
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if not isinstance(call.func, ast.Attribute) or call.func.attr not in ('append', 'add_module'):
                    continue
                container = ast.unparse(call.func.value)
                if container not in observed_containers:
                    continue
                arg_pos = 0 if call.func.attr == 'append' else 1
                if len(call.args) <= arg_pos or not isinstance(call.args[arg_pos], ast.Name):
                    continue
                bound = env.get(call.args[arg_pos].id)
                if bound is None:
                    continue
                child_id, constructor = bound
                if (child_id, ids[call]) in emitted:
                    continue
                emitted.add((child_id, ids[call]))
                derived_edge(graph, unit, call, definition_id, child_id,
                             'HAS_NESTED_MODULE_TEMPLATE', RULE)
                templates.append({
                    'target_definition_id': child_id,
                    'append_call_site_id': stable_id(unit.path, 'semantic', 'call_site', ids[call], ''),
                    'constructor_call_site_id': stable_id(unit.path, 'semantic', 'call_site', ids[constructor], ''),
                    'container_expression': container,
                    'repeat_expressions': [ast.unparse(loop.iter) for loop in loops],
                    'guard_expressions': [ast.unparse(guard.test) for guard in guards],
                    'count': None if (loops or guards) else 1,
                    'resolution': 'conditional',
                    'source_refs': [span(unit, constructor), span(unit, call)],
                })

    walk(constructors[0].body, {}, (), ())
    return sorted(templates, key=lambda r:(r['source_refs'][-1]['start_byte'],r['target_definition_id']))
