"""Source-backed fixed-count nested container construction."""
from __future__ import annotations

import ast
from typing import Any

from ..ir.canonical import stable_id
from ..ir.models import Graph
from ..linking.project import Unit, derived_edge, span, syntax_ids
from .factory_specialization import _bind, _selected_constructor, _Symbol

RULE = 'fixed-container-construction/v1'
MAX_STAGES = 32
MAX_MEMBERS = 256


def _single_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    found = [x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == name]
    return found[0] if len(found) == 1 else None


def _registered(statements: list[ast.stmt], variable: str, owner: str, method: str) -> ast.Call | None:
    calls = []
    for stmt in statements:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            continue
        call = stmt.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr != method:
            continue
        if ast.unparse(call.func.value) != owner:
            continue
        if not call.args or not isinstance(call.args[-1], ast.Name) or call.args[-1].id != variable:
            continue
        if (method == 'add_module' and len(call.args) == 2 or
                method == 'append' and len(call.args) == 1):
            calls.append(call)
    return calls[0] if len(calls) == 1 else None


def _chain(unit: Unit, requested: str):
    name = requested.split(':')[-1]
    funcs = [f for f in unit.tree.body if isinstance(f, ast.FunctionDef)]
    classes = [c for c in unit.tree.body if isinstance(c, ast.ClassDef)]
    if len({f.name for f in funcs}) != len(funcs) or len({c.name for c in classes}) != len(classes):
        return None
    by_func = {f.name: f for f in funcs}
    by_class = {c.name: c for c in classes}
    current = by_func.get(name)
    if current is None:
        return None
    factory = current
    symbols = {c.name: _Symbol(c.name, stable_id(unit.path, 'semantic', 'class_definition', syntax_ids(unit)[c], ''), 'class_definition') for c in classes}
    bound: dict[str, Any] = {}
    calls: list[ast.Call] = []
    visited: set[str] = set()
    for _ in range(6):
        if current.name in visited:
            return None
        visited.add(current.name)
        call = _selected_constructor(current)
        if call is None or not isinstance(call.func, ast.Name) or any(isinstance(arg, ast.Starred) for arg in call.args):
            return None
        # Factory arguments used in a constructor cannot be locally rebound.
        call_names = {n.id for value in [*call.args, *(kw.value for kw in call.keywords)]
                      for n in ast.walk(value) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        tracked = {name for name in call_names if name in bound and
                   (type(bound[name]) is int or isinstance(bound[name], tuple))}
        if any(isinstance(n, ast.Name) and n.id in tracked and isinstance(n.ctx, (ast.Store, ast.Del))
               for statement in current.body for n in ast.walk(statement)):
            return None
        calls.append(call)
        target = call.func.id
        if target in by_class:
            klass = by_class[target]
            init = _single_method(klass, '__init__')
            if init is None:
                return None
            init_env = _bind(init, call, bound, symbols, skip_self=True)
            if init_env is None:
                return None
            return factory, klass, init, init_env, calls, by_class
        nxt = by_func.get(target)
        if nxt is None:
            return None
        nxt_env = _bind(nxt, call, bound, symbols)
        if nxt_env is None:
            return None
        current, bound = nxt, nxt_env
    return None


def _stage_pattern(init: ast.FunctionDef, names: dict[str, ast.ClassDef]):
    loops = [x for x in init.body if isinstance(x, ast.For)
             and isinstance(x.iter, ast.Call) and isinstance(x.iter.func, ast.Name)
             and x.iter.func.id == 'enumerate']
    if len(loops) != 1:
        return None
    loop = loops[0]
    if (not isinstance(loop.target, ast.Tuple) or len(loop.target.elts) != 2 or
            not all(isinstance(n, ast.Name) for n in loop.target.elts)):
        return None
    index_name, count_name = (n.id for n in loop.target.elts)
    it = loop.iter
    if (not isinstance(it, ast.Call) or not isinstance(it.func, ast.Name) or
            it.func.id != 'enumerate' or len(it.args) != 1 or it.keywords or
            not isinstance(it.args[0], ast.Name) or loop.orelse):
        return None
    config_name = it.args[0].id
    # No assignment to block_config outside this iterable is supported.
    if any(isinstance(n, ast.Name) and n.id == config_name and isinstance(n.ctx, (ast.Store, ast.Del))
           for n in ast.walk(init)):
        return None
    containers = set()
    for stmt in init.body:
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and
                isinstance(stmt.targets[0], ast.Attribute) and
                isinstance(stmt.targets[0].value, ast.Name) and stmt.targets[0].value.id == 'self' and
                isinstance(stmt.value, ast.Call) and
                ast.unparse(stmt.value.func).split('.')[-1] in {'Sequential', 'ModuleList', 'ModuleDict'}):
            containers.add(ast.unparse(stmt.targets[0]))
    if len(containers) != 1:
        return None
    container = next(iter(containers))
    container_name = container.split('.')[-1]
    container_assignments = [n for n in ast.walk(init) if isinstance(n, ast.Attribute)
                             and isinstance(n.ctx, (ast.Store, ast.Del))
                             and isinstance(n.value, ast.Name) and n.value.id == 'self'
                             and n.attr == container_name]
    if len(container_assignments) != 1:
        return None
    construction = [(stmt, stmt.targets[0].id, stmt.value.func.id)
                    for stmt in loop.body if isinstance(stmt, ast.Assign) and
                    len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name)
                    and stmt.value.func.id in names]
    immediate = []
    for stmt, variable, klass_name in construction:
        call = _registered(loop.body, variable, container, 'add_module')
        if call is not None:
            immediate.append((stmt, variable, klass_name, call))
    if len(immediate) != 1:
        return None
    block_assign, block_var, block_class, block_reg = immediate[0]
    block_region = loop.body[loop.body.index(block_assign) + 1:]
    if any(isinstance(n, ast.Name) and n.id == block_var and isinstance(n.ctx, (ast.Store, ast.Del))
           for stmt in block_region for n in ast.walk(stmt)):
        return None
    block_call = block_assign.value
    block_initializer = _single_method(names[block_class], '__init__')
    inner = _member_pattern(names[block_class], names)
    if block_initializer is None or inner is None:
        return None
    member_param = inner[0].iter.args[0].id
    parameters = [arg.arg for arg in block_initializer.args.args[1:]]
    if member_param not in parameters:
        return None
    at = parameters.index(member_param)
    block_count_arg = block_call.args[at] if at < len(block_call.args) else next(
        (kw.value for kw in block_call.keywords if kw.arg == member_param), None)
    if not isinstance(block_count_arg, ast.Name) or block_count_arg.id != count_name:
        return None
    guards = [x for x in loop.body if isinstance(x, ast.If)]
    if len(guards) != 1 or guards[0].orelse:
        return None
    guard = guards[0]
    # Verify i != len(block_config) - 1; no arbitrary conditional expansion.
    if ast.dump(guard.test, include_attributes=False) != ast.dump(
        ast.parse(f'{index_name} != len({config_name}) - 1', mode='eval').body,
        include_attributes=False):
        return None
    transition = [(stmt, stmt.targets[0].id, stmt.value.func.id)
                  for stmt in guard.body if isinstance(stmt, ast.Assign) and
                  len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
                  and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name)
                  and stmt.value.func.id in names]
    transitions = [(stmt, variable, cls_name, _registered(guard.body, variable, container, 'add_module'))
                   for stmt, variable, cls_name in transition]
    transitions = [t for t in transitions if t[3] is not None]
    if len(transitions) != 1:
        return None
    trans_assign, trans_var, trans_class, trans_reg = transitions[0]
    # No unrecognized module registration in the selected loop.
    registrations = [n for n in ast.walk(loop) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_module'
                     and ast.unparse(n.func.value) == container]
    if len(registrations) != 2:
        return None
    return config_name, container, loop, (block_assign, block_reg, block_class), (trans_assign, trans_reg, trans_class)


def _member_pattern(klass: ast.ClassDef, names: dict[str, ast.ClassDef]):
    init = _single_method(klass, '__init__')
    if init is None:
        return None
    loops = [x for x in init.body if isinstance(x, ast.For)]
    if len(loops) != 1:
        return None
    loop = loops[0]
    if (not isinstance(loop.target, ast.Name) or not isinstance(loop.iter, ast.Call) or
            not isinstance(loop.iter.func, ast.Name) or loop.iter.func.id != 'range' or
            len(loop.iter.args) != 1 or not isinstance(loop.iter.args[0], ast.Name) or
            loop.iter.keywords or loop.orelse):
        return None
    count_param = loop.iter.args[0].id
    if count_param not in [a.arg for a in init.args.args[1:]]:
        return None
    if any(isinstance(n, ast.Name) and n.id == count_param and isinstance(n.ctx, (ast.Store, ast.Del))
           for n in ast.walk(init)):
        return None
    constructors = [(stmt, stmt.targets[0].id, stmt.value.func.id)
                    for stmt in loop.body if isinstance(stmt, ast.Assign) and
                    len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name)
                    and stmt.value.func.id in names]
    candidates = [(stmt, variable, name, _registered(loop.body, variable, 'self', 'add_module'))
                  for stmt, variable, name in constructors]
    candidates = [c for c in candidates if c[3] is not None]
    if len(candidates) != 1 or any(isinstance(s, (ast.If, ast.While, ast.Try, ast.Break, ast.Continue))
                                   for s in loop.body):
        return None
    member_assignment, member_var, _, _ = candidates[0]
    after = loop.body[loop.body.index(member_assignment) + 1:]
    if any(isinstance(n, ast.Name) and n.id == member_var and isinstance(n.ctx, (ast.Store, ast.Del))
           for stmt in after for n in ast.walk(stmt)):
        return None
    return loop, candidates[0]


def specialize_containers(graph: Graph, units: list[Unit], requested: str | None):
    """Resolve bounded constructor registration counts from a unique factory chain."""
    if not requested or ':' not in requested:
        return None, None
    module, _ = requested.split(':', 1)
    options = [unit for unit in units if unit.module == module]
    if len(options) != 1:
        return None, None
    unit = options[0]
    chain = _chain(unit, requested)
    if chain is None:
        return None, None
    factory, root_class, init, env, calls, classes = chain
    pattern = _stage_pattern(init, classes)
    if pattern is None:
        return None, None
    config_var, container_expression, stage_loop, block_parts, transition_parts = pattern
    counts = env.get(config_var)
    if (not isinstance(counts, tuple) or not 1 <= len(counts) <= MAX_STAGES or
            any(type(v) is not int or not 1 <= v <= MAX_MEMBERS for v in counts) or
            sum(counts) > MAX_MEMBERS):
        return None, None
    block_assign, block_registration, block_name = block_parts
    trans_assign, trans_registration, trans_name = transition_parts
    member = _member_pattern(classes[block_name], classes)
    if member is None:
        return None, None
    member_loop, (member_assign, member_name, member_class, member_registration) = member
    # A fixed factory chain selects the count; constructor statements carry the template.
    id_maps = syntax_ids(unit)
    def ident(node: ast.AST, kind: str) -> str:
        return stable_id(unit.path, 'semantic', kind, id_maps[node], '')
    root_id, block_id, trans_id, member_id = (ident(classes[x], 'class_definition')
                                             for x in (root_class.name, block_name, trans_name, member_class))
    factory_id = ident(factory, 'function_definition')
    config_id = stable_id(unit.path, 'container-config', factory_id, root_id, counts,
                          block_id, member_id, trans_id)
    graph.add_node({'id': config_id, 'kind': 'container_configuration',
                    'attributes': {'factory': factory.name, 'model': root_class.name,
                                   'stage_counts': list(counts), 'total_members': sum(counts),
                                   'container_expression': container_expression, 'scope': 'static_constructor_blueprint'},
                    'source_refs': [span(unit, factory), span(unit, calls[0]),
                                    span(unit, stage_loop), span(unit, member_loop)],
                    'evidence_class': 'derived', 'analysis_rule': RULE, 'resolution': 'conditional'})
    derived_edge(graph, unit, factory, factory_id, config_id, 'CONFIGURES_CONTAINER', RULE)
    derived_edge(graph, unit, root_class, config_id, root_id, 'CONTAINER_CONSTRUCTS_CLASS', RULE)
    stages = []
    previous_registration = None
    for index, count in enumerate(counts):
        stage_id = stable_id(unit.path, 'container-stage', config_id, index, count)
        graph.add_node({'id': stage_id, 'kind': 'container_stage',
                        'attributes': {'position': index, 'count': count, 'block_type': block_name,
                                       'scope': 'static_constructor_blueprint'},
                        'source_refs': [span(unit, stage_loop), span(unit, block_registration)],
                        'evidence_class': 'derived', 'analysis_rule': RULE, 'resolution': 'conditional'})
        derived_edge(graph, unit, stage_loop, config_id, stage_id, 'CONTAINER_HAS_STAGE', RULE)
        block_instance = stable_id(unit.path, 'container-block', stage_id, block_id)
        graph.add_node({'id': block_instance, 'kind': 'container_block_instance',
                        'attributes': {'position': index, 'block_type': block_name,
                                       'scope': 'static_constructor_blueprint'},
                        'source_refs': [span(unit, block_assign), span(unit, block_registration)],
                        'evidence_class': 'derived', 'analysis_rule': RULE, 'resolution': 'conditional'})
        derived_edge(graph, unit, block_registration, stage_id, block_instance, 'STAGE_REGISTERS_BLOCK', RULE)
        derived_edge(graph, unit, block_assign, block_instance, block_id, 'CONTAINER_INSTANCE_OF_CLASS', RULE)
        if previous_registration:
            derived_edge(graph, unit, block_registration, previous_registration,
                         block_instance, 'STATIC_MODULE_REGISTRATION_ORDER', RULE)
        member_ids = []
        last = None
        for position in range(count):
            element = stable_id(unit.path, 'container-member', block_instance, position, member_id)
            graph.add_node({'id': element, 'kind': 'container_member_instance',
                            'attributes': {'stage': index, 'position': position,
                                           'module_type': member_class, 'scope': 'static_constructor_blueprint'},
                            'source_refs': [span(unit, member_loop), span(unit, member_assign),
                                            span(unit, member_registration)],
                            'evidence_class': 'derived', 'analysis_rule': RULE, 'resolution': 'conditional'})
            derived_edge(graph, unit, member_registration, block_instance, element,
                         'CONTAINER_HAS_MEMBER', RULE)
            derived_edge(graph, unit, member_assign, element, member_id,
                         'CONTAINER_INSTANCE_OF_CLASS', RULE)
            if last:
                derived_edge(graph, unit, member_registration, last, element,
                             'STATIC_MODULE_REGISTRATION_ORDER', RULE)
            last = element
            member_ids.append(element)
        transition_id = None
        if index != len(counts) - 1:
            transition_id = stable_id(unit.path, 'container-transition', stage_id, trans_id)
            graph.add_node({'id': transition_id, 'kind': 'container_transition_instance',
                            'attributes': {'position': index, 'module_type': trans_name,
                                           'scope': 'static_constructor_blueprint'},
                            'source_refs': [span(unit, trans_assign), span(unit, trans_registration)],
                            'evidence_class': 'derived', 'analysis_rule': RULE, 'resolution': 'conditional'})
            derived_edge(graph, unit, trans_registration, stage_id, transition_id,
                         'CONTAINER_HAS_TRANSITION', RULE)
            derived_edge(graph, unit, trans_assign, transition_id, trans_id,
                         'CONTAINER_INSTANCE_OF_CLASS', RULE)
            derived_edge(graph, unit, trans_registration, block_instance, transition_id,
                         'STATIC_MODULE_REGISTRATION_ORDER', RULE)
        previous_registration = transition_id or block_instance
        stages.append({'stage_id': stage_id, 'position': index, 'count': count,
                       'block_instance_id': block_instance, 'member_instance_ids': member_ids,
                       'transition_instance_id': transition_id,
                       'source_refs': [span(unit, stage_loop)]})
    spec = {'factory_definition_id': factory_id, 'configuration_id': config_id,
            'model_definition_id': root_id, 'block_definition_id': block_id,
            'member_definition_id': member_id, 'transition_definition_id': trans_id,
            'stage_counts': list(counts), 'stages': stages, 'total_members': sum(counts),
            'block_type': block_name, 'member_type': member_class, 'transition_type': trans_name,
            'container_expression': container_expression,
            'scope': 'static_constructor_blueprint', 'resolution': 'conditional'}
    return root_id, spec
