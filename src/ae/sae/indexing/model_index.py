"""Entrypoint-scoped architecture inventory backed exclusively by SAE IR evidence.

This is an index, *not* an inferred runtime execution graph or a visualization.
It never evaluates target Python and never discards nodes from the full IR.
"""
from __future__ import annotations

import ast
from typing import Any

from ..ir.canonical import stable_id
from ..ir.models import Graph
from ..linking.project import Unit, derived_edge, span, syntax_ids

RULE = 'model-architecture-index/v1'


def _class_id(unit: Unit, klass: ast.ClassDef) -> str:
    return stable_id(unit.path, 'semantic', 'class_definition', syntax_ids(unit)[klass], '')


def _method_id(unit: Unit, method: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    return stable_id(unit.path, 'semantic', 'method_definition', syntax_ids(unit)[method], '')


def _is_self_target(target: ast.AST) -> bool:
    return isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self'


def _walk_body(node: ast.AST):
    """Walk statements without claiming that a nested function is executed."""
    def walk(current: ast.AST):
        yield current
        for child in ast.iter_child_nodes(current):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            yield from walk(child)
    for stmt in getattr(node, 'body', ()):
        yield from walk(stmt)


def build_model_index(graph: Graph, units: list[Unit], entrypoint_id: str | None,
                      requested_entrypoint: str | None) -> dict[str, Any]:
    """Build deterministic, source-grounded references to architectural entities.

    IR edges and already linked calls determine the associations. Syntactic
    ordering is exposed as lexical order, never runtime execution order.
    """
    result: dict[str, Any] = {
        'schema_version': '1.0', 'entrypoint': requested_entrypoint,
        'root_definition_id': entrypoint_id, 'status': 'no_entrypoint' if not requested_entrypoint else 'unresolved',
        'modules': [], 'unresolved': [],
        'limitations': [
            'Lexical call order is not execution order',
            'Module construction sites do not prove runtime instantiation',
            'Only uniquely linked local module definitions are recursively indexed',
            'Functional call names are source expressions, not resolved function identities',
        ],
    }
    if entrypoint_id is None:
        if requested_entrypoint:
            result['unresolved'].append({'kind': 'model_entrypoint', 'reason': 'No uniquely resolved entrypoint'})
        return result

    definitions: dict[str, tuple[Unit, ast.ClassDef]] = {}
    for unit in units:
        for statement in unit.tree.body:
            if isinstance(statement, ast.ClassDef):
                definitions[_class_id(unit, statement)] = (unit, statement)
    if entrypoint_id not in definitions:
        result['status'] = 'unsupported_entrypoint'
        result['unresolved'].append({'kind': 'model_entrypoint', 'reason': 'Entrypoint is not a top-level class'})
        return result

    nodes = {node['id']: node for node in graph.nodes}
    edge_list = list(graph.edges)  # first pass records pre-existing resolved links
    links: dict[tuple[str, str], list[str]] = {}
    for e in edge_list:
        links.setdefault((e['source'], e['relation']), []).append(e['target'])
    for targets in links.values():
        targets.sort()

    def semantic_for_ast(unit: Unit, node: ast.AST, kind: str, ids: dict) -> str:
        return stable_id(unit.path, 'semantic', kind, ids[node], '')

    seen: set[str] = set()
    active: set[str] = set()

    def visit(definition_id: str, *, via: str | None = None) -> None:
        if definition_id in active:
            result['unresolved'].append({'kind': 'recursive_module_reference',
                                         'definition_id': definition_id, 'via_site_id': via})
            return
        if definition_id in seen:
            return
        if definition_id not in definitions:
            result['unresolved'].append({'kind': 'external_module_definition',
                                         'definition_id': definition_id, 'via_site_id': via})
            return
        active.add(definition_id)
        unit, klass = definitions[definition_id]
        ids = syntax_ids(unit)
        # A module's constructor and forward are lexically identified, not assumed executed.
        initializers = [n for n in klass.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == '__init__']
        forwards = [n for n in klass.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == 'forward']
        forward_id = _method_id(unit, forwards[0]) if len(forwards) == 1 else None
        if len(forwards) != 1:
            result['unresolved'].append({'kind': 'forward_resolution', 'definition_id': definition_id,
                                         'reason': f'Expected one declared forward, found {len(forwards)}'})
        else:
            derived_edge(graph, unit, forwards[0], definition_id, forward_id, 'DECLARES_FORWARD', RULE)

        components: list[dict[str, Any]] = []
        dependencies: set[str] = set()
        for init in initializers:
            for n in _walk_body(init):
                if not isinstance(n, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for target in targets:
                    if not _is_self_target(target):
                        continue
                    value = n.value
                    if not isinstance(value, ast.Call):
                        continue
                    call_id = semantic_for_ast(unit, value, 'call_site', ids)
                    sites = [e['id'] for e in graph.nodes
                             if e['kind'] == 'module_construction_site'
                             and e['source_refs'][0]['path'] == unit.path
                             and e['source_refs'][0]['start_byte'] == span(unit, value)['start_byte']
                             and e['source_refs'][0]['end_byte'] == span(unit, value)['end_byte']]
                    if len(sites) > 1:
                        # Existing semantics cannot prove which construction event is intended.
                        result['unresolved'].append({'kind': 'ambiguous_component_construction',
                              'owner_definition_id': definition_id, 'assignment_ast_id': ids[target],
                              'candidate_ids': sorted(sites)})
                        continue
                    if not sites:
                        result['unresolved'].append({'kind': 'unrecognized_component_assignment',
                              'owner_definition_id': definition_id, 'assignment_ast_id': ids[target],
                              'expression': ast.unparse(value), 'source_refs': [span(unit, value)]})
                        continue
                    site_id = sites[0]
                    target_ids = links.get((site_id, 'CONSTRUCTS_LOCAL_MODULE'), [])
                    unique_target = target_ids[0] if len(target_ids) == 1 else None
                    layer_specs = [e['source'] for e in edge_list
                                   if e['relation'] == 'DESCRIBES_PYTORCH_CALL'
                                   and e['target'] == call_id]
                    record = {'construction_site_id': site_id,
                              'layer_spec_ids': sorted(layer_specs),
                              'owner_expression': ast.unparse(target),
                              'declared_type': nodes[site_id]['attributes'].get('declared_type'),
                              'target_definition_id': unique_target,
                              'source_refs': [span(unit, value)]}
                    components.append(record)
                    derived_edge(graph, unit, value, definition_id, site_id,
                                 'DECLARES_COMPONENT', RULE)
                    if unique_target is not None:
                        dependencies.add(unique_target)
                    if len(target_ids) > 1:
                        result['unresolved'].append({'kind': 'ambiguous_component_target',
                           'construction_site_id': site_id, 'candidate_ids': target_ids})

        # Preserve source-order call sites and operations in every directly defined forward.
        calls: list[dict[str, Any]] = []
        operations: list[dict[str, Any]] = []
        forward_relations: list[str] = []
        if forward_id is not None:
            for node in _walk_body(forwards[0]):
                if isinstance(node, ast.Call):
                    call_id = semantic_for_ast(unit, node, 'call_site', ids)
                    if call_id not in nodes:
                        continue
                    targets = links.get((call_id, 'USES_MODULE_REFERENCE'), [])
                    local_targets = links.get((call_id, 'CALLS_LOCAL_DEFINITION'), [])
                    calls.append({'call_site_id': call_id,
                                  'callee_expression': nodes[call_id]['attributes']['callee_expression'],
                                  'module_construction_site_ids': targets,
                                  'local_definition_ids': local_targets,
                                  'source_refs': [span(unit, node)]})
                    derived_edge(graph, unit, node, forward_id, call_id, 'HAS_FORWARD_CALL', RULE)
                elif isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Subscript)):
                    op_id = semantic_for_ast(unit, node, 'operation_site', ids)
                    if op_id in nodes:
                        operations.append({'operation_site_id': op_id, 'operator': nodes[op_id]['attributes']['operator'],
                                           'source_refs': [span(unit, node)]})
                        derived_edge(graph, unit, node, forward_id, op_id, 'HAS_FORWARD_OPERATION', RULE)

        if forward_id is not None:
            member_syntax = {ids[n] for n in ast.walk(forwards[0])}
            member_semantic = {e['source'] for e in edge_list
                               if e['relation'] == 'APPEARS_AT' and e['target'] in member_syntax}
            members = member_syntax | member_semantic
            relations = {'EXPRESSION_DEPENDS_ON', 'VALUE_EXPRESSION',
                         'READS_LEXICAL_DEFINITION', 'RETURNS_EXPRESSION',
                         'HAS_ARGUMENT', 'HAS_KEYWORD', 'CALLEE_EXPRESSION',
                         'CONDITION_EXPRESSION', 'HAS_BRANCH_STATEMENT'}
            forward_relations = sorted(e['id'] for e in edge_list
                                       if e['relation'] in relations
                                       and e['source'] in members and e['target'] in members)

        def sort_by_source(record: dict[str, Any]):
            ref = record['source_refs'][0]
            return (ref['path'], ref['start_byte'], ref['end_byte'])
        components.sort(key=sort_by_source)
        calls.sort(key=sort_by_source)
        operations.sort(key=sort_by_source)
        module = {'definition_id': definition_id, 'qualname': f'{unit.module}.{klass.name}',
                  'source_refs': [span(unit, klass)], 'forward_method_id': forward_id,
                  'components': components, 'forward_calls': calls,
                  'forward_operations': operations,
                  'forward_relation_ids': forward_relations}
        result['modules'].append(module)
        seen.add(definition_id)
        for child in sorted(dependencies):
            visit(child, via=next(c['construction_site_id'] for c in components if c['target_definition_id'] == child))
        active.remove(definition_id)

    visit(entrypoint_id)
    result['status'] = 'indexed' if result['modules'] else 'unresolved'
    result['modules'].sort(key=lambda m: (m['qualname'], m['definition_id']))
    result['unresolved'].sort(key=lambda x: (x['kind'], x.get('definition_id', ''),
                                            x.get('owner_definition_id', ''), x.get('assignment_ast_id', ''),
                                            x.get('construction_site_id', '')))
    return result


def validate_model_index(index: dict[str, Any], ir: dict[str, Any]) -> None:
    """Guard Presenter handoff from missing/invalid IR references."""
    if index.get('schema_version') != '1.0':
        raise ValueError('Unsupported model index version')
    if index.get('root_definition_id') != ir.get('entrypoint_target_id'):
        raise ValueError('Model index entrypoint differs from IR')
    nodes = {n['id']: n for n in ir['nodes']}
    def check(reference: str | None, kind: str | None = None):
        if reference is None:
            return
        if reference not in nodes:
            raise ValueError(f'Model index dangling reference: {reference}')
        if kind and nodes[reference]['kind'] != kind:
            raise ValueError(f'Model index unexpected node kind: {reference}')
    check(index.get('root_definition_id'))
    factory = index.get('factory_specialization')
    if factory:
        check(factory['factory_definition_id'], 'function_definition')
        check(factory['factory_configuration_id'], 'factory_configuration')
        check(factory['model_definition_id'], 'class_definition')
        check(factory['block_definition_id'], 'class_definition')
        if factory['model_definition_id'] != index.get('root_definition_id'):
            raise ValueError('Factory model definition differs from index root')
        if sum(factory['stage_counts']) != factory['total_blocks']:
            raise ValueError('Factory block count mismatch')
        if len(factory['stages']) != len(factory['stage_counts']):
            raise ValueError('Factory stage count mismatch')
        for i, stage in enumerate(factory['stages']):
            check(stage['stage_id'], 'factory_stage')
            check(stage['block_definition_id'], 'class_definition')
            if stage['position'] != i or stage['count'] != factory['stage_counts'][i]:
                raise ValueError('Factory stage order mismatch')
            if len(stage['block_instance_ids']) != stage['count']:
                raise ValueError('Factory instance count mismatch')
            for ident in stage['block_instance_ids']:
                check(ident, 'factory_block_instance')
    for module in index.get('modules', []):
        check(module['definition_id'], 'class_definition')
        check(module['forward_method_id'], 'method_definition')
        for comp in module['components']:
            check(comp['construction_site_id'], 'module_construction_site')
            check(comp['target_definition_id'], 'class_definition')
            for spec_id in comp['layer_spec_ids']:
                check(spec_id, 'pytorch_layer_spec')
        for call in module['forward_calls']:
            check(call['call_site_id'], 'call_site')
            for ident in call['module_construction_site_ids']:
                check(ident, 'module_construction_site')
            for ident in call['local_definition_ids']:
                check(ident)
        for op in module['forward_operations']:
            check(op['operation_site_id'], 'operation_site')
        edge_ids = {e['id'] for e in ir['edges']}
        for relation_id in module['forward_relation_ids']:
            if relation_id not in edge_ids:
                raise ValueError(f'Model index dangling relation: {relation_id}')
