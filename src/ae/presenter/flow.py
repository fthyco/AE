"""Source-backed static flow projection."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from ae.sae.ir.canonical import stable_id
from .health import issue
from .reader import Bundle


FLOW_EDGES = frozenset({'STATIC_VALUE_DEPENDENCY', 'STATIC_FORWARD_RETURN', 'STATIC_CONDITIONAL_DEPENDENCY', 'STATIC_GUARD_DEPENDENCY', 'STATIC_GUARDED_CALL', 'STATIC_COLLECTION_ELEMENT', 'STATIC_COLLECTION_MEMBER', 'STATIC_COLLECTION_ELEMENT_BINDING'})
FLOW_VALUES = frozenset({'forward_input', 'forward_output', 'forward_unpack', 'forward_choice', 'forward_guard', 'forward_augassign', 'forward_collection', 'forward_collection_element'})


def _caption(call: dict, components: dict[str, dict]) -> str:
    callee = call['callee_expression']
    constructions = call.get('module_construction_site_ids', [])
    if len(constructions) == 1 and constructions[0] in components:
        comp = components[constructions[0]]
        label = comp.get('owner_expression', callee).removeprefix('self.')
        return f"{label}: {(comp.get('declared_type') or 'Module').split('.')[-1]}"
    if callee.startswith('self.'):
        return callee[5:]
    return callee.replace('torch.nn.functional.', 'F.').replace('torch.', '')


def project_flow(bundle: Bundle, level: str = 'block', focus: str | None = None) -> dict[str, Any]:
    if level not in ('overview', 'block', 'operation'):
        raise ValueError(f'Unsupported flow level: {level}')
    index = bundle.model_index
    graph: dict[str, Any] = {'nodes': [], 'links': [], 'issues': [], 'level': level, 'view': 'flow', 'focus': focus}
    if index.get('status') != 'indexed':
        return graph
    modules = index.get('modules', [])
    if level == 'overview':
        modules = [m for m in modules if m['definition_id'] == index.get('root_definition_id')]
    else:
        modules = sorted(modules, key=lambda m: (m['definition_id'] != index.get('root_definition_id'), m['qualname']))

    if focus is not None:
        selected = [m for m in modules if m['qualname'] == focus or m['qualname'].split('.')[-1] == focus]
        if len(selected) != 1:
            graph['issues'].append(issue('FOCUS_UNRESOLVED',
                f'Expected exactly one module matching {focus!r}; found {len(selected)}', severity='error'))
            return graph
        modules = selected

    flow_by_forward: dict[str, list[dict]] = defaultdict(list)
    for n in bundle.ir['nodes']:
        if n['kind'] in FLOW_VALUES:
            flow_by_forward[n['attributes'].get('forward_id')].append(n)
    flow_edges = [e for e in bundle.ir['edges'] if e['relation'] in FLOW_EDGES]
    shape_facts = {n['attributes']['value_id']: n for n in bundle.ir['nodes']
                   if n['kind'] == 'symbolic_shape_fact'}
    shape_edges = {e['source']:e['id'] for e in bundle.ir['edges']
                   if e['relation'] == 'HAS_SYMBOLIC_SHAPE'}
    guarded = {e['target'] for e in bundle.ir['edges'] if e['relation'] == 'STATIC_GUARDED_CALL'}
    global_mapped: dict[str, str] = {}
    mapped_lane: dict[str, str] = {}
    for module in modules:
        fid = module.get('forward_method_id')
        if not fid:
            graph['issues'].append(issue('NO_FORWARD', f"No unique forward method for {module['qualname']}",
                                         ids=[module['definition_id']], refs=module['source_refs']))
            continue
        definition = module['definition_id']
        root_id = stable_id('flow-lane/v1', definition)
        graph['nodes'].append({'id': root_id, 'kind': 'flow-module', 'parent_id': None,
                               'label': module['qualname'].split('.')[-1], 'details': module['qualname'],
                               'is_root': definition == index.get('root_definition_id'),
                               'ir_node_ids': [definition], 'ir_edge_ids': [],
                               'source_refs': module['source_refs'], 'evidence': 'explicit'})
        comps = {c['construction_site_id']: c for c in module['components']}
        steps: list[tuple[str, str, str, str]] = []
        # tuple: kind, label, source IR ID, resolution
        for value in flow_by_forward.get(fid, []):
            steps.append((value['kind'], value['attributes']['name'], value['id'], value['resolution']))
        for call in module.get('forward_calls', []):
            sid = call['call_site_id']
            unknown = call['callee_expression'].startswith('self.') and not (
                call.get('module_construction_site_ids') or call.get('local_definition_ids'))
            if unknown:
                graph['issues'].append(issue('UNRESOLVED_FORWARD_CALL',
                    f"Unresolved forward call: {call['callee_expression']}", ids=[sid], refs=call['source_refs']))
            steps.append(('unresolved-call' if unknown else 'flow-call', _caption(call, comps), sid,
                          'unresolved' if unknown else 'conditional' if sid in guarded else 'resolved'))
        flow_members = {e['source'] for e in flow_edges} | {e['target'] for e in flow_edges}
        for op in module.get('forward_operations', []):
            if op['operation_site_id'] in flow_members:
                steps.append(('flow-operation', op['operator'], op['operation_site_id'], 'resolved'))

        # Retain source IR identities.
        mapped: dict[str, str] = {}
        for kind, label, source_id, state in sorted(steps, key=lambda x: (
                bundle.nodes[x[2]]['source_refs'][0]['start_byte'], x[2])):
            if kind == 'forward_unpack' and level == 'block':
                # Keep unpacking and fan-out links.
                pass
            vid = stable_id('flow-node/v1', definition, source_id)
            if source_id in mapped:
                continue
            mapped[source_id] = vid
            global_mapped[source_id] = vid
            mapped_lane[source_id] = root_id
            source = bundle.nodes[source_id]
            fact = shape_facts.get(source_id)
            shape = '[' + ', '.join(fact['attributes']['dimensions']) + ']' if fact else None
            graph['nodes'].append({'id': vid, 'kind': kind, 'parent_id': root_id,
                                   'label': label, 'details': source.get('attributes', {}).get('name', ''),
                                   'shape': shape,
                                   'ir_node_ids': [source_id] + ([fact['id']] if fact else []),
                                   'ir_edge_ids': [shape_edges[source_id]] if fact and source_id in shape_edges else [],
                                   'source_refs': source['source_refs'],
                                   'evidence': 'conditional' if state == 'conditional' else
                                               'unresolved' if state == 'unresolved' else 'explicit'})
        for edge in flow_edges:
            if edge['source'] not in mapped or edge['target'] not in mapped:
                continue
            visual_id = stable_id('flow-link/v1', definition, edge['id'])
            graph['links'].append({'id': visual_id, 'source': mapped[edge['source']],
                                   'target': mapped[edge['target']],
                                   'relation_kind': edge['relation'], 'ir_edge_ids': [edge['id']],
                                   'source_refs': edge['source_refs'], 'style': 'control' if edge['relation']=='STATIC_GUARDED_CALL' else 'conditional-value' if edge['relation'] in ('STATIC_CONDITIONAL_DEPENDENCY','STATIC_GUARD_DEPENDENCY') else 'static-value'})
        for record in bundle.ir['unresolved']:
            if record.get('kind') == 'forward_flow_incomplete' and record.get('function_id') == fid:
                graph['issues'].append(issue('STATIC_FLOW_INCOMPLETE', record.get('reason', 'Unknown flow'),
                                            refs=record.get('source_refs', [])))
        for record in bundle.ir['unresolved']:
            if record.get('function_id') != fid:
                continue
            if record.get('kind') in ('interprocedural_flow_incomplete', 'symbolic_shape_conflict'):
                graph['issues'].append(issue(record['kind'].upper(), record.get('reason', 'Unresolved semantic relationship'),
                                            ids=[record['site_id']] if record.get('site_id') else [],
                                            refs=record.get('source_refs', [])))
        if not any(n['kind'] == 'forward_output' and n['parent_id'] == root_id for n in graph['nodes']):
            graph['issues'].append(issue('STATIC_FLOW_INCOMPLETE',
                                         f"No straight-line return represented in {module['qualname']}",
                                         ids=[fid], refs=bundle.nodes[fid]['source_refs']))
        if not any(e['source'] in mapped and e['target'] in mapped for e in flow_edges):
            graph['issues'].append(issue('STATIC_FLOW_INCOMPLETE',
                                         f"No source-backed value dependency in {module['qualname']}",
                                         ids=[fid], refs=bundle.nodes[fid]['source_refs']))

    # Draw links between visible module lanes.
    for edge in flow_edges:
        if (edge['relation'] != 'STATIC_COLLECTION_ELEMENT_BINDING'
                or edge['source'] not in global_mapped
                or edge['target'] not in global_mapped
                or mapped_lane[edge['source']] == mapped_lane[edge['target']]):
            continue
        graph['links'].append({
            'id': stable_id('flow-cross-module/v1', edge['id']),
            'source': global_mapped[edge['source']],
            'target': global_mapped[edge['target']],
            'relation_kind': edge['relation'], 'ir_edge_ids': [edge['id']],
            'source_refs': edge['source_refs'], 'style': 'interprocedural'})
    if not graph['nodes']:
        graph['issues'].append(issue('NO_STATIC_FLOW_EVIDENCE' , 'No usable static forward flow. Re-extract with a compatible SAE.', severity='error'))
    graph['nodes'].sort(key=lambda n: (n['parent_id'] or '', n['source_refs'][0]['path'],
                                       n['source_refs'][0]['start_byte'], n['id']))
    graph['links'].sort(key=lambda e: e['id'])
    graph['issues'].sort(key=lambda x: (x['code'], x['message']))
    return graph
