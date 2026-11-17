"""Conservative model-centric visual projection. No source parsing or tensor-flow claims."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from ae.sae.ir.canonical import stable_id
from .health import issue
from .reader import Bundle


def project(bundle: Bundle, level: str = 'block') -> dict[str, Any]:
    if level not in ('overview', 'block', 'operation'):
        raise ValueError(f'Unsupported level: {level}')
    index, ir = bundle.model_index, bundle.ir
    modules = {m['definition_id']: m for m in index.get('modules', [])}
    graph: dict[str, Any] = {'nodes': [], 'links': [], 'issues': [], 'level': level}
    if index.get('status') != 'indexed' or index.get('root_definition_id') not in modules:
        return graph
    ir_edges_by_tuple: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for e in ir['edges']:
        ir_edges_by_tuple[e['source'], e['target'], e['relation']].append(e)
    visual_ids: set[str] = set()
    link_ids: set[str] = set()
    graph_node_by_id: dict[str, dict] = {}

    def add(kind: str, label: str, path: tuple, ir_ids: list[str], *,
            parent: str | None = None, edge_ids: list[str] | None = None,
            state: str = 'explicit', details: str = '') -> str:
        ids = sorted(set(ir_ids))
        if not ids or any(i not in bundle.nodes for i in ids):
            raise ValueError('Visual node missing actual IR source identity')
        refs = [span for i in ids for span in bundle.nodes[i].get('source_refs', [])]
        if not refs:
            raise ValueError('Visual node missing source refs')
        ident = stable_id('presenter-node/v1', kind, path, ids)
        if ident in visual_ids:
            return ident
        visual_ids.add(ident)
        entry = {'id': ident, 'kind': kind, 'label': label, 'parent_id': parent,
                 'ir_node_ids': ids, 'ir_edge_ids': sorted(set(edge_ids or [])),
                 'evidence': state, 'source_refs': refs, 'details': details}
        graph['nodes'].append(entry)
        graph_node_by_id[ident] = entry
        return ident

    def edges_between(src: str, target: str, relation: str) -> list[str]:
        return [e['id'] for e in ir_edges_by_tuple.get((src, target, relation), [])]

    def connect(source: str, target: str, relation: str, backing_edges: list[str], *,
                style: str = 'reference') -> None:
        if not backing_edges:
            return  # No evidence -> no visual edge
        for eid in backing_edges:
            e = bundle.edges.get(eid)
            if not e or not e.get('source_refs'):
                raise ValueError('Visual link missing proven IR edge and provenance')
        ident = stable_id('presenter-link/v1', source, target, relation, sorted(backing_edges))
        if ident in link_ids:
            return
        link_ids.add(ident)
        graph['links'].append({'id': ident, 'source': source, 'target': target,
                               'relation_kind': relation, 'ir_edge_ids': sorted(set(backing_edges)),
                               'source_refs': [s for eid in backing_edges for s in bundle.edges[eid]['source_refs']],
                               'style': style})

    # A definition has one visual identity per statically indexed *construction path*.
    # This avoids collapsing distinct instances of the same custom module definition.
    def render_module(definition: str, parent: str | None, path: tuple, active: frozenset[str],
                      via_comp: str | None = None, via_site: str | None = None) -> str:
        m = modules[definition]
        label = m['qualname'].split('.')[-1]
        mod_id = add('model' if parent is None else 'module', label, path, [definition], parent=parent,
                     details=m['qualname'])
        if parent and via_comp and via_site:
            connect(via_comp, mod_id, 'CONSTRUCTS_LOCAL_MODULE',
                    edges_between(via_site, definition, 'CONSTRUCTS_LOCAL_MODULE'))
        factory = index.get('factory_specialization')
        if parent is None and factory and factory['model_definition_id'] == definition:
            cfg = factory['factory_configuration_id']
            cv = add('factory', f"{index['entrypoint'].split(':')[-1]}: {factory['block_type']}",
                     (*path, 'factory', cfg), [cfg], parent=mod_id, state='conditional',
                     details='Static constructor blueprint')
            connect(cv, mod_id, 'FACTORY_CONSTRUCTS_CLASS',
                    edges_between(cfg, definition, 'FACTORY_CONSTRUCTS_CLASS'))
            for stage in factory['stages']:
                sid = stage['stage_id']
                sv = add('factory-stage', f"{stage['name']}: {stage['count']} × {stage['block_type']}",
                         (*path, 'stage', sid), [sid], parent=cv, state='conditional',
                         details=f"Stage {stage['position'] + 1}")
                connect(cv, sv, 'FACTORY_HAS_STAGE', edges_between(cfg, sid, 'FACTORY_HAS_STAGE'))
                if level != 'overview':
                    for i, instance_id in enumerate(stage['block_instance_ids']):
                        iv = add('factory-block', f"{stage['block_type']} [{i}]",
                                 (*path, 'stage', sid, 'block', instance_id), [instance_id],
                                 parent=sv, state='conditional', details=f"{stage['name']}.{i}")
                        connect(sv, iv, 'STAGE_HAS_BLOCK_INSTANCE',
                                edges_between(sid, instance_id, 'STAGE_HAS_BLOCK_INSTANCE'))
        comp_views: dict[str, str] = {}
        for comp in sorted(m['components'], key=lambda c:(c['source_refs'][0]['start_byte'], c['construction_site_id'])):
            cid = comp['construction_site_id']
            declared_type = comp['declared_type'] or 'Unknown'
            label_ = f"{comp['owner_expression'].replace('self.', '')}: {declared_type.split('.')[-1]}"
            arg_node = bundle.nodes[cid]
            args = arg_node.get('attributes',{}).get('constructor_arguments',[])
            arg_str = ', '.join((str(a.get('value')) if a.get('kind') == 'literal' else a.get('name') or a.get('text') or '?') if isinstance(a, dict) else str(a) for a in args)
            cc = add('component', label_, (*path,'component',cid), [cid], parent=mod_id,
                     details=f'{declared_type}({arg_str})')
            comp_views[cid] = cc
            connect(mod_id, cc, 'DECLARES_COMPONENT', edges_between(definition,cid,'DECLARES_COMPONENT'))
            child = comp.get('target_definition_id')
            if level != 'overview' and child and child in modules:
                if child in active:
                    graph['issues'].append(issue('UNRESOLVED_CONNECTION',
                         f'Recursive module expansion was stopped at {label_}', ids=[cid],refs=comp.get('source_refs',[])))
                else:
                    render_module(child, cc, (*path,'module-instance',cid), active | {child}, cc, cid)
            # Optional nested constructor facts for nn.Sequential and similar containers.
            if level == 'operation':
                for layer_spec_id in comp.get('layer_spec_ids', []):
                    nested = sorted((e for e in ir['edges']
                                     if e['relation'] == 'CONTAINS_NESTED_CONSTRUCTOR' and e['source'] == layer_spec_id),
                                     key=lambda e:e['target'])
                    for e in nested:
                        spec = bundle.nodes[e['target']]
                        sublabel = spec.get('attributes',{}).get('declared_type','component').split('.')[-1]
                        nn = add('nested-constructor', sublabel, (*path,'nested',cid,e['target']),
                                 [e['target']], parent=cc, edge_ids=[e['id']])
                        connect(cc, nn, 'CONTAINS_NESTED_CONSTRUCTOR', [e['id']])
        if level != 'overview':
            forward_id = m.get('forward_method_id')
            if not forward_id:
                graph['issues'].append(issue('UNRESOLVED_CONNECTION',
                    f'Missing unique forward method: {m["qualname"]}', ids=[definition],refs=m['source_refs']))
            call_views: dict[str, str] = {}
            operation_views: dict[str, str] = {}
            for call in sorted(m['forward_calls'],key=lambda x:(x['source_refs'][0]['start_byte'],x['call_site_id'])):
                csid = call['call_site_id']; callee = call['callee_expression']
                unknown = callee.startswith('self.') and not call['module_construction_site_ids'] and not call['local_definition_ids']
                vi = add('unresolved-call' if unknown else 'call',
                         f'{callee}()' + (' [?]' if unknown else ''), (*path,'call',csid), [csid],
                         parent=mod_id, state='unresolved' if unknown else 'explicit')
                call_views[csid] = vi
                if forward_id:
                    connect(mod_id, vi, 'HAS_FORWARD_CALL',edges_between(forward_id,csid,'HAS_FORWARD_CALL'))
                for construction in call['module_construction_site_ids']:
                    if construction in comp_views:
                        connect(vi, comp_views[construction], 'USES_MODULE_REFERENCE',
                                edges_between(csid,construction,'USES_MODULE_REFERENCE'))
                if unknown:
                    graph['issues'].append(issue('MISSING_LOCAL_OR_EXTERNAL_DEFINITION',
                        f'Cannot resolve call {callee} in {m["qualname"]}',ids=[csid],refs=call['source_refs']))
            if level == 'operation' or level == 'block':
                for op in sorted(m['forward_operations'],key=lambda x:(x['source_refs'][0]['start_byte'],x['operation_site_id'])):
                    opid=op['operation_site_id']
                    vo=add('operation',op['operator'],(*path,'operation',opid),[opid],parent=mod_id)
                    operation_views[opid]=vo
                    if forward_id:
                        connect(mod_id, vo,'HAS_FORWARD_OPERATION',edges_between(forward_id,opid,'HAS_FORWARD_OPERATION'))
            if level == 'operation':
                active_views = {**call_views,**operation_views}
                for rid in m['forward_relation_ids']:
                    e=bundle.edges[rid]
                    if e['relation']=='EXPRESSION_DEPENDS_ON' and e['source'] in active_views and e['target'] in active_views:
                        connect(active_views[e['source']],active_views[e['target']],
                                'EXPRESSION_DEPENDS_ON',[rid],style='syntax-dependency')
        # Model-index unresolved assignments are not silently interpreted as components.
        for u in index.get('unresolved',[]):
            if u.get('owner_definition_id') != definition:
                continue
            astid=u.get('assignment_ast_id')
            if astid and astid in bundle.nodes:
                add('unresolved-component',f"Unresolved: {u.get('expression','component')}",
                    (*path,'unresolved',astid),[astid],parent=mod_id,state='unresolved')
            graph['issues'].append(issue('MISSING_LOCAL_OR_EXTERNAL_DEFINITION',
                     f"Unresolved model component: {u.get('expression',u['kind'])}",
                     ids=[astid] if astid in bundle.nodes else [],refs=u.get('source_refs',[])))
        return mod_id

    root=index['root_definition_id']
    render_module(root,None,('root',root),frozenset({root}))
    graph['nodes'].sort(key=lambda n:(n['parent_id'] or '',n['kind'],n['id']))
    graph['links'].sort(key=lambda e:(e['relation_kind'],e['source'],e['target'],e['id']))
    graph['issues'].sort(key=lambda x:(x['code'],x['message']))
    return graph
