"""Deterministic left-to-right DAG layout for source-grounded forward-value graphs."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

STEP_W, STEP_H, DX, DY = 152, 56, 210, 88
MARGIN, TOP = 42, 108


def make_flow_scene(graph: dict[str, Any], *, status: str, entrypoint: str | None) -> dict[str, Any]:
    all_nodes = {n['id']: n for n in graph['nodes']}
    lanes = [n for n in graph['nodes'] if n['kind'] == 'flow-module']
    lanes.sort(key=lambda n: (not n.get('is_root', False), n['details'], n['id']))
    rects: list[dict] = []
    texts: list[dict] = []
    paths: list[dict] = []
    positions: dict[str, dict] = {}
    texts.append({'x': MARGIN, 'y': 34, 'text': (entrypoint or 'Model').replace(':', ' / '), 'size': 21, 'bold': True})
    texts.append({'x': MARGIN, 'y': 56, 'text': 'STATIC FORWARD VALUE DEPENDENCIES   |   source backed', 'size': 10, 'bold': False})
    curr_y = TOP
    max_x = 650

    for lane in lanes:
        members = [n for n in graph['nodes'] if n['parent_id'] == lane['id']]
        local = {n['id']: n for n in members}
        edges = [e for e in graph['links'] if e['source'] in local and e['target'] in local]
        src_order = lambda n: (n['source_refs'][0]['path'], n['source_refs'][0]['start_byte'], n['id'])
        members.sort(key=src_order)
        # Kahn ranks: every proven edge pushes the consumer to the right.
        indeg = {n['id']: 0 for n in members}
        adj: dict[str, list[str]] = defaultdict(list)
        for e in edges:
            if e['source'] == e['target']:
                continue
            indeg[e['target']] += 1
            adj[e['source']].append(e['target'])
        q = deque(sorted((x for x, degree in indeg.items() if not degree), key=lambda x: src_order(local[x])))
        rank = {n['id']: 0 for n in members}
        used = 0
        while q:
            current = q.popleft(); used += 1
            for target in sorted(adj[current], key=lambda x: src_order(local[x])):
                rank[target] = max(rank[target], rank[current] + 1)
                indeg[target] -= 1
                if indeg[target] == 0:
                    q.append(target)
        if used != len(members):
            # SSA-free source analysis must never produce cycles as verified
            # execution flow; fall back to source order with a visible warning.
            graph['issues'].append({'code': 'STATIC_FLOW_CYCLE', 'severity': 'warning',
                                    'message': f'Unexpected dependency cycle in {lane["label"]}',
                                    'related_ir_ids': lane['ir_node_ids'],
                                    'source_refs': lane['source_refs'], 'affects_view': True})
            rank = {n['id']: i for i, n in enumerate(members)}
        columns: dict[int, list[str]] = defaultdict(list)
        for n in members:
            columns[rank[n['id']]].append(n['id'])
        for values in columns.values():
            values.sort(key=lambda x: src_order(local[x]))
        lane_h = max(130, 68 + max((len(v) for v in columns.values()), default=1) * DY + 35)
        lane_w = max(410, 70 + (max(columns, default=0) + 1) * DX)
        rects.append({'id': lane['id'], 'ir_ids': lane['ir_node_ids'],
                      'x': MARGIN, 'y': curr_y, 'w': lane_w, 'h': lane_h,
                      'kind': 'flow-module', 'fill': '#ffffff', 'dash': 'none'})
        rects.append({'x': MARGIN, 'y': curr_y, 'w': lane_w, 'h': 37,
                      'kind': 'header', 'fill': '#e9e9e9', 'dash': 'none'})
        texts.append({'x': MARGIN+14, 'y': curr_y+25, 'text': lane['label'], 'size': 14, 'bold': True})
        for column in sorted(columns):
            for row, ident in enumerate(columns[column]):
                node = local[ident]
                pos = {'x': MARGIN+26+column*DX,
                       'y': curr_y+57+row*DY, 'w': STEP_W, 'h': STEP_H}
                positions[ident] = pos
                unresolved = node['evidence'] in ('conditional','unresolved')
                rects.append({'id': ident, 'ir_ids': node['ir_node_ids'], **pos,
                              'kind': node['kind'], 'fill': '#f5f5f5' if node['kind'] in ('flow-call', 'flow-operation') else '#ffffff',
                              'dash': '5,4' if unresolved else 'none'})
                label = node['label']
                if len(label)>23:
                    label=label[:22]+'…'
                texts.append({'x': pos['x']+10, 'y': pos['y']+24, 'text': label, 'size': 11, 'bold': node['kind'] in ('forward_input','forward_output')})
                texts.append({'x': pos['x']+10, 'y': pos['y']+44, 'text':
                              'conditional' if node['evidence']=='conditional' else node['kind'].replace('flow-', '').replace('forward_', ''),
                              'size': 9, 'bold': False})
        # Value-dependency arrows are directed only by proven source relations.
        for e in edges:
            if e['source'] not in positions or e['target'] not in positions:
                continue
            a, b = positions[e['source']], positions[e['target']]
            x1, y1 = a['x']+a['w'], a['y']+a['h']/2
            x2, y2 = b['x'], b['y']+b['h']/2
            p_kind = e['style']
            curve = {'id': e['id'], 'ir_edge_ids': e['ir_edge_ids'],
                     'kind': e['style'], 'label': e['relation_kind'],
                     'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            if rank[e['target']] > rank[e['source']]+1:
                # A skip path is routed above the blocks to avoid masquerading
                # as an intermediate linear connection.
                top = (curr_y+lane_h-23 if p_kind != 'static-value' else
                       curr_y+48 - 15*(rank[e['target']]-rank[e['source']]-2))
                curve.update({'cx1': x1+35, 'cy1': top, 'cx2': x2-35, 'cy2': top})
            else:
                curve.update({'cx1': x1+34, 'cy1': y1, 'cx2': x2-34, 'cy2': y2})
            paths.append(curve)
        curr_y += lane_h+26
        max_x = max(max_x, MARGIN+lane_w+35)
    height = int(curr_y+50)
    texts.append({'x':MARGIN,'y':height-43,'text':
                  'Solid = source value dependency; dashed = conditional value; dotted = conditional call',
                  'size':10,'bold':False})
    if status == 'partial':
        texts.append({'x':MARGIN,'y':height-25,'text':'PARTIAL / NOT FULLY VERIFIED: see presentation-health.json',
                      'size':10,'bold':True})
    else:
        texts.append({'x':MARGIN,'y':height-25,'text':'Scope: static forward expressions and conditional alternatives only',
                      'size':10,'bold':False})
    paths.sort(key=lambda p: (p['y1'],p['x1'],p['y2'],p['x2'],p['id']))
    return {'width':int(max_x),'height':height,'rects':rects,'texts':texts,
            'paths':paths,'title':entrypoint or 'Static Forward Architecture',
            'status':status,'positions':positions}
