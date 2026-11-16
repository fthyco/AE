"""Deterministic, source-grounded layered layout with orthogonal edge channels.

No edge is invented here: each route corresponds to an existing IR-supported
link. The visual style never changes extraction or health decisions.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from .themes import select_theme

STEP_W, STEP_H, DX, DY = 188, 56, 244, 76
MARGIN, TOP, HEADER_H = 44, 105, 38


def make_flow_scene(graph: dict[str, Any], *, status: str, entrypoint: str | None,
                    theme: str = 'scientific-latex') -> dict[str, Any]:
    style = select_theme(theme)
    lanes = sorted((n for n in graph['nodes'] if n['kind'] == 'flow-module'),
                   key=lambda n: (not n.get('is_root', False), n['details'], n['id']))
    rects: list[dict] = []
    texts: list[dict] = []
    paths: list[dict] = []
    positions: dict[str, dict] = {}
    texts.append({'x': MARGIN, 'y': 32, 'text': (entrypoint or 'Model').replace(':', ' / '),
                  'size': 18, 'bold': True})
    texts.append({'x': MARGIN, 'y': 54, 'text': 'STATIC FORWARD VALUE DEPENDENCIES  |  Source-backed',
                  'size': 10, 'bold': False})
    curr_y, max_x = TOP, 660

    for lane in lanes:
        members = [n for n in graph['nodes'] if n['parent_id'] == lane['id']]
        local = {n['id']: n for n in members}
        edges = sorted((e for e in graph['links']
                        if e['source'] in local and e['target'] in local),
                       key=lambda e: e['id'])
        def lexical(ident: str) -> tuple:
            ref = local[ident]['source_refs'][0]
            return ref['path'], ref['start_byte'], ident
        members.sort(key=lambda n: lexical(n['id']))
        indeg = {n['id']: 0 for n in members}
        adj: dict[str, list[str]] = defaultdict(list)
        for e in edges:
            if e['source'] == e['target']:
                continue
            indeg[e['target']] += 1
            adj[e['source']].append(e['target'])
        queue = deque(sorted((id_ for id_, deg in indeg.items() if deg == 0), key=lexical))
        rank = {n['id']: 0 for n in members}
        visited = 0
        while queue:
            current = queue.popleft()
            visited += 1
            for target in sorted(adj[current], key=lexical):
                rank[target] = max(rank[target], rank[current] + 1)
                indeg[target] -= 1
                if indeg[target] == 0:
                    queue.append(target)
        if visited != len(members):
            graph['issues'].append({
                'code': 'STATIC_FLOW_CYCLE', 'severity': 'warning',
                'message': f'Unexpected dependency cycle in {lane["label"]}',
                'related_ir_ids': lane['ir_node_ids'],
                'source_refs': lane['source_refs'], 'affects_view': True})
            rank = {n['id']: i for i, n in enumerate(members)}

        columns: dict[int, list[str]] = defaultdict(list)
        for node in members:
            columns[rank[node['id']]].append(node['id'])
        for values in columns.values():
            values.sort(key=lexical)

        skips = [e for e in edges if rank[e['target']] - rank[e['source']] > 1]
        # Reserve visible tracks above all blocks, below the lane header. Tracks
        # are bounded for large graphs; focus mode provides detailed subgraphs.
        tracks = min(len(skips), 8)
        first_node_y = curr_y + 63 + 13 * tracks
        lane_h = max(143, first_node_y - curr_y +
                     max((len(ids) for ids in columns.values()), default=1) * DY + 10)
        lane_w = max(460, 56 + max(1, len(columns)) * DX)
        rects.append({'id': lane['id'], 'ir_ids': lane['ir_node_ids'],
                      'x': MARGIN, 'y': curr_y, 'w': lane_w, 'h': lane_h,
                      'kind': 'flow-module', 'fill': '#ffffff', 'dash': 'none'})
        rects.append({'x': MARGIN, 'y': curr_y, 'w': lane_w, 'h': HEADER_H,
                      'kind': 'header', 'fill': '#e9e9e9', 'dash': 'none'})
        texts.append({'x': MARGIN+13, 'y': curr_y+25, 'text': lane['label'],
                      'size': 13, 'bold': True})

        for column in sorted(columns):
            for row, ident in enumerate(columns[column]):
                node = local[ident]
                box = {'x': MARGIN+26+column*DX,
                       'y': first_node_y+row*DY, 'w': STEP_W, 'h': STEP_H}
                positions[ident] = box
                unresolved = node['evidence'] in ('conditional', 'unresolved')
                rects.append({'id': ident, 'ir_ids': node['ir_node_ids'], **box,
                              'kind': node['kind'],
                              'fill': '#f4f4f4' if node['kind'] in ('flow-call', 'flow-operation') else '#ffffff',
                              'dash': '4,3' if unresolved else 'none'})
                label = node['label']
                if len(label) > 30:
                    label = label[:29]+'…'
                texts.append({'x': box['x']+9, 'y': box['y']+23,
                              'text': label, 'size': 10.5,
                              'bold': node['kind'] in ('forward_input', 'forward_output')})
                sub = (node.get('shape') + '  (symbolic)' if node.get('shape') else
                       'conditional' if node['evidence']=='conditional' else
                       node['kind'].replace('flow-', '').replace('forward_', ''))
                texts.append({'x': box['x']+9, 'y': box['y']+43,
                              'text': sub, 'size': 8.5, 'bold': False})

        skips = sorted(skips, key=lambda e: (rank[e['source']], rank[e['target']], e['id']))
        track_idx = {e['id']: i % max(1, tracks) for i, e in enumerate(skips)}
        for e in edges:
            a, b = positions[e['source']], positions[e['target']]
            x1, y1 = a['x']+a['w'], a['y']+a['h']/2
            x2, y2 = b['x'], b['y']+b['h']/2
            if e['id'] in track_idx:
                # Travel only through inter-column channels, not through boxes.
                band_y = curr_y + 47 + track_idx[e['id']]*13
                points = [(x1,y1),(x1+17,y1),(x1+17,band_y),
                          (x2-17,band_y),(x2-17,y2),(x2,y2)]
            elif x2 > x1:
                mid = (x1+x2)/2
                points = [(x1,y1),(mid,y1),(mid,y2),(x2,y2)]
            else:
                # A cycle or backward reference is drawn outside the module's
                # member boxes and remains explicitly labeled by its evidence.
                outer = MARGIN + lane_w - 13
                below = curr_y + lane_h - 8
                points = [(x1,y1),(outer,y1),(outer,below),
                          (x2-17,below),(x2-17,y2),(x2,y2)]
            # Remove zero-length steps while preserving exact orthogonality.
            points = [pt for i,pt in enumerate(points) if i==0 or pt!=points[i-1]]
            paths.append({'id': e['id'], 'ir_edge_ids': e['ir_edge_ids'],
                          'kind': e['style'], 'label': e['relation_kind'],
                          'points': points, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
        curr_y += lane_h + 24
        max_x = max(max_x, MARGIN+lane_w+35)

    height = int(curr_y+51)
    texts.append({'x':MARGIN,'y':height-43,
                  'text':'Solid = static value dependency; dashed = conditional; dotted = conditional call',
                  'size':9,'bold':False})
    texts.append({'x':MARGIN,'y':height-25,
                  'text':('PARTIAL / NOT FULLY VERIFIED: see presentation-health.json'
                          if status=='partial' else
                          'Scope: static forward expressions and guarded alternatives only'),
                  'size':9,'bold':status=='partial'})
    paths.sort(key=lambda p: (p['y1'],p['x1'],p['y2'],p['x2'],p['id']))
    return {'width': int(max_x), 'height': height, 'rects': rects,
            'texts': texts, 'paths': paths, 'title': entrypoint or 'Static Forward Architecture',
            'status': status, 'positions': positions, 'style': style}
