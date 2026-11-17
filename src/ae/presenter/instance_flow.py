"""Render source-backed instance-forward artifact without changing its dependencies."""
from __future__ import annotations

import json

from .health import issue
from .instance_hierarchy import project_vit_hierarchy
from .themes import select_theme


def project_instance_forward(bundle, level='operation', focus=None):
    if level not in ('overview','block','operation'):
        return {'view':'instance','level':level,'nodes':[],'links':[],
                'issues':[issue('UNSUPPORTED_INSTANCE_PROJECTION',
                                'Unsupported instance hierarchy level',severity='error')]}
    file=bundle.directory/'instance-forward.json'
    if not file.is_file():
        return {'view':'instance','level':level,'nodes':[],'links':[],
                'issues':[issue('MISSING_INSTANCE_FORWARD',
                                'Extract with --input-shape to build an instance graph',severity='error')]}
    data=json.loads(file.read_bytes())
    if data.get('status')!='expanded_static':
        return {'view':'instance','level':level,'nodes':[],'links':[],
                'issues':[issue('INSTANCE_FORWARD_UNRESOLVED',
                                '; '.join(data.get('issues',[])),severity='error')]}
    nodes=[{'id':x['id'],'kind':'instance-operation','label':x['target'],
            'shape':'×'.join(map(str,x['shape'])) if x['shape'] is not None else 'metadata','evidence':'derived',
            'parent_id':None,'ir_edge_ids':[],'ir_node_ids':[x['id']] + x['source_ir_ids'] +
                ([x['factory_instance_id']] if x.get('factory_instance_id') else []),
            'source_refs':x['source_refs'],'instance_node_id':x['id'],
            'op':x['op'],'target':x['target']}
           for x in data['nodes']]
    links=[{'id':x['id'],'source':x['source'],'target':x['target'],
            'relation_kind':x.get('relation','STATIC_INSTANCE_TENSOR_DEPENDENCY'),
            'ir_edge_ids':[x['id']],'instance_edge_id':x['id'],'input_slot':x['input_slot'],
            'source_refs':[], 'style':'static-value'} for x in data['edges']]
    graph = {'view':'instance','level':'operation','nodes':nodes,'links':links,
            'issues':[issue('INDEPENDENT_EXECUTION_NOT_IN_BUNDLE',
                            'Static expansion is source-proven; runtime matching is performed by the separate verifier')],
            'input_shape':data['input_conditions']['input_shape'],
            'source_path':data['source_path']}
    if level != 'operation' or focus is not None:
        return project_vit_hierarchy(
            graph, level=level, focus=focus, source_scope=data.get('scope','')
        )
    return graph


def make_instance_scene(graph, *, status, entrypoint):
    """One top-to-bottom pass, with separate branch lanes for residual projections."""
    style=select_theme('scientific-latex')
    nodes=graph['nodes'];links=graph['links']
    rects=[];texts=[];paths=[]
    step=62;top=125;main_x=180;project_x=510
    positions={}
    texts.extend([{'x':55,'y':32,'text':entrypoint or 'Instance forward','size':18,'bold':True},
                  {'x':55,'y':54,'text':'SOURCE-EXPANDED INSTANCE TENSOR GRAPH','size':10,'bold':False},
                  {'x':55,'y':73,'text':f"Input shape: {graph.get('input_shape')}",
                   'size':10,'bold':False}])
    last_stage=None
    for idx,node in enumerate(nodes):
        tgt=node['target']
        stage=('ARCHITECTURE' if graph.get('level') in ('overview','block') else
               tgt.split('.')[0] if tgt.startswith('layer') else
               'INPUT' if node['op']=='placeholder' else
               'OUTPUT' if node['op']=='output' else
               'HEAD' if tgt in {'avgpool','fc','flatten'} else 'STEM')
        y=top+idx*step
        if stage != last_stage:
            texts.append({'x':55,'y':y-11,'text':stage.upper(), 'size':11,'bold':True})
            last_stage=stage
        x=project_x if '.downsample.' in tgt else main_x
        pos={'x':x,'y':y,'w':237,'h':46}
        positions[node['id']]=pos
        rects.append({'id':node['id'],'ir_ids':node['ir_node_ids'],**pos,
                      'kind':'instance-operation','fill':'#f4f4f4','dash':'none'})
        label=tgt if len(tgt)<=37 else tgt[:34]+'...'
        texts.append({'x':x+9,'y':y+19,'text':label,'size':10,'bold':node['op'] in ('placeholder','output')})
        texts.append({'x':x+9,'y':y+37,'text':node['shape'],'size':8.4,'bold':False})
    for idx,edge in enumerate(links):
        a=positions[edge['source']];b=positions[edge['target']]
        x1=a['x']+a['w']/2;y1=a['y']+a['h']
        x2=b['x']+b['w']/2;y2=b['y']
        if b['y']>a['y'] and b['y']-a['y']<=step+0.01 and a['x']==b['x']:
            points=[(x1,y1),(x2,y2)]
        else:
            # All branch routes stay outside component rectangles.
            gap=18 + (idx%13)*10
            route=(project_x+237+gap if a['x']==project_x or b['x']==project_x else main_x-30-gap)
            points=[(x1,y1),(route,y1+6),(route,y2-6),(x2,y2)]
            # Insert orthogonal segments where endpoints differ from the outer lane.
            points=[(x1,y1),(route,y1),(route,y2),(x2,y2)]
        paths.append({'id':edge['id'],'ir_edge_ids':edge['ir_edge_ids'],
                      'kind':'static-value','label':edge['relation_kind'],'points':points,
                      'x1':x1,'y1':y1,'x2':x2,'y2':y2})
    height=top+len(nodes)*step+80
    texts.append({'x':55,'y':height-43,'text':'Shapes apply to the selected concrete input; runtime verification is separate.',
                  'size':9,'bold':False})
    texts.append({'x':55,'y':height-24,'text':f'Source-derived graph; health status: {status}',
                  'size':9,'bold':False})
    return {'width':1000,'height':height,'rects':rects,'texts':texts,'paths':paths,
            'title':entrypoint or 'Instance forward','status':status,
            'positions':positions,'style':style}
