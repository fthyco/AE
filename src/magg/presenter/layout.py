"""Single deterministic scene for all scientific renderers.

A parent-child relation is shown as nesting, NOT as a tensor-flow arrow.
Cross-reference edges have explicit semantic labels and dashed styles.
"""
from __future__ import annotations

from collections import defaultdict
import textwrap
from typing import Any

LEAF_W, LEAF_H = 250, 62
PAD, GAP, HEAD = 22, 12, 64


def make_scene(graph: dict[str, Any], *, status: str, entrypoint: str | None) -> dict[str, Any]:
    nodes = {n['id']: n for n in graph['nodes']}
    children: dict[str | None, list[str]] = defaultdict(list)
    for n in nodes.values():
        children[n['parent_id']].append(n['id'])
    def lexical(ident: str) -> tuple:
        r = nodes[ident]['source_refs'][0]
        return (r['path'],r['start_byte'],nodes[ident]['kind'],ident)
    for members in children.values():
        members.sort(key=lexical)

    sizes: dict[str, tuple[float,float]] = {}
    def measure(ident: str) -> tuple[float,float]:
        n=nodes[ident]; kids=children[ident]
        if not kids:
            sizes[ident]=(LEAF_W,LEAF_H)
        elif n['kind'] in ('model','module'):
            left=[i for i in kids if nodes[i]['kind'] in ('component','unresolved-component','nested-constructor')]
            right=[i for i in kids if i not in left]
            leftdims=[measure(i) for i in left]
            rightdims=[measure(i) for i in right]
            lw=max((a for a,_ in leftdims),default=0)
            rw=max((a for a,_ in rightdims),default=0)
            lh=sum(h for _,h in leftdims)+max(0,len(left)-1)*GAP
            rh=sum(h for _,h in rightdims)+max(0,len(right)-1)*GAP
            extra=PAD if left and right else 0
            width=max(LEAF_W+2*PAD, lw+rw+2*PAD+extra)
            height=HEAD+max(lh,rh,LEAF_H)+PAD
            sizes[ident]=(width,height)
        else:
            childdims=[measure(i) for i in kids]
            sizes[ident]=(max(LEAF_W, max(w for w,_ in childdims)+2*PAD),
                          HEAD+sum(h for _,h in childdims)+max(0,len(kids)-1)*GAP+PAD)
        return sizes[ident]

    roots=children[None]
    for root in roots:
        measure(root)
    layout: dict[str,dict[str,float]]={}
    def place(ident: str,x: float,y: float) -> None:
        w,h=sizes[ident]; layout[ident]={'x':x,'y':y,'w':w,'h':h}
        n=nodes[ident];kids=children[ident]
        if n['kind'] in ('model','module'):
            left=[i for i in kids if nodes[i]['kind'] in ('component','unresolved-component','nested-constructor')]
            right=[i for i in kids if i not in left]
            lw=max((sizes[i][0] for i in left),default=0)
            cursor=y+HEAD
            for child in left:
                place(child,x+PAD,cursor);cursor+=sizes[child][1]+GAP
            cursor=y+HEAD
            rx=x+PAD+lw+(PAD if left and right else 0)
            for child in right:
                place(child,rx,cursor);cursor+=sizes[child][1]+GAP
        else:
            cursor=y+HEAD
            for child in kids:
                place(child,x+PAD,cursor);cursor+=sizes[child][1]+GAP

    ypos=82
    for root in roots:
        place(root,40,ypos); ypos+=sizes[root][1]+36
    canvas_w=max((p['x']+p['w'] for p in layout.values()),default=700)+40
    canvas_h=max((p['y']+p['h'] for p in layout.values()),default=200)+95
    scene={'width':int(canvas_w),'height':int(canvas_h),'rects':[], 'texts':[],
           'paths':[], 'title':(entrypoint or 'Static Model Architecture').replace(':',' / '),
           'status':status, 'positions':layout}
    scene['texts'].append({'x':40,'y':35,'text':scene['title'],'size':21,'bold':True})
    scene['texts'].append({'x':40,'y':57,'text':'STATIC ARCHITECTURE  |  source-backed module / call relationships','size':10,'bold':False})
    for ident in sorted(layout,key=lambda i:(layout[i]['y'],layout[i]['x'],i)):
        n=nodes[ident];p=layout[ident]
        grouped=bool(children[ident])
        ismodule=n['kind'] in ('model','module')
        unresolved=n['evidence']=='unresolved'
        rect={'id':ident,'ir_ids':n['ir_node_ids'],'x':p['x'],'y':p['y'],'w':p['w'],'h':p['h'],'kind':n['kind'],
              'dash': '5,4' if unresolved else 'none',
              'fill':'#ffffff' if (ismodule or grouped) else '#f5f5f5' if n['kind']=='component' else '#ffffff'}
        scene['rects'].append(rect)
        if grouped:
            scene['rects'].append({'x':p['x'],'y':p['y'],'w':p['w'],'h':HEAD-15,'kind':'header',
                                  'dash':'none','fill':'#e9e9e9' if ismodule else '#f0f0f0'})
        max_chars=max(12,int((p['w']-28)/7.2))
        lines=textwrap.wrap(n['label'],width=min(max_chars,48),break_long_words=True,break_on_hyphens=False)
        if not lines:lines=['?']
        if len(lines)>2 and not grouped:
            lines=[lines[0],lines[1][:max(0,max_chars-1)]+'…']
        ypos_text=p['y']+(25 if grouped else 22)
        for i,t in enumerate(lines[:2] if not grouped else lines[:1]):
            scene['texts'].append({'x':p['x']+14,'y':ypos_text+i*17,'text':t,
                                   'size':13 if grouped else 12,'bold':ismodule or grouped})
        if grouped:
            scene['texts'].append({'x':p['x']+14,'y':p['y']+45,'text':n['kind'].upper(), 'size':9,'bold':False})
        elif n['details']:
            d=n['details']
            if len(d)>max_chars:d=d[:max_chars-1]+'…'
            scene['texts'].append({'x':p['x']+14,'y':p['y']+51,'text':d,'size':9,'bold':False})
        if unresolved:
            scene['texts'].append({'x':p['x']+p['w']-78,'y':p['y']+11,'text':'UNRESOLVED', 'size':8,'bold':True})

    suppress={'DECLARES_COMPONENT','CONSTRUCTS_LOCAL_MODULE','HAS_FORWARD_CALL',
              'HAS_FORWARD_OPERATION','CONTAINS_NESTED_CONSTRUCTOR'}
    for link in graph['links']:
        if link['relation_kind'] in suppress:continue
        a,b=layout[link['source']],layout[link['target']]
        if link['relation_kind']=='USES_MODULE_REFERENCE':
            x1,y1=a['x'],a['y']+a['h']/2
            x2,y2=b['x']+b['w'],b['y']+min(b['h'],LEAF_H)/2
            bend=max(28,abs(x1-x2)*0.35)
            curve={'x1':x1,'y1':y1,'cx1':x1-bend,'cy1':y1,
                   'cx2':x2+bend,'cy2':y2,'x2':x2,'y2':y2,'label':'uses','kind':'reference'}
        elif link['relation_kind']=='EXPRESSION_DEPENDS_ON':
            x1,y1=a['x']+a['w'],a['y']+min(a['h'],LEAF_H)/2
            x2,y2=b['x']+b['w'],b['y']+min(b['h'],LEAF_H)/2
            bend=35
            curve={'x1':x1,'y1':y1,'cx1':x1+bend,'cy1':y1,
                   'cx2':x2+bend,'cy2':y2,'x2':x2,'y2':y2,
                   'label':'expr dependency','kind':'syntax-dependency'}
        else:
            continue
        curve['id']=link['id'];curve['ir_edge_ids']=link['ir_edge_ids'];scene['paths'].append(curve)
    scene['texts'].append({'x':40,'y':canvas_h-49,
                           'text':'Nesting = declared structure  |  dashed arrows = module references  |  dotted = source-expression dependency',
                           'size':9,'bold':False})
    scene['texts'].append({'x':40,'y':canvas_h-31,
                           'text':('PARTIAL / NOT FULLY VERIFIED  |  ' if status=='partial' else '')+
                                  'NO RUNTIME TENSOR FLOW IS IMPLIED',
                           'size':10,'bold':status=='partial'})
    scene['paths'].sort(key=lambda a:(a['kind'],a['y1'],a['x1'],a['y2'],a['x2']))
    return scene
