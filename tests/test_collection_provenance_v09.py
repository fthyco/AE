"""Explicit tuple/list element identity without executing user model code."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ae.sae.api import extract, validate_bundle
from ae.presenter.api import present
from ae.presenter.reader import load_bundle
from ae.presenter.flow import project_flow
from ae.presenter.flow_layout import make_flow_scene

ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / 'tests' / 'architecture_cases'


def run(tmp_path: Path, source: str, *, entrypoint: str='model:Model'):
    src=tmp_path/'src';src.mkdir()
    (src/'model.py').write_text(source)
    dest=tmp_path/'sae'
    extract(src, dest, entrypoint=entrypoint)
    validate_bundle(dest)
    return load_bundle(dest)


def rel(bundle, relation: str) -> list[dict]:
    return [e for e in bundle.ir['edges'] if e['relation'] == relation]


def test_tuple_return_is_indexed_and_traced_to_real_source(tmp_path):
    b=run(tmp_path, '''from torch import nn
class Model(nn.Module):
 def __init__(self):
  self.a=nn.Linear(4,7)
  self.b=nn.Linear(4,1)
 def forward(self,x):
  a=self.a(x)
  b=self.b(x)
  return a,b
''')
    collection=[n for n in b.ir['nodes'] if n['kind']=='forward_collection' and n['attributes'].get('collection_kind')=='tuple']
    assert len(collection)==1 and collection[0]['attributes']['element_count']==2
    members=[n for n in b.ir['nodes'] if n['kind']=='forward_collection_element' and
             n['attributes']['collection_id']==collection[0]['id']]
    assert {n['attributes']['element_index'] for n in members} == {0,1}
    assert len(rel(b,'STATIC_COLLECTION_ELEMENT'))==2
    assert len([e for e in rel(b,'STATIC_COLLECTION_MEMBER') if e['target']==collection[0]['id']])==2
    assert any(e['source']==collection[0]['id'] for e in rel(b,'STATIC_FORWARD_RETURN'))
    shapes={n['attributes']['value_id']:n['attributes']['dimensions'] for n in b.ir['nodes']
            if n['kind']=='symbolic_shape_fact'}
    assert {tuple(shapes[n['id']]) for n in members}=={('...','7'),('...','1')}
    assert collection[0]['id'] not in shapes  # tuple isn't a tensor


def test_named_tuple_variable_is_unpacked_by_proven_position(tmp_path):
    b=run(tmp_path, '''from torch import nn
class Model(nn.Module):
 def forward(self,x,y):
  pair=(x,y)
  a,b=pair
  return a
''')
    assert len(rel(b,'STATIC_COLLECTION_ELEMENT_BINDING'))==2
    assert not any('Unpacking arity' in x['reason'] for x in b.ir['unresolved'])
    assert len({e['target'] for e in rel(b,'STATIC_COLLECTION_ELEMENT_BINDING')})==2


def test_repeated_python_collection_expression_has_unique_node(tmp_path):
    b=run(tmp_path, '''from torch import nn
class Model(nn.Module):
 def forward(self,x,y):
  a=b=(x,y)
  return a
''')
    assert sum(x['kind']=='forward_collection' for x in b.ir['nodes'])==1


def test_unet_encoder_to_parent_unpack_preserves_order(tmp_path):
    out=tmp_path/'sae'
    extract(CASE_ROOT/'04_unet_skip_concat',out,entrypoint='model:UNetMini')
    validate_bundle(out)
    b=load_bundle(out)
    edges=rel(b,'STATIC_COLLECTION_ELEMENT_BINDING')
    cross=[e for e in edges if b.nodes[e['source']]['attributes'].get('forward_id') !=
           b.nodes[e['target']]['attributes'].get('forward_id')]
    assert len(cross)==2
    assert {b.nodes[e['source']]['attributes']['element_index'] for e in cross}=={0,1}
    for e in cross:
        assert b.nodes[e['source']]['attributes']['element_index']==b.nodes[e['target']]['attributes']['element_index']
        assert e['source_refs'] and e['analysis_rule']=='unique-static-return-collection-elements/v1'
    assert not any('Unpacking arity' in x['reason'] for x in b.ir['unresolved'])
    result=present(out,tmp_path/'figure',view='flow',level='operation',format='all')
    assert result['status']=='complete' # the source-dependency view, not concat tensor shape
    manifest=json.loads((tmp_path/'figure/presentation-manifest.json').read_text())
    actual=[x for x in manifest['visual_links'].values() if x['relation_kind']=='STATIC_COLLECTION_ELEMENT_BINDING']
    assert len(actual)==2 and all(e['ir_edge_ids'] for e in actual)
    graph=project_flow(b,level='operation')
    scene=make_flow_scene(graph,status=result['status'],entrypoint='model:UNetMini')
    bridges=[p for p in scene['paths'] if p['kind']=='interprocedural']
    assert len(bridges)==2
    assert all(all(a[0]==c[0] or a[1]==c[1] for a,c in zip(p['points'],p['points'][1:]))
               for p in bridges)
    assert all(p['points'][-1] != p['points'][0] for p in bridges)


def test_multihead_return_elements_preserve_distinct_shapes(tmp_path):
    out=tmp_path/'sae'
    extract(CASE_ROOT/'06_multihead_output',out,entrypoint='model:MultiHeadNet')
    b=load_bundle(out)
    members=[n for n in b.ir['nodes'] if n['kind']=='forward_collection_element']
    shapes={n['attributes']['value_id']:tuple(n['attributes']['dimensions']) for n in b.ir['nodes']
            if n['kind']=='symbolic_shape_fact'}
    assert {shapes[n['id']] for n in members}=={('...','7'),('...','1')}
    assert len(rel(b,'STATIC_COLLECTION_MEMBER'))==2


def test_unknown_chunk_arity_remains_partial_and_has_no_fabricated_elements(tmp_path):
    out=tmp_path/'sae'
    extract(CASE_ROOT/'05_attention_qkv',out,entrypoint='model:AttentionModel')
    b=load_bundle(out)
    assert not rel(b,'STATIC_COLLECTION_ELEMENT_BINDING')
    assert any('Unpacking arity/element identity unverified' in x['reason'] for x in b.ir['unresolved'])
    assert present(out,tmp_path/'flow',view='flow')['status']=='partial'


def test_mismatched_callee_tuple_length_does_not_link(tmp_path):
    b=run(tmp_path, '''from torch import nn
def helper(x):
 return x,x,x
class Model(nn.Module):
 def forward(self,x):
  a,b=helper(x)
  return a
''')
    assert not rel(b,'STATIC_COLLECTION_ELEMENT_BINDING')
    assert any('arity does not match caller unpack' in u['reason'] for u in b.ir['unresolved'])
    assert any('Unpacking arity/element identity unverified' in u['reason'] for u in b.ir['unresolved'])


def test_starred_collection_does_not_claim_fixed_arity(tmp_path):
    b=run(tmp_path, '''from torch import nn
class Model(nn.Module):
 def forward(self,x,y):
  return (x,*y)
''')
    collections=[n for n in b.ir['nodes'] if n['kind']=='forward_collection']
    assert collections and all('element_count' not in n['attributes'] for n in collections)
    assert not rel(b,'STATIC_COLLECTION_MEMBER')
    assert any('Starred collection length' in x['reason'] for x in b.ir['unresolved'])


def test_collection_in_python_concat_only_infers_tensor_not_aggregate_shape(tmp_path):
    b=run(tmp_path, '''import torch
from torch import nn
class Model(nn.Module):
 def __init__(self):
  self.a=nn.Linear(16,32)
  self.b=nn.Linear(16,8)
 def forward(self,x):
  y=torch.cat([self.a(x),self.b(x)],dim=-1)
  return y
''')
    aggregate=next(n for n in b.ir['nodes'] if n['kind']=='forward_collection')
    shapes={n['attributes']['value_id']:n['attributes']['dimensions'] for n in b.ir['nodes']
            if n['kind']=='symbolic_shape_fact'}
    assert aggregate['id'] not in shapes
    concat=next(c['call_site_id'] for m in b.model_index['modules'] for c in m['forward_calls']
                if c['callee_expression']=='torch.cat')
    assert shapes[concat] == ['...', '40']  # shared input and fixed Linear outputs


def test_full_six_views_and_different_hash_seeds(tmp_path):
    source=CASE_ROOT/'04_unet_skip_concat'
    bundle=tmp_path/'sae';extract(source,bundle,entrypoint='model:UNetMini')
    for view in ('structure','flow'):
        for level in ('overview','block','operation'):
            p=tmp_path/f'{view}-{level}'
            result=present(bundle,p,view=view,level=level)
            assert result['render_allowed']
            assert (p/'architecture.svg').exists()
    for seed in ('7','148'):
        dest=tmp_path/('seed'+seed)
        env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONHASHSEED':seed}
        cmd=[sys.executable,'-m','ae.cli','present',str(bundle),'--view','flow',
             '--level','operation','--format','svg','--output',str(dest)]
        run=subprocess.run(cmd,env=env,capture_output=True,check=False)
        assert run.returncode==0,run.stderr.decode()
    assert (tmp_path/'seed7/architecture.svg').read_bytes()==(tmp_path/'seed148/architecture.svg').read_bytes()
