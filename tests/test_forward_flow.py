"""Source-backed static forward-flow validation."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys

from magg.sae.api import extract
from magg.presenter.api import present
from magg.presenter.reader import load_bundle
from magg.presenter.flow import project_flow, FLOW_EDGES

ROOT = Path(__file__).parents[1]

def bundle(tmp_path, name, entry):
    target=tmp_path/'sae'
    extract(ROOT/'examples'/name,target,entrypoint=entry)
    return load_bundle(target)

def test_residual_paths_have_actual_ir_evidence(tmp_path):
    b=bundle(tmp_path,'cnn_residual_mvp','model:CNN')
    mod=next(m for m in b.model_index['modules'] if m['qualname'].endswith('.ResidualBlock'))
    fid=mod['forward_method_id']
    inputs=[n for n in b.ir['nodes'] if n['kind']=='forward_input' and n['attributes']['forward_id']==fid]
    add=next(o['operation_site_id'] for o in mod['forward_operations'] if o['operator']=='Add')
    conv2=next(c['call_site_id'] for c in mod['forward_calls'] if c['callee_expression']=='self.conv2')
    assert len(inputs)==1
    assert any(e['source']==inputs[0]['id'] and e['target']==add and e['relation']=='STATIC_VALUE_DEPENDENCY' for e in b.ir['edges'])
    assert any(e['source']==conv2 and e['target']==add and e['relation']=='STATIC_VALUE_DEPENDENCY' for e in b.ir['edges'])
    g=project_flow(b)
    assert all(e['relation_kind'] in FLOW_EDGES for e in g['links'])
    edges = {edge['id']: edge for edge in b.ir['edges']}
    assert all(edges[eid]['relation'] == link['relation_kind']
               for link in g['links'] for eid in link['ir_edge_ids'])
    assert all(e['ir_edge_ids'] and e['source_refs'] for e in g['links'])

def test_incomplete_attention_unpacked_values_are_conditional(tmp_path):
    b=bundle(tmp_path,'transformer_mvp','model:Transformer')
    parts=[n for n in b.ir['nodes'] if n['kind']=='forward_unpack']
    assert sorted(p['attributes']['name'] for p in parts)==['k','q','v']
    assert all(p['resolution']=='conditional' for p in parts)
    result=present(b.directory,tmp_path/'paper',view='flow',mode='publication')
    assert result['status']=='partial' and result['render_allowed'] is False

def test_flow_vector_exports_and_hash_seed_determinism(tmp_path):
    b=bundle(tmp_path,'cnn_residual_mvp','model:CNN')
    for seed in ('4','173'):
        output=tmp_path/('out'+seed)
        env={**os.environ,'PYTHONHASHSEED':seed,'PYTHONPATH':str(ROOT/'src')}
        proc=subprocess.run([sys.executable,'-m','magg.cli','present',str(b.directory),
            '--view','flow','--format','all','--output',str(output)],env=env,capture_output=True)
        assert proc.returncode==0,proc.stderr.decode()
    for filename in ('architecture.svg','architecture.pdf','architecture.tex','presentation-manifest.json','presentation-health.json'):
        assert (tmp_path/'out4'/filename).read_bytes()==(tmp_path/'out173'/filename).read_bytes(),filename
    m=json.loads((tmp_path/'out4/presentation-manifest.json').read_text())
    assert m['view']=='flow'
    assert m['visual_links'] and all(v['ir_edge_ids'] for v in m['visual_links'].values())

def test_branch_does_not_invent_forward_dependency(tmp_path):
    src=tmp_path/'source';src.mkdir()
    (src/'model.py').write_text('from torch import nn\nclass M(nn.Module):\n def forward(self,x,flag):\n  if flag:\n   x = self.fc(x)\n  return x\n')
    target=tmp_path/'sae'
    extract(src,target,entrypoint='model:M')
    result=present(target,tmp_path/'paper',view='flow')
    assert result['status']=='partial'
    assert any(i['code']=='STATIC_FLOW_INCOMPLETE' for i in json.loads((tmp_path/'paper/presentation-health.json').read_text())['issues'])
