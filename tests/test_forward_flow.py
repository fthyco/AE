"""Static forward-flow evidence, fidelity, publication gates, reproducibility."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from magg.sae.api import extract
from magg.presenter.api import present
from magg.presenter.reader import load_bundle
from magg.presenter.flow import project_flow

ROOT = Path(__file__).parents[1]


def collect(tmp_path, fixture='cnn_residual_mvp', entrypoint='model:CNN'):
    path = tmp_path / 'sae'
    extract(ROOT/'examples'/fixture,path,entrypoint=entrypoint)
    return load_bundle(path)


def flow_nodes(bundle, module_name):
    module = next(m for m in bundle.model_index['modules'] if m['qualname'].endswith('.'+module_name))
    fid = module['forward_method_id']
    values = {n['id']:n for n in bundle.ir['nodes'] if n['kind'].startswith('forward_')
              and n['attributes'].get('forward_id')==fid}
    calls = {c['call_site_id']:bundle.nodes[c['call_site_id']] for c in module['forward_calls']}
    ops = {o['operation_site_id']:bundle.nodes[o['operation_site_id']] for o in module['forward_operations']}
    return values, calls, ops


def dependencies(bundle, src: str, dst: str) -> list[dict]:
    return [e for e in bundle.ir['edges'] if e['source']==src and e['target']==dst
            and e['relation'] in ('STATIC_VALUE_DEPENDENCY','STATIC_FORWARD_RETURN')]


def test_residual_value_dependencies_are_correct_and_proven(tmp_path):
    b=collect(tmp_path)
    values,calls,ops=flow_nodes(b,'ResidualBlock')
    x=next(id for id,n in values.items() if n['kind']=='forward_input' and n['attributes']['name']=='x')
    add=next(id for id,n in ops.items() if n['attributes']['operator']=='Add')
    conv2=next(id for id,n in calls.items() if n['attributes']['callee_expression']=='self.conv2')
    assert dependencies(b,x,add), 'missing direct residual skip'
    assert dependencies(b,conv2,add), 'missing convolution branch entering residual add'
    for edge in dependencies(b,x,add)+dependencies(b,conv2,add):
        assert edge['analysis_rule']=='straight-line-forward-value/v1'
        assert edge['source_refs'][0]['path'].endswith('model.py')
    assert not any(e['relation']=='TENSOR_FLOW' for e in b.ir['edges'])


def test_nested_calls_chain_and_return(tmp_path):
    b=collect(tmp_path,'mlp_mvp','model:MLP')
    vals,calls,_=flow_nodes(b,'MLP')
    input_id=next(i for i,n in vals.items() if n['kind']=='forward_input')
    proj=next(i for i,n in calls.items() if n['attributes']['callee_expression']=='self.proj')
    gelu=next(i for i,n in calls.items() if n['attributes']['callee_expression']=='F.gelu')
    head=next(i for i,n in calls.items() if n['attributes']['callee_expression']=='self.head')
    output=next(i for i,n in vals.items() if n['kind']=='forward_output')
    for src,dst in [(input_id,proj),(proj,gelu),(gelu,head),(head,output)]:
        assert dependencies(b,src,dst)


def test_attention_destructuring_stays_conditional(tmp_path):
    b=collect(tmp_path,'transformer_mvp','model:Transformer')
    value_nodes,_,_=flow_nodes(b,'Attention')
    unpack=[n for n in value_nodes.values() if n['kind']=='forward_unpack']
    assert sorted(n['attributes']['name'] for n in unpack)==['k','q','v']
    assert all(n['resolution']=='conditional' for n in unpack)
    g=project_flow(b,'operation')
    assert any(i['code']=='STATIC_FLOW_INCOMPLETE' for i in g['issues'])
    assert all(e['ir_edge_ids'] and e['source_refs'] for e in g['links'])


def test_literal_axis_tuples_do_not_become_tensor_producers(tmp_path):
    b=collect(tmp_path)
    g=project_flow(b)
    assert not any(i['code']=='STATIC_FLOW_INCOMPLETE' for i in g['issues'])
    assert all(n['label']!='USub' for n in g['nodes'])


def test_unknown_control_flow_is_partial_not_invented(tmp_path):
    code='''from torch import nn
class M(nn.Module):
 def forward(self, x, flag):
  if flag:
   x = self.a(x)
  return x
'''
    source=tmp_path/'source';source.mkdir();(source/'model.py').write_text(code)
    target=tmp_path/'bundle';extract(source,target,entrypoint='model:M')
    report=present(target,tmp_path/'draw',view='flow')
    assert report['status']=='partial'
    assert any(i['code']=='STATIC_FLOW_INCOMPLETE' for i in json.loads((tmp_path/'draw/presentation-health.json').read_text())['issues'])
    assert present(target,tmp_path/'paper',view='flow',mode='publication')['render_allowed'] is False


def test_forward_flow_is_subgraph_with_valid_references(tmp_path):
    b=collect(tmp_path,'transformer_mvp','model:Transformer')
    g=project_flow(b,'block')
    for node in g['nodes']:
        assert node['ir_node_ids']
        assert all(x in b.nodes for x in node['ir_node_ids'])
        assert node['source_refs']
    for link in g['links']:
        assert all(x in b.edges for x in link['ir_edge_ids'])
        assert link['relation_kind'] in ('STATIC_VALUE_DEPENDENCY','STATIC_FORWARD_RETURN')
    assert len(g['links'])>=20


def test_no_flow_without_new_sae_evidence(tmp_path):
    # Older verified bundle cannot be silently promoted to a forward-flow graph.
    bundle=collect(tmp_path)
    base=bundle.directory
    ir=json.loads((base/'architecture.ir.json').read_text())
    ir['edges']=[e for e in ir['edges'] if not e['relation'].startswith('STATIC_')]
    # Changing the IR hash must block entirely, not create a diagram from guesses.
    (base/'architecture.ir.json').write_text(json.dumps(ir))
    result=present(base,tmp_path/'out',view='flow')
    assert result['status']=='blocked'
    assert not (tmp_path/'out/architecture.svg').exists()


def test_flow_output_is_deterministic_across_hash_seeds(tmp_path):
    b=collect(tmp_path,'cnn_residual_mvp','model:CNN')
    for seed in ('4','173'):
        env={**os.environ,'PYTHONHASHSEED':seed,'PYTHONPATH':str(ROOT/'src')}
        out=tmp_path/f'out-{seed}'
        proc=subprocess.run([sys.executable,'-m','magg.cli','present',str(b.directory),
             '--view','flow','--level','block','--format','all','--output',str(out)],
             env=env,capture_output=True)
        assert proc.returncode==0,proc.stderr.decode()
    for file in ('architecture.svg','architecture.pdf','architecture.tex','presentation-health.json','presentation-manifest.json'):
        assert (tmp_path/'out-4'/file).read_bytes()==(tmp_path/'out-173'/file).read_bytes(),file


def test_pdf_tikz_and_figure_mapping_from_one_scene(tmp_path):
    b=collect(tmp_path)
    p=tmp_path/'paper'
    result=present(b.directory,p,view='flow',format='all')
    assert result['status']=='complete'
    assert (p/'architecture.pdf').read_bytes().startswith(b'%PDF-')
    assert b'\\begin{tikzpicture}' in (p/'architecture.tex').read_bytes()
    manifest=json.loads((p/'presentation-manifest.json').read_bytes())
    assert manifest['view']=='flow'
    assert all(x['ir_edge_ids'] for x in manifest['visual_links'].values())
