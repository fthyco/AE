"""End-to-end Presenter contract and source-grounded integrity tests."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ae.sae.api import extract
from ae.presenter.api import present
from ae.presenter.reader import load_bundle
from ae.presenter.projection import project

ROOT=Path(__file__).parents[1]


def fixture_bundle(tmp_path:Path,name='mlp_mvp',entrypoint=None)->Path:
    target=tmp_path/name
    extract(ROOT/'examples'/name,target,entrypoint=entrypoint or {'mlp_mvp':'model:MLP',
            'cnn_residual_mvp':'model:CNN','transformer_mvp':'model:Transformer'}[name])
    return target


@pytest.mark.parametrize('name', ['mlp_mvp','cnn_residual_mvp','transformer_mvp'])
def test_end_to_end_fixtures(tmp_path,name):
    bundle=fixture_bundle(tmp_path,name)
    result=present(bundle,tmp_path/'figure',format='all',level='block')
    assert result['status']=='complete'
    for ext in ('svg','pdf','tex'):
        assert (tmp_path/'figure'/('architecture.'+ext)).stat().st_size>100
    assert (tmp_path/'figure'/'architecture.pdf').read_bytes().startswith(b'%PDF-')
    assert b'\\begin{tikzpicture}' in (tmp_path/'figure'/'architecture.tex').read_bytes()


def test_every_visual_element_and_link_has_source_ir_evidence(tmp_path):
    bundle=fixture_bundle(tmp_path,'transformer_mvp')
    b=load_bundle(bundle)
    for level in ('overview','block','operation'):
        present(bundle,tmp_path/'figure',level=level)
        m=json.loads((tmp_path/'figure'/'presentation-manifest.json').read_bytes())
        for entry in m['visual_elements'].values():
            assert entry['ir_node_ids']
            assert entry['source_refs']
            assert set(entry['ir_node_ids'])<=set(b.nodes)
        for entry in m['visual_links'].values():
            assert entry['ir_edge_ids']
            assert entry['source_refs']
            assert set(entry['ir_edge_ids'])<=set(b.edges)
            assert entry['relation_kind'] not in ('TENSOR_FLOW','DATA_FLOW')


def test_no_lexical_sequence_claim(tmp_path):
    bundle=fixture_bundle(tmp_path,'mlp_mvp')
    g=project(load_bundle(bundle),'operation')
    assert g['links']
    assert all(e['relation_kind'] in ('DECLARES_COMPONENT','CONSTRUCTS_LOCAL_MODULE',
          'CONTAINS_NESTED_CONSTRUCTOR','HAS_FORWARD_CALL','HAS_FORWARD_OPERATION',
          'USES_MODULE_REFERENCE','EXPRESSION_DEPENDS_ON') for e in g['links'])
    present(bundle,tmp_path/'out',level='operation')
    assert 'NO RUNTIME TENSOR FLOW IS IMPLIED' in (tmp_path/'out'/'architecture.svg').read_text()


def _make_unknown(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'model.py').write_text('''from torch import nn
from missing import Encoder
class Net(nn.Module):
 def __init__(self):
  super().__init__()
  self.encoder = Encoder(128)
  self.head = nn.Linear(128, 10)
 def forward(self, x):
  return self.head(self.encoder(x))
''')
    bundle=tmp_path/'bundle';extract(source,bundle,entrypoint='model:Net')
    result=present(bundle,tmp_path/'view')
    assert result['status']=='partial'
    h=json.loads((tmp_path/'view'/'presentation-health.json').read_bytes())
    assert any(i['code']=='MISSING_LOCAL_OR_EXTERNAL_DEFINITION' for i in h['issues'])
    assert b'UNRESOLVED' in (tmp_path/'view'/'architecture.svg').read_bytes()
    return bundle


def test_unknown_module_remains_partial(tmp_path):
    _make_unknown(tmp_path)


def test_publication_gate_and_override(tmp_path):
    bundle=_make_unknown(tmp_path)
    denied=present(bundle,tmp_path/'pub',mode='publication')
    assert denied['status']=='partial' and denied['render_allowed'] is False
    assert not (tmp_path/'pub'/'architecture.svg').exists()
    h=json.loads((tmp_path/'pub'/'presentation-health.json').read_bytes())
    assert h['publication_gate']=='withheld_partial'
    permitted=present(bundle,tmp_path/'pub',mode='publication',allow_partial=True)
    assert permitted['render_allowed'] is True
    assert b'PARTIAL / NOT FULLY VERIFIED' in (tmp_path/'pub'/'architecture.svg').read_bytes()


def test_corrupt_ir_blocks_and_deletes_stale_svg(tmp_path):
    bundle=fixture_bundle(tmp_path)
    present(bundle,tmp_path/'view')
    assert (tmp_path/'view'/'architecture.svg').exists()
    ir=bundle/'architecture.ir.json'
    ir.write_bytes(ir.read_bytes()+b' ')
    result=present(bundle,tmp_path/'view')
    assert result['status']=='blocked' and result['render_allowed'] is False
    assert not (tmp_path/'view'/'architecture.svg').exists()
    issues=json.loads((tmp_path/'view'/'presentation-health.json').read_bytes())['issues']
    assert issues[0]['code']=='INVALID_BUNDLE'


def test_missing_entrypoint_blocks(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'model.py').write_text('from torch import nn\nclass M(nn.Module):\n pass\n')
    bundle=tmp_path/'bundle';extract(source,bundle,entrypoint='model:Missing')
    result=present(bundle,tmp_path/'out')
    assert result['status']=='blocked'
    assert not (tmp_path/'out'/'architecture.svg').exists()


def test_only_unrelated_syntax_error_does_not_downgrade(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'model.py').write_text('''from torch import nn
class Model(nn.Module):
 def __init__(self):
  self.proj=nn.Linear(4,8)
 def forward(self,x):
  return self.proj(x)
''')
    (source/'other.py').write_text('def broken(:\n')
    bundle=tmp_path/'bundle';extract(source,bundle,entrypoint='model:Model')
    result=present(bundle,tmp_path/'out',level='block')
    assert result['status']=='complete'


def test_empty_model_is_not_called_complete(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'model.py').write_text('class Blank:\n pass\n')
    bundle=tmp_path/'bundle';extract(source,bundle,entrypoint='model:Blank')
    assert present(bundle,tmp_path/'out')['status']=='blocked'


def test_repeatable_svg_pdf_tikz(tmp_path):
    bundle=fixture_bundle(tmp_path,'cnn_residual_mvp')
    present(bundle,tmp_path/'one',format='all',level='operation')
    present(bundle,tmp_path/'two',format='all',level='operation')
    for name in ('architecture.svg','architecture.pdf','architecture.tex',
                 'presentation-manifest.json','presentation-health.json'):
        assert (tmp_path/'one'/name).read_bytes()==(tmp_path/'two'/name).read_bytes(),name


def test_subprocess_hash_seed(tmp_path):
    bundle=fixture_bundle(tmp_path,'transformer_mvp')
    for seed in ('3','981'):
        out=tmp_path/f'out{seed}'
        env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONHASHSEED':seed}
        x=subprocess.run([sys.executable,'-m','ae.cli','present',str(bundle),'--output',str(out),
                          '--level','operation','--format','svg'],capture_output=True,env=env)
        assert x.returncode==0,x.stderr.decode()
    assert (tmp_path/'out3'/'architecture.svg').read_bytes()==(tmp_path/'out981'/'architecture.svg').read_bytes()


def test_cli_policy_exit_code(tmp_path):
    bundle=fixture_bundle(tmp_path)
    env={**os.environ,'PYTHONPATH':str(ROOT/'src')}
    p=subprocess.run([sys.executable,'-m','ae.cli','present',str(bundle),'--output',str(tmp_path/'out'),
                     '--mode','publication','--level','overview'],env=env,capture_output=True)
    assert p.returncode==0,p.stderr.decode()
