"""v0.6 conditional data dependency and per-module figure acceptance."""
from __future__ import annotations

import json
from pathlib import Path

from magg.sae.api import extract
from magg.presenter.api import present
from magg.presenter.flow import project_flow
from magg.presenter.reader import load_bundle


def bundle_for(tmp_path: Path, code: str):
    src=tmp_path/'src'; src.mkdir(); (src/'net.py').write_text(code)
    dest=tmp_path/'sae'
    extract(src,dest,entrypoint='net:M')
    return load_bundle(dest)


def test_conditional_downsample_and_augassign_are_connected(tmp_path):
    code='''from torch import nn
class M(nn.Module):
 def forward(self,x):
  identity=x
  out=self.conv(x)
  if self.downsample is not None:
   identity=self.downsample(x)
  out += identity
  return self.relu(out)
'''
    b=bundle_for(tmp_path,code)
    nodes={n['id']:n for n in b.ir['nodes']}
    value_nodes=[n for n in nodes.values() if n['kind'] in ('forward_choice','forward_augassign')]
    assert sorted(n['kind'] for n in value_nodes)==['forward_augassign','forward_choice']
    choose=next(n for n in value_nodes if n['kind']=='forward_choice')
    aug=next(n for n in value_nodes if n['kind']=='forward_augassign')
    incoming=[e for e in b.ir['edges'] if e['target']==choose['id'] and e['relation']=='STATIC_CONDITIONAL_DEPENDENCY']
    assert len(incoming)==2
    assert {nodes[e['source']]['kind'] for e in incoming} == {'forward_input','call_site'}
    aug_inputs=[e for e in b.ir['edges'] if e['target']==aug['id'] and e['relation']=='STATIC_VALUE_DEPENDENCY']
    assert {e['source'] for e in aug_inputs} == {choose['id'], next(n['id'] for n in nodes.values() if n['kind']=='call_site' and n['attributes'].get('callee_expression')=='self.conv')}
    guarded=[e for e in b.ir['edges'] if e['relation']=='STATIC_GUARDED_CALL']
    assert len(guarded)==1
    assert any('mutation/aliases' in u['reason'] for u in b.ir['unresolved'] if u['kind']=='forward_flow_incomplete')
    r=present(b.directory,tmp_path/'img',view='flow',focus='M',format='all')
    assert r['status']=='partial' and r['render_allowed']
    svg=(tmp_path/'img/architecture.svg').read_text()
    assert 'net:M' in svg and 'net:M / M' not in svg
    assert 'identity' in svg and 'forward_augassign' not in svg  # node is displayed by semantic label
    manifest=json.loads((tmp_path/'img/presentation-manifest.json').read_text())
    assert manifest['focus']=='M'


def test_if_else_keeps_both_paths_with_join(tmp_path):
    code='''from torch import nn
class M(nn.Module):
 def forward(self,x,flag):
  if flag:
   y=self.a(x)
  else:
   y=self.b(x)
  return y
'''
    b=bundle_for(tmp_path,code)
    choices=[n for n in b.ir['nodes'] if n['kind']=='forward_choice']
    assert len(choices)==1
    edges=[e for e in b.ir['edges'] if e['target']==choices[0]['id'] and e['relation']=='STATIC_CONDITIONAL_DEPENDENCY']
    assert len(edges)==2
    g=project_flow(b,focus='M')
    assert len([e for e in g['links'] if e['relation_kind']=='STATIC_CONDITIONAL_DEPENDENCY'])==2


def test_unbound_branch_does_not_claim_unconditional_return(tmp_path):
    code='''class M:
 def forward(self,x,flag):
  if flag:
   y=x
  return y
'''
    b=bundle_for(tmp_path,code)
    assert any('unbound on one path' in x['reason'] for x in b.ir['unresolved'])
    output=next(n for n in b.ir['nodes'] if n['kind']=='forward_output')
    assert not any(e['target']==output['id'] and e['relation']=='STATIC_FORWARD_RETURN' for e in b.ir['edges'])


def test_unknown_focus_blocks_rendering(tmp_path):
    b=bundle_for(tmp_path,'''class M:
 def forward(self,x): return x
''')
    result=present(b.directory,tmp_path/'img',view='flow',focus='DoesNotExist')
    assert result['status']=='blocked'
    assert not (tmp_path/'img/architecture.svg').exists()


def test_unhandled_loop_stays_partial(tmp_path):
    code='''class M:
 def forward(self,x):
  for i in range(4):
   x=self.layer(x)
  return x
'''
    b=bundle_for(tmp_path,code)
    result=present(b.directory,tmp_path/'img',view='flow',focus='M')
    assert result['status']=='partial'
    assert any('Unsupported forward statement: For' in u['reason'] for u in b.ir['unresolved'] if u['kind']=='forward_flow_incomplete')
