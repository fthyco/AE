"""Concrete tensor-operation shape inference tests."""
from pathlib import Path

import pytest

from ae.sae.api import extract, validate_bundle
from ae.presenter.reader import load_bundle


def shapes(tmp_path: Path, body: str, imports: str = 'import torch'):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'model.py').write_text(f'''{imports}
class Model(torch.nn.Module):
 def forward(self, x):
{body}
''')
    output = tmp_path / 'bundle'
    extract(source, output, entrypoint='model:Model')
    assert validate_bundle(output)['valid']
    bundle = load_bundle(output)
    facts = {n['attributes']['value_id']:n['attributes']['dimensions'] for n in bundle.ir['nodes']
             if n['kind'] == 'symbolic_shape_fact' and n['analysis_rule'] == 'concrete-torch-operations/v1'}
    nodes = {n['id']:n for n in bundle.ir['nodes']}
    named = {}
    for ident, dimensions in facts.items():
        name = nodes[ident]['attributes'].get('callee_expression') or nodes[ident]['attributes'].get('expression')
        named[name] = dimensions
    return named, bundle


def test_literal_creator_and_matrix_multiply(tmp_path):
    known,_ = shapes(tmp_path, '  a = torch.ones(2,3)\n  b = torch.zeros(3,4)\n  return torch.matmul(a,b)')
    assert known['torch.ones'] == ['2','3']
    assert known['torch.zeros'] == ['3','4']
    assert known['torch.matmul'] == ['2','4']


def test_operator_matmul_and_batched_broadcast(tmp_path):
    known,_ = shapes(tmp_path, '  a = torch.ones(2,1,3,4)\n  b = torch.ones(1,5,4,6)\n  return a @ b')
    assert known['a @ b'] == ['2','5','3','6']


def test_matmul_mismatch_does_not_infer(tmp_path):
    known,_ = shapes(tmp_path, '  a=torch.ones(2,3)\n  b=torch.ones(4,5)\n  return torch.matmul(a,b)')
    assert 'torch.matmul' not in known


def test_matmul_batch_mismatch_does_not_infer(tmp_path):
    known,_ = shapes(tmp_path, '  a=torch.ones(2,3,4)\n  b=torch.ones(5,4,6)\n  return a @ b')
    assert 'a @ b' not in known


def test_matmul_vector_cases(tmp_path):
    known,_ = shapes(tmp_path, '  a=torch.ones(2,3)\n  b=torch.ones(3)\n  return torch.matmul(a,b)')
    assert known['torch.matmul'] == ['2']


def test_matmul_vector_matrix(tmp_path):
    known,_ = shapes(tmp_path, '  a=torch.ones(3)\n  b=torch.ones(3,5)\n  return torch.matmul(a,b)')
    assert known['torch.matmul'] == ['5']


def test_cat_exact_and_negative_axis(tmp_path):
    known,_ = shapes(tmp_path, '  a=torch.zeros(2,3)\n  b=torch.zeros(2,5)\n  return torch.cat([a,b], dim=-1)')
    assert known['torch.cat'] == ['2','8']


def test_cat_mismatched_non_concat_dim_does_not_infer(tmp_path):
    known,_ = shapes(tmp_path, '  a=torch.zeros(2,3)\n  b=torch.zeros(3,5)\n  return torch.cat([a,b], dim=1)')
    assert 'torch.cat' not in known


def test_cat_unknown_dynamic_collection_does_not_infer(tmp_path):
    known,_ = shapes(tmp_path, '  a=torch.zeros(2,3)\n  return torch.cat([a,*x], dim=0)')
    assert 'torch.cat' not in known


def test_reshape_infer_minus_one(tmp_path):
    known,_ = shapes(tmp_path, '  a=torch.zeros(2,3,4)\n  return a.reshape(6,-1)')
    assert known['a.reshape'] == ['6','4']


def test_torch_reshape_explicit_shape(tmp_path):
    known,_ = shapes(tmp_path, '  a=torch.zeros(2,3,4)\n  return torch.reshape(a,(4,6))')
    assert known['torch.reshape'] == ['4','6']


def test_reshape_invalid_numel_does_not_infer(tmp_path):
    known,_ = shapes(tmp_path, '  a=torch.zeros(2,3,4)\n  return a.view(5,-1)')
    assert 'a.view' not in known


def test_reshape_unknown_target_does_not_infer(tmp_path):
    known,_ = shapes(tmp_path, '  a=torch.zeros(2,3,4)\n  return a.reshape(x,-1)')
    assert 'a.reshape' not in known


def test_local_torch_shadow_does_not_create_shapes(tmp_path):
    known,_ = shapes(tmp_path, '  torch=x\n  return torch.zeros(2,3)')
    assert not known


def test_unknown_alias_does_not_create_shapes(tmp_path):
    known,_ = shapes(tmp_path, '  return phantom.zeros(2,3)')
    assert not known


def test_unrelated_module_forward_not_seeded(tmp_path):
    known,_ = shapes(tmp_path, '  return x')
    assert not known


def test_invalid_cat_axis_does_not_infer(tmp_path):
    known,_ = shapes(tmp_path, '  a=torch.zeros(2,3)\n  b=torch.zeros(2,3)\n  return torch.cat([a,b],dim=7)')
    assert 'torch.cat' not in known


def test_shared_linear_prefix_concat(tmp_path):
    src = tmp_path/'src'
    src.mkdir()
    (src/'model.py').write_text('''import torch
class Model(torch.nn.Module):
 def __init__(self):
  self.a=torch.nn.Linear(4,7)
  self.b=torch.nn.Linear(4,5)
 def forward(self,x):
  a=self.a(x)
  b=self.b(x)
  return torch.cat([a,b],dim=-1)
''')
    out=tmp_path/'bundle'
    extract(src,out,entrypoint='model:Model')
    b=load_bundle(out)
    calls={n['id']:n for n in b.ir['nodes'] if n['kind']=='call_site'}
    cat_id=next(k for k,n in calls.items() if n['attributes']['callee_expression']=='torch.cat')
    facts=[n for n in b.ir['nodes'] if n['kind']=='symbolic_shape_fact' and n['attributes']['value_id']==cat_id]
    assert len(facts)==1
    assert facts[0]['attributes']['dimensions']==['...','12']


def test_different_linear_inputs_do_not_share_concat_prefix(tmp_path):
    src=tmp_path/'src';src.mkdir()
    (src/'model.py').write_text('''import torch
class Model(torch.nn.Module):
 def __init__(self):
  self.a=torch.nn.Linear(4,7)
  self.b=torch.nn.Linear(4,5)
 def forward(self,x,y):
  a=self.a(x)
  b=self.b(y)
  return torch.cat([a,b],dim=-1)
''')
    out=tmp_path/'bundle'
    extract(src,out,entrypoint='model:Model')
    b=load_bundle(out)
    cat_id=next(n['id'] for n in b.ir['nodes'] if n['kind']=='call_site'
                and n['attributes']['callee_expression']=='torch.cat')
    assert not [n for n in b.ir['nodes'] if n['kind']=='symbolic_shape_fact'
                and n['attributes']['value_id']==cat_id]


def test_module_level_torch_rebinding_rejected(tmp_path):
    src=tmp_path/'src';src.mkdir()
    (src/'model.py').write_text('''import torch
torch = object()
class Model:
 def forward(self,x):
  return torch.zeros(2,3)
''')
    out=tmp_path/'bundle';extract(src,out,entrypoint='model:Model')
    b=load_bundle(out)
    assert not [n for n in b.ir['nodes'] if n['kind']=='symbolic_shape_fact'
                and n.get('analysis_rule')=='concrete-torch-operations/v1']
