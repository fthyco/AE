"""Interprocedural source-evidence MVP: assertions deliberately avoid runtime claims."""
from pathlib import Path

from ae.sae.api import extract
from ae.presenter.reader import load_bundle


def parse(tmp_path: Path, text: str, *, entry='model:Model'):
    source = tmp_path/'source'; source.mkdir()
    (source/'model.py').write_text(text)
    dest = tmp_path/'bundle'
    extract(source, dest, entrypoint=entry)
    return load_bundle(dest)


def types(bundle, relation):
    return [e for e in bundle.ir['edges'] if e['relation'] == relation]


def test_unique_nested_module_forward_binding(tmp_path):
    b = parse(tmp_path, '''from torch import nn
class Block(nn.Module):
 def __init__(self):
  self.proj=nn.Linear(128,64)
 def forward(self,x):
  return self.proj(x)
class Model(nn.Module):
 def __init__(self):
  self.block=Block()
 def forward(self,x):
  return self.block(x)
''')
    assert len(types(b,'STATIC_CALL_ARGUMENT_BINDING')) == 1
    assert len(types(b,'STATIC_CALL_RETURN')) == 1
    edge = types(b,'STATIC_CALL_RETURN')[0]
    assert b.nodes[edge['source']]['kind'] == 'forward_output'
    assert b.nodes[edge['target']]['kind'] == 'call_site'


def test_local_helper_function_binding(tmp_path):
    b=parse(tmp_path, '''from torch import nn
def helper(x):
 return x
class Model(nn.Module):
 def forward(self,x):
  return helper(x)
''')
    assert len(types(b,'STATIC_CALL_ARGUMENT_BINDING')) == 1
    assert len(types(b,'STATIC_CALL_RETURN')) == 1


def test_two_calls_same_function_do_not_share_context(tmp_path):
    b=parse(tmp_path, '''from torch import nn
def helper(x):
 return x
class Model(nn.Module):
 def forward(self,x):
  a=helper(x)
  b=helper(a)
  return b
''')
    assert types(b,'STATIC_CALL_RETURN') == []
    assert any('Repeated or recursive' in i.get('reason','') for i in b.ir['unresolved'])


def test_unknown_actual_is_not_fabricated(tmp_path):
    b=parse(tmp_path, '''from torch import nn
def helper(x):
 return x
class Model(nn.Module):
 def forward(self,x):
  return helper(5)
''')
    assert types(b,'STATIC_CALL_RETURN') == []
    assert any('provenance' in i.get('reason','') for i in b.ir['unresolved'])


def test_recursive_helper_is_not_connected(tmp_path):
    b=parse(tmp_path, '''from torch import nn
def helper(x):
 return helper(x)
class Model(nn.Module):
 def forward(self,x):
  return helper(x)
''')
    assert types(b,'STATIC_CALL_RETURN') == []


def test_keyword_binding(tmp_path):
    b=parse(tmp_path, '''from torch import nn
def helper(x,y):
 return x+y
class Model(nn.Module):
 def forward(self,x,y):
  return helper(y=y,x=x)
''')
    assert len(types(b,'STATIC_CALL_ARGUMENT_BINDING')) == 2
    assert len(types(b,'STATIC_CALL_RETURN')) == 1


def test_symbolic_shape_flows_across_custom_module_and_return(tmp_path):
    b = parse(tmp_path, '''from torch import nn
class Block(nn.Module):
 def __init__(self):
  self.proj = nn.Linear(128, 64)
 def forward(self,x):
  return self.proj(x)
class Model(nn.Module):
 def __init__(self):
  self.block=Block()
  self.head=nn.Linear(64,10)
 def forward(self,x):
  return self.head(self.block(x))
''')
    shapes = {n['attributes']['value_id']: n['attributes']['dimensions'] for n in b.ir['nodes']
              if n['kind'] == 'symbolic_shape_fact'}
    indexed = b.model_index['modules']
    parent = next(m for m in indexed if m['qualname'].endswith('Model'))
    nested = next(c for c in parent['forward_calls'] if c['callee_expression']=='self.block')
    head = next(c for c in parent['forward_calls'] if c['callee_expression']=='self.head')
    assert shapes[nested['call_site_id']] == ['...','64']
    assert shapes[head['call_site_id']] == ['...','10']
    from ae.presenter.flow import project_flow
    visual=project_flow(b,focus='Model')
    rendered=next(n for n in visual['nodes'] if nested['call_site_id'] in n['ir_node_ids'])
    assert rendered['shape'] == '[..., 64]'
    assert len(rendered['ir_node_ids'])==2  # original + symbolic fact
    assert rendered['ir_edge_ids']


def test_linear_mismatch_is_explicit_and_not_verified(tmp_path):
    b=parse(tmp_path, '''from torch import nn
class Model(nn.Module):
 def __init__(self):
  self.a=nn.Linear(16, 32)
  self.b=nn.Linear(64, 8)
 def forward(self,x):
  return self.b(self.a(x))
''')
    assert any(n['kind']=='symbolic_shape_conflict' for n in b.ir['unresolved'])
    from ae.presenter.api import present
    result=present(b.directory,tmp_path/'figure',view='flow')
    assert result['status']=='partial'


def test_no_unknown_output_dimensions_are_guessed(tmp_path):
    b=parse(tmp_path, '''from torch import nn
class Model(nn.Module):
 def __init__(self):
  self.proj=nn.Linear(cfg.input_dim,cfg.output_dim)
 def forward(self,x):
  return self.proj(x)
''')
    facts=[n for n in b.ir['nodes'] if n['kind']=='symbolic_shape_fact']
    assert any(n['attributes']['dimensions']==['...','cfg.output_dim'] for n in facts)
    assert all(n['attributes']['scope']=='conditional_type_level' for n in facts)


def test_shape_preserving_relu_transfers_supported_template(tmp_path):
    b=parse(tmp_path, '''from torch import nn
class Model(nn.Module):
 def __init__(self):
  self.proj=nn.Linear(16,32)
  self.act=nn.ReLU()
 def forward(self,x):
  return self.act(self.proj(x))
''')
    by_call={c['callee_expression']:c['call_site_id'] for m in b.model_index['modules']
             for c in m['forward_calls']}
    shapes={n['attributes']['value_id']:n['attributes']['dimensions'] for n in b.ir['nodes']
            if n['kind']=='symbolic_shape_fact'}
    assert shapes[by_call['self.act']]==['...','32']


def test_shape_is_not_propagated_through_unknown_transform(tmp_path):
    b=parse(tmp_path, '''from torch import nn
class Model(nn.Module):
 def __init__(self):
  self.proj=nn.Linear(16,32)
 def forward(self,x):
  return unknown_transform(self.proj(x))
''')
    by_call={c['callee_expression']:c['call_site_id'] for m in b.model_index['modules']
             for c in m['forward_calls']}
    shapes={n['attributes']['value_id']:n['attributes']['dimensions'] for n in b.ir['nodes']
            if n['kind']=='symbolic_shape_fact'}
    assert by_call['unknown_transform'] not in shapes
