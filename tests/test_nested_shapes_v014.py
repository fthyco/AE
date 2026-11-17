"""Fixed-point shape propagation across uniquely resolved forwards."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ae.sae.api import extract, validate_bundle
from ae.sae.analysis.nested_shapes import propagate_nested_shapes
from ae.sae.ir.models import Graph


def bundle(tmp_path, body: str):
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'model.py').write_text('import torch\nimport torch.nn as nn\n' + body)
    target = tmp_path / 'out'
    extract(src, target, entrypoint='model:Outer')
    assert validate_bundle(target)['valid']
    ir = json.loads((target / 'architecture.ir.json').read_text())
    nodes = {n['id']: n for n in ir['nodes']}
    facts = {}
    for item in ir['nodes']:
        if item['kind'] != 'symbolic_shape_fact':
            continue
        obj = nodes[item['attributes']['value_id']]
        facts.setdefault((obj['kind'], obj['source_refs'][0]['start_line'],
                          obj['attributes'].get('callee_expression')), set()).add(tuple(item['attributes']['dimensions']))
    return ir, facts, target


def for_call(facts, expression):
    return set().union(*(values for (kind, _, name), values in facts.items()
                         if kind == 'call_site' and name == expression))


INNER = '''
class Inner(nn.Module):
    def forward(self, x):
        a = torch.ones(2, 3)
        b = torch.ones(3, 4)
        return torch.matmul(a, b)
'''


def test_matmul_output_propagates_to_parent(tmp_path):
    ir, facts, _ = bundle(tmp_path, INNER + '''
class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Inner()
    def forward(self, x):
        y = self.inner(x)
        return y
''')
    assert ("2", "4") in for_call(facts, 'self.inner')
    assert any(('2', '4') in dims for (kind, line, _), dims in facts.items()
               if kind == 'forward_output')
    assert sum(n['analysis_rule'] == 'nested-static-shape-binding/v1'
               for n in ir['nodes'] if n['kind'] == 'symbolic_shape_fact') >= 3


def test_nested_reshape_receives_child_shape(tmp_path):
    ir, facts, _ = bundle(tmp_path, INNER + '''
class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Inner()
    def forward(self, x):
        y = self.inner(x)
        return y.reshape(4, -1)
''')
    assert ('4', '2') in for_call(facts, 'y.reshape')
    assert any(('4','2') in dims for (kind, line, _),dims in facts.items()
               if kind == 'forward_output')


def test_nested_concat_receives_child_shape(tmp_path):
    ir, facts, _ = bundle(tmp_path, INNER + '''
class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Inner()
    def forward(self, x):
        y = self.inner(x)
        z = torch.ones(2, 6)
        return torch.cat([y, z], dim=1)
''')
    assert ('2','10') in for_call(facts, 'torch.cat')


def test_three_level_forward_propagates(tmp_path):
    ir, facts, _ = bundle(tmp_path, INNER + '''
class Middle(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Inner()
    def forward(self, x):
        return self.inner(x)
class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.middle = Middle()
    def forward(self, x):
        return self.middle(x)
''')
    assert ('2','4') in for_call(facts, 'self.middle')


def test_unknown_input_matmul_is_not_propagated(tmp_path):
    ir, facts, _ = bundle(tmp_path, '''
class Inner(nn.Module):
    def forward(self, x):
        return torch.matmul(x, x)
class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Inner()
    def forward(self, x):
        return self.inner(x)
''')
    assert not facts


def test_repeated_callee_is_not_bound(tmp_path):
    ir, facts, _ = bundle(tmp_path, INNER + '''
class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = Inner()
        self.b = Inner()
    def forward(self, x):
        return self.a(x) + self.b(x)
''')
    assert not any(kind == 'call_site' and expression in ('self.a','self.b')
                   for (kind, line, expression) in facts)
    assert any('Repeated or recursive callee' in u.get('reason','') for u in ir['unresolved'])


def test_incompatible_matmul_dimensions_stay_unknown(tmp_path):
    ir, facts, _ = bundle(tmp_path, INNER.replace('torch.ones(3, 4)', 'torch.ones(5, 4)') + '''
class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Inner()
    def forward(self, x):
        return self.inner(x)
''')
    assert not any(expr == 'self.inner' for _,_,expr in facts)


def test_aggregate_does_not_receive_tensor_shape(tmp_path):
    ir, facts, _ = bundle(tmp_path, '''
class Inner(nn.Module):
    def forward(self, x):
        return (torch.ones(2,3), torch.ones(2,5))
class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Inner()
    def forward(self, x):
        a,b = self.inner(x)
        return a
''')
    assert not any(kind == 'forward_collection' for (kind,_,_) in facts)
    assert any(kind == 'forward_unpack' and ('2','3') in dims for (kind,_,_), dims in facts.items())


def test_conflicting_shapes_do_not_propagate():
    graph = Graph()
    ref = {'path':'model.py','start_byte':0,'end_byte':1,'start_line':1,'start_column':0,'end_line':1,'end_column':1,'precision':'exact'}
    for ident in ('a','b','target'):
        graph.add_node({'id':ident,'kind':'call_site','attributes':{},'source_refs':[ref],
                        'evidence_class':'explicit','resolution':'resolved'})
    for key, dimensions, value in [('s1',['2','3'],'a'),('s2',['2','5'],'b')]:
        graph.add_node({'id':key,'kind':'symbolic_shape_fact','attributes':{'value_id':value,'dimensions':dimensions},
                        'source_refs':[ref],'evidence_class':'derived','analysis_rule':'test','resolution':'conditional'})
    for key, src in [('e1','a'),('e2','b')]:
        graph.add_edge({'id':key,'source':src,'target':'target','relation':'STATIC_CALL_RETURN',
                        'source_refs':[ref],'evidence_class':'derived','analysis_rule':'test'})
    assert propagate_nested_shapes(graph) == 0
    assert not any(n['kind']=='symbolic_shape_fact' and n['attributes'].get('value_id')=='target'
                   for n in graph.nodes)


def test_determinism_in_cross_hash_seed_subprocess(tmp_path):
    body = INNER + '''
class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Inner()
    def forward(self, x):
        return self.inner(x)
'''
    ir, facts, output = bundle(tmp_path, body)
    src = tmp_path / 'src'
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = '971'
    dest = tmp_path / 'again'
    subprocess.run([sys.executable, '-m', 'ae.cli','extract',str(src), '--entrypoint',
                    'model:Outer','--output',str(dest)],check=True, capture_output=True, env=env)
    assert (output / 'architecture.ir.json').read_bytes() == (dest / 'architecture.ir.json').read_bytes()
