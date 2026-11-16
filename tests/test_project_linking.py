"""Phase 1: unique local project links, conservative scope restrictions."""
import json
from pathlib import Path
from magg.sae import extract


def run(tmp_path, files):
    root = tmp_path / 'src'; root.mkdir()
    for name, code in files.items():
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(code)
    out = tmp_path / 'out'; extract(root, out)
    return json.loads((out / 'architecture.ir.json').read_text())


def rels(ir, type_):
    return [e for e in ir['edges'] if e['relation'] == type_]


def test_local_import_and_cross_file_call(tmp_path):
    ir = run(tmp_path, {
        'pkg/__init__.py': '',
        'pkg/ops.py': 'def transform(x):\n    return x\n',
        'model.py': 'from pkg.ops import transform as f\ndef forward(x):\n    return f(x)\n'
    })
    assert len(rels(ir, 'IMPORTS_LOCAL_DEFINITION')) == 1
    assert len(rels(ir, 'CALLS_LOCAL_DEFINITION')) == 1
    assert rels(ir, 'IMPORTS_LOCAL_DEFINITION')[0]['target'] == rels(ir, 'CALLS_LOCAL_DEFINITION')[0]['target']


def test_relative_import_and_module_alias(tmp_path):
    ir = run(tmp_path, {
        'pkg/__init__.py': '',
        'pkg/utils.py': 'def norm(x):\n    return x\n',
        'pkg/model.py': 'from .utils import norm\nimport pkg.utils as utils\ndef f(x):\n    return norm(utils.norm(x))\n'
    })
    assert len(rels(ir, 'CALLS_LOCAL_DEFINITION')) == 2
    assert len(rels(ir, 'IMPORTS_LOCAL_DEFINITION')) == 1
    assert len(rels(ir, 'IMPORTS_LOCAL_MODULE')) == 1


def test_local_shadow_blocks_imported_call(tmp_path):
    ir = run(tmp_path, {
        'ops.py': 'def f(x):\n    return x\n',
        'model.py': 'from ops import f\ndef forward(x):\n    f = x\n    return f(x)\n'
    })
    assert not rels(ir, 'CALLS_LOCAL_DEFINITION')


def test_resolve_same_class_method(tmp_path):
    ir = run(tmp_path, {'model.py': '''class M:
    def forward(self, x):
        return self.inner(x)
    def inner(self, x):
        return x
'''})
    assert len(rels(ir, 'CALLS_LOCAL_DEFINITION')) == 1


def test_reassignment_of_self_prevents_false_links(tmp_path):
    ir = run(tmp_path, {'model.py': '''class M:
    def forward(self, x):
        self = x
        return self.inner(x)
    def inner(self, x):
        return x
'''})
    assert not rels(ir, 'CALLS_LOCAL_DEFINITION')


def test_duplicate_definitions_not_assumed_unique(tmp_path):
    ir = run(tmp_path, {'model.py': '''def f(x): return x
def f(x): return x
def g(x): return f(x)
'''})
    assert not rels(ir, 'CALLS_LOCAL_DEFINITION')
    assert any(u['kind']=='ambiguous_local_call' for u in ir['unresolved'])


def test_external_import_kept_without_fake_internal_link(tmp_path):
    ir = run(tmp_path, {'m.py':'import torch\ndef f(x): return torch.sigmoid(x)\n'})
    assert not rels(ir, 'IMPORTS_LOCAL_MODULE')
    assert not rels(ir, 'CALLS_LOCAL_DEFINITION')


def test_custom_module_cross_file_construction_and_reuse(tmp_path):
    ir=run(tmp_path, {
        'layers.py': 'from torch import nn\nclass Block(nn.Module):\n    def forward(self,x): return x\n',
        'model.py': '''from torch import nn
from layers import Block
class Model(nn.Module):
    def __init__(self):
        self.block = Block()
    def forward(self, x):
        a = self.block(x)
        return self.block(a)
'''
    })
    assert len([n for n in ir['nodes'] if n['kind']=='pytorch_module_definition'])==2
    custom=[e for e in ir['edges'] if e['relation']=='CONSTRUCTS_LOCAL_MODULE']
    assert len(custom)==1
    refs=[e for e in ir['edges'] if e['relation']=='USES_MODULE_REFERENCE']
    assert len(refs)==2 and len({e['target'] for e in refs})==1


def test_transitive_module_base_local(tmp_path):
    ir=run(tmp_path,{
        'base.py':'import torch\nclass Base(torch.nn.Module): pass\n',
        'derived.py':'from base import Base\nclass Child(Base): pass\n',
    })
    assert len([n for n in ir['nodes'] if n['kind']=='pytorch_module_definition'])==2
    assert len([e for e in ir['edges'] if e['relation']=='INHERITS_LOCAL_DEFINITION'])==1
