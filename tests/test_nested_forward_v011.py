"""Static forward inheritance and nested construction source contracts."""
from __future__ import annotations

import json
from pathlib import Path

from ae.sae.api import extract, validate_bundle
from ae.presenter.api import present


def analyze(tmp_path: Path, files: dict[str, str], entrypoint='model:Model'):
    src = tmp_path / 'source'
    for name, text in files.items():
        dest = src / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding='utf-8')
    out = tmp_path / 'sae'
    extract(src, out, entrypoint=entrypoint)
    assert validate_bundle(out)['valid']
    return out, json.loads((out / 'model-index.json').read_text()), json.loads((out / 'architecture.ir.json').read_text())


def test_imported_reexport_inherits_forward_and_links_child(tmp_path):
    out, index, ir = analyze(tmp_path, {
        'pkg/base/__init__.py': 'from .model import Parent\n',
        'pkg/base/model.py': '''from torch import nn
class Parent(nn.Module):
    def forward(self, x):
        return self.child(x)
''',
        'model.py': '''from torch import nn
from pkg.base import Parent
class Child(nn.Module):
    def forward(self, x):
        return x
class Model(Parent):
    def __init__(self):
        super().__init__()
        self.child = Child()
''',
    })
    root = next(m for m in index['modules'] if m['qualname'] == 'model.Model')
    assert root['forward_origin'] == 'inherited'
    assert root['forward_owner_definition_id'] != root['definition_id']
    assert len(root['inherited_base_definition_ids']) == 1
    assert len(root['nested_forward_calls']) == 1
    assert root['nested_forward_calls'][0]['resolution'] == 'conditional'
    assert root['forward_calls'][0]['callee_expression'] == 'self.child'
    assert len([e for e in ir['edges'] if e['relation'] == 'USES_INHERITED_FORWARD']) == 1
    nested = [e for e in ir['edges'] if e['relation'] == 'STATIC_NESTED_FORWARD_INVOCATION']
    assert len(nested) == 1 and nested[0]['source_refs']
    view = present(out, tmp_path / 'nested', view='nested', level='block', format='svg')
    assert view['render_allowed']
    manifest = json.loads((tmp_path / 'nested/presentation-manifest.json').read_text())
    visuals = [e for e in manifest['visual_links'].values()
               if e['relation_kind'] == 'STATIC_NESTED_FORWARD_INVOCATION']
    assert len(visuals) == 1 and visuals[0]['ir_edge_ids'] == [nested[0]['id']]
    old = present(out, tmp_path / 'flow', view='flow', level='block')
    assert old['render_allowed']
    old_links = json.loads((tmp_path / 'flow/presentation-manifest.json').read_text())['visual_links']
    assert not any(v['relation_kind'] == 'STATIC_NESTED_FORWARD_INVOCATION'
                   for v in old_links.values())


def test_multiple_inheritance_does_not_guess_method_owner(tmp_path):
    _, index, _ = analyze(tmp_path, {'model.py': '''from torch import nn
class A(nn.Module):
    def forward(self, x): return x
class B(nn.Module):
    def forward(self, x): return x
class Model(A, B):
    pass
'''})
    root = next(m for m in index['modules'] if m['qualname'].endswith('Model'))
    assert root['forward_method_id'] is None
    assert root['forward_origin'] == 'unsupported-or-multiple-bases'


def test_missing_imported_base_is_unresolved(tmp_path):
    _, index, _ = analyze(tmp_path, {'model.py': '''from torch import nn
from external.package import Parent
class Model(Parent):
    pass
'''})
    root = next(m for m in index['modules'] if m['qualname'].endswith('Model'))
    assert root['forward_method_id'] is None


def test_nested_container_templates_preserve_unknown_loop_count(tmp_path):
    out, index, ir = analyze(tmp_path, {'model.py': '''from torch import nn
class Block(nn.Module):
    def forward(self, x): return x
class Model(nn.Module):
    def __init__(self, counts):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(counts):
            block = Block()
            self.blocks.append(block)
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
'''})
    root = next(m for m in index['modules'] if m['qualname'].endswith('Model'))
    assert len(root['nested_module_templates']) == 1
    template = root['nested_module_templates'][0]
    assert template['count'] is None
    assert template['repeat_expressions'] == ['range(counts)']
    assert template['container_expression'] == 'self.blocks'
    assert len(index['modules']) == 2
    assert len([e for e in ir['edges'] if e['relation'] == 'HAS_NESTED_MODULE_TEMPLATE']) == 1
    result = present(out, tmp_path / 'structure', view='structure', level='block')
    assert result['render_allowed']
    result = present(out, tmp_path / 'nested', view='nested', level='operation')
    assert result['status'] == 'partial'
    health = json.loads((tmp_path / 'nested/presentation-health.json').read_text())
    assert 'DYNAMIC_MODULE_TEMPLATES' in [i['code'] for i in health['issues']]


def test_torchscript_forward_overload_stubs_are_not_runtime_methods(tmp_path):
    _, index, _ = analyze(tmp_path, {'model.py': '''import torch
from torch import nn
class Model(nn.Module):
    @torch.jit._overload_method
    def forward(self, x: list): pass
    @torch.jit._overload_method
    def forward(self, x: tuple): pass
    def forward(self, x): return x
'''})
    m = index['modules'][0]
    assert m['forward_origin'] == 'declared'
    assert m['forward_method_id'] is not None


def test_dynamic_forward_definitions_do_not_choose_last_arbitrarily(tmp_path):
    _, index, _ = analyze(tmp_path, {'model.py': '''from torch import nn
class Model(nn.Module):
    def forward(self, x): return x
    def forward(self, y): return y
'''})
    assert index['modules'][0]['forward_method_id'] is None


def test_unrelated_append_is_not_a_module_template(tmp_path):
    _, index, ir = analyze(tmp_path, {'model.py': '''from torch import nn
class Model(nn.Module):
    def __init__(self):
        self.data = []
        value = nn.ReLU()
        self.data.append(value)
    def forward(self, x): return x
'''})
    assert index['modules'][0]['nested_module_templates'] == []
    assert not any(e['relation'] == 'HAS_NESTED_MODULE_TEMPLATE' for e in ir['edges'])
