"""Source-backed bounded container configuration contracts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ae.sae.api import extract, validate_bundle
from ae.presenter.api import present


SKELETON = '''from torch import nn

class Piece(nn.Module):
    def forward(self, x): return x

class Group(nn.ModuleDict):
    def __init__(self, n):
        super().__init__()
        for k in range(n):
            part = Piece()
            self.add_module("part%d" % (k + 1), part)
    def forward(self, x):
        for name, part in self.items(): x = part(x)
        return x

class Bridge(nn.Sequential):
    def __init__(self): super().__init__()

class Network(nn.Module):
    def __init__(self, plan=(2, 3)):
        super().__init__()
        self.sequence = nn.Sequential()
        for idx, n in enumerate(plan):
            group = Group(n)
            self.sequence.add_module("group%d" % (idx + 1), group)
            if idx != len(plan) - 1:
                bridge = Bridge()
                self.sequence.add_module("bridge%d" % (idx + 1), bridge)
    def forward(self, x): return self.sequence(x)

def build_helper(plan):
    obj = Network(plan)
    return obj

def network_variant():
    return build_helper((2, 3))
'''


def check(tmp_path, content=SKELETON, entry='model:network_variant'):
    source = tmp_path / 'source'
    source.mkdir(parents=True)
    (source / 'model.py').write_text(content)
    bundle = tmp_path / 'bundle'
    extract(source, bundle, entrypoint=entry)
    assert validate_bundle(bundle)['valid']
    return bundle, json.loads((bundle / 'model-index.json').read_text()), json.loads((bundle / 'architecture.ir.json').read_text())


def test_static_factory_expands_bounded_registration(tmp_path):
    bundle, index, ir = check(tmp_path)
    spec = index['container_specialization']
    assert spec['stage_counts'] == [2, 3]
    assert spec['total_members'] == 5
    assert spec['block_type'] == 'Group' and spec['member_type'] == 'Piece'
    assert [s['count'] for s in spec['stages']] == [2, 3]
    assert sum(s['transition_instance_id'] is not None for s in spec['stages']) == 1
    counts = {kind: sum(n['kind'] == kind for n in ir['nodes'])
              for kind in ['container_configuration', 'container_stage', 'container_block_instance',
                           'container_member_instance', 'container_transition_instance']}
    assert counts == {'container_configuration': 1, 'container_stage': 2,
                      'container_block_instance': 2, 'container_member_instance': 5,
                      'container_transition_instance': 1}
    assert len([e for e in ir['edges'] if e['relation'] == 'CONTAINER_HAS_MEMBER']) == 5
    assert all(n['source_refs'] and n['resolution'] == 'conditional'
               for n in ir['nodes'] if n['kind'].startswith('container_'))
    present(bundle, tmp_path / 'figure', view='structure', level='operation')
    manifest = json.loads((tmp_path / 'figure/presentation-manifest.json').read_text())
    rendered = {x for item in manifest['visual_elements'].values() for x in item['ir_node_ids']}
    assert all(x in rendered for stage in spec['stages'] for x in stage['member_instance_ids'])
    assert all(item['ir_edge_ids'] for item in manifest['visual_links'].values())


def test_factory_config_changes_instance_count_and_ids(tmp_path):
    _, original, _ = check(tmp_path / 'first')
    _, changed, _ = check(tmp_path / 'second', SKELETON.replace('build_helper((2, 3))', 'build_helper((1, 4))'))
    assert changed['container_specialization']['stage_counts'] == [1, 4]
    assert original['container_specialization']['configuration_id'] != changed['container_specialization']['configuration_id']


@pytest.mark.parametrize('replacement', [
    'return build_helper((2, 3))',
    'return build_helper((2, unknown))',
    'return build_helper((2, 999))',
])
def test_factory_rejects_unproven_or_oversize_counts(tmp_path, replacement):
    text = SKELETON.replace('return build_helper((2, 3))', replacement)
    _, index, _ = check(tmp_path, text)
    if replacement == 'return build_helper((2, 3))':
        assert index['container_specialization']['total_members'] == 5
    else:
        assert 'container_specialization' not in index


def test_factory_rejects_changed_loop_guard(tmp_path):
    text = SKELETON.replace('idx != len(plan) - 1', 'idx > 0')
    _, index, _ = check(tmp_path, text)
    assert 'container_specialization' not in index


def test_factory_rejects_reassigned_config(tmp_path):
    text = SKELETON.replace('obj = Network(plan)', 'plan = (99,)\n    obj = Network(plan)')
    _, index, _ = check(tmp_path, text)
    assert 'container_specialization' not in index


def test_factory_rejects_guarded_member_creation(tmp_path):
    text = SKELETON.replace('part = Piece()', 'if k > 0: break\n            part = Piece()')
    _, index, _ = check(tmp_path, text)
    assert 'container_specialization' not in index


def test_no_specialization_without_function_factory(tmp_path):
    _, index, _ = check(tmp_path, SKELETON, entry='model:Network')
    assert 'container_specialization' not in index


def test_dynamic_builder_is_not_specialized(tmp_path):
    text = SKELETON.replace('return build_helper((2, 3))', 'return build_helper((2, int(input())))')
    _, index, _ = check(tmp_path, text)
    assert 'container_specialization' not in index


def test_deterministic_ir_across_runs(tmp_path):
    a, ia, _ = check(tmp_path / 'run1')
    b, ib, _ = check(tmp_path / 'run2')
    assert (a / 'architecture.ir.json').read_bytes() == (b / 'architecture.ir.json').read_bytes()
    assert ia['container_specialization'] == ib['container_specialization']


def test_factory_rejects_container_replacement(tmp_path):
    text = SKELETON.replace('    def forward(self, x): return self.sequence(x)',
                            '        self.sequence = nn.Sequential()\n    def forward(self, x): return self.sequence(x)')
    _, index, _ = check(tmp_path, text)
    assert 'container_specialization' not in index


def test_factory_rejects_member_rebinding_before_registration(tmp_path):
    text = SKELETON.replace('            self.add_module("part%d"',
                            '            part = dynamic_part()\n            self.add_module("part%d"')
    _, index, _ = check(tmp_path, text)
    assert 'container_specialization' not in index


def test_factory_rejects_block_rebinding_before_registration(tmp_path):
    text = SKELETON.replace('            self.sequence.add_module("group%d"',
                            '            group = dynamic_group()\n            self.sequence.add_module("group%d"')
    _, index, _ = check(tmp_path, text)
    assert 'container_specialization' not in index
