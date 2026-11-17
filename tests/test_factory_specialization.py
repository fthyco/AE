"""Source-only factory specialization and conservative fallback tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ae.sae.api import extract, validate_bundle
from ae.presenter.api import present


MODEL = '''
import torch.nn as nn
class BasicBlock(nn.Module):
    def forward(self, x):
        return x
class Bottleneck(nn.Module):
    def forward(self, x):
        return x
class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])
    def _make_layer(self, block, planes, blocks):
        layers = []
        layers.append(block(3, planes))
        for _ in range(1, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)
def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model
def resnet18(**kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)
def resnet50(**kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)
def unresolved_factory(counts):
    return _resnet(BasicBlock, counts)
'''


def _run(tmp_path: Path, factory: str, source: str = MODEL):
    source_root = tmp_path / 'src'
    source_root.mkdir(parents=True)
    (source_root / 'model.py').write_text(source)
    output = tmp_path / 'sae'
    extract(source_root, output, entrypoint=f'model:{factory}')
    assert validate_bundle(output)['valid']
    return (json.loads((output / 'architecture.ir.json').read_text()),
            json.loads((output / 'model-index.json').read_text()), output)


@pytest.mark.parametrize(('name', 'kind', 'counts', 'total'), [
    ('resnet18', 'BasicBlock', [2, 2, 2, 2], 8),
    ('resnet50', 'Bottleneck', [3, 4, 6, 3], 16),
])
def test_factory_specializes_stage_count_and_block_class(tmp_path, name, kind, counts, total):
    ir, index, output = _run(tmp_path, name)
    config = index['factory_specialization']
    assert index['status'] == 'indexed'
    assert config['block_type'] == kind
    assert config['stage_counts'] == counts
    assert config['total_blocks'] == total
    assert len([n for n in ir['nodes'] if n['kind'] == 'factory_stage']) == 4
    assert len([n for n in ir['nodes'] if n['kind'] == 'factory_block_instance']) == total
    assert len([e for e in ir['edges'] if e['relation'] == 'STATIC_STAGE_ORDER']) == 3
    assert len([e for e in ir['edges'] if e['relation'] == 'STATIC_BLOCK_ORDER']) == total - 4
    assert all(n['source_refs'] for n in ir['nodes'] if n['kind'].startswith('factory_'))
    report = present(output, tmp_path / 'figure', view='structure', level='block', format='svg')
    assert report['render_allowed']
    manifest = json.loads((tmp_path / 'figure/presentation-manifest.json').read_text())
    assert len(manifest['visual_elements']) >= total + 4
    assert (tmp_path / 'figure/architecture.svg').is_file()
    # The stage blueprint is not a verified runtime tensor-flow graph.
    flow = present(output, tmp_path / 'flow', view='flow', level='operation', format='svg')
    assert flow['status'] == 'partial'


def test_factory_determinism(tmp_path):
    a, _, _ = _run(tmp_path / 'first', 'resnet18')
    b, _, _ = _run(tmp_path / 'second', 'resnet18')
    assert a == b


def test_unknown_repetition_is_not_guessed(tmp_path):
    ir, index, _ = _run(tmp_path, 'unresolved_factory')
    assert index['status'] == 'unsupported_entrypoint'
    assert 'factory_specialization' not in index
    assert not any(n['kind'] == 'factory_block_instance' for n in ir['nodes'])
    assert any(u['kind'] == 'factory_specialization' for u in ir['unresolved'])


def test_unsupported_loop_does_not_invent_count(tmp_path):
    wrong = MODEL.replace('range(1, blocks)', 'range(blocks)')
    ir, index, _ = _run(tmp_path, 'resnet18', wrong)
    assert index['status'] == 'unsupported_entrypoint'
    assert 'factory_specialization' not in index
    assert not any(n['kind'] == 'factory_block_instance' for n in ir['nodes'])


def test_direct_class_entrypoint_unchanged(tmp_path):
    ir, index, _ = _run(tmp_path, 'ResNet')
    assert index['status'] == 'indexed'
    assert 'factory_specialization' not in index
    assert not any(n['kind'] == 'factory_block_instance' for n in ir['nodes'])
