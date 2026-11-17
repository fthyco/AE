"""Multi-architecture, six-view and depth regression tests; never execute analyzed models."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from magg.sae.api import extract, validate_bundle
from magg.presenter.api import present

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'tests' / 'architecture_cases'
SCENARIOS = [
    ('01_linear_chain', 'model:LinearChain'),
    ('02_nested_blocks', 'model:DeepStack'),
    ('03_residual_cnn', 'model:ResidualCNN'),
    ('04_unet_skip_concat', 'model:UNetMini'),
    ('05_attention_qkv', 'model:AttentionModel'),
    ('06_multihead_output', 'model:MultiHeadNet'),
    ('07_conditional_router', 'model:ConditionalRouter'),
    ('08_layer_reuse', 'model:ReusedLayer'),
    ('09_shape_conflict', 'model:ShapeConflict'),
    ('10_missing_dependency', 'model:MissingEncoder'),
    ('mlp_mvp', 'model:MLP'),
    ('cnn_residual_mvp', 'model:CNN'),
    ('transformer_mvp', 'model:Transformer'),
]

@pytest.mark.parametrize('name,entrypoint', SCENARIOS)
def test_all_six_levels_and_source_evidence(tmp_path, name, entrypoint):
    source = ROOT/'examples'/name if name in {'mlp_mvp','cnn_residual_mvp','transformer_mvp'} else CASES/name
    extract(source, tmp_path/'sae', entrypoint=entrypoint)
    validate_bundle(tmp_path/'sae')
    ir = json.loads((tmp_path/'sae/architecture.ir.json').read_bytes())
    nodes = {x['id'] for x in ir['nodes']}
    edges = {x['id'] for x in ir['edges']}
    for view in ('structure','flow'):
        for level in ('overview','block','operation'):
            output = tmp_path / (view+'-'+level)
            result = present(tmp_path/'sae',output,view=view,level=level,format='svg')
            assert result['status'] in {'complete','partial'}
            assert (output/'architecture.svg').stat().st_size > 200
            m = json.loads((output/'presentation-manifest.json').read_bytes())
            for entry in m['visual_elements'].values():
                assert entry['source_refs'] and entry['ir_node_ids']
                assert set(entry['ir_node_ids']) <= nodes
            for entry in m['visual_links'].values():
                assert entry['source_refs'] and entry['ir_edge_ids']
                assert set(entry['ir_edge_ids']) <= edges
    if name == '09_shape_conflict':
        assert any(x['kind']=='symbolic_shape_conflict'
                   and 'requires 20' in x['reason']
                   and 'supplies 12' in x['reason'] for x in ir['unresolved'])
    if name == '10_missing_dependency':
        health = json.loads((tmp_path/'flow-operation/presentation-health.json').read_bytes())
        assert health['status']=='partial'

@pytest.mark.parametrize('depth',[2,4,8,16])
def test_increasing_linear_depth(tmp_path,depth):
    source = tmp_path/'src'
    source.mkdir()
    code = ['from torch import nn','class Network(nn.Module):',
            '    def __init__(self):','        super().__init__()']
    code += [f'        self.layer{i}=nn.Linear(32,32)' for i in range(depth)]
    code += ['    def forward(self,x):']
    code += [f'        x=self.layer{i}(x)' for i in range(depth)]
    code += ['        return x']
    (source/'model.py').write_text('\n'.join(code)+'\n')
    extract(source,tmp_path/'sae',entrypoint='model:Network')
    ir=json.loads((tmp_path/'sae/architecture.ir.json').read_bytes())
    assert sum(n['kind']=='symbolic_shape_fact' for n in ir['nodes'])==depth+1
    result=present(tmp_path/'sae',tmp_path/'view',view='flow',level='operation')
    assert result['status']=='complete'
    manifest=json.loads((tmp_path/'view/presentation-manifest.json').read_bytes())
    assert len(manifest['visual_links'])==depth+1

def test_unparseable_entrypoint_blocks(tmp_path):
    src=tmp_path/'source'
    src.mkdir()
    (src/'model.py').write_text('class Broken(:\n pass\n')
    extract(src,tmp_path/'sae',entrypoint='model:Broken')
    result=present(tmp_path/'sae',tmp_path/'view',view='flow',level='operation')
    assert result['status']=='blocked'
    assert not result['render_allowed']
    assert not (tmp_path/'view/architecture.svg').exists()
