"""Bounded SMP U-Net ResNet34 static expansion; fail-closed regression suite."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ae.sae.api import extract,validate_bundle
from ae.presenter.api import present


@pytest.fixture(scope='module')
def original_files():
    smp=pytest.importorskip('segmentation_models_pytorch')
    from scripts.verify_full_unet_resnet34 import snapshot_sources
    # Real package bytes are read but no model forward is run in extraction tests.
    return {'root':smp,'copy':snapshot_sources}


def _prepare(tmp_path,source_factory,shape=(1,3,64,64),
             *,change=None,missing=None,entrypoint='decoders.unet.model:Unet'):
    src=tmp_path/'src'
    source_factory(src)
    if change is not None:
        file,old,new=change
        path=src/file
        contents=path.read_text()
        assert old in contents,(file,old)
        path.write_text(contents.replace(old,new,1))
    if missing:
        (src/missing).unlink()
    out=tmp_path/'bundle'
    extract(src,out,entrypoint=entrypoint,input_shape=shape)
    assert validate_bundle(out)['valid']
    f=out/'instance-forward.json'
    return json.loads(f.read_text()) if f.is_file() else None, out


def test_full_graph_and_diagram(tmp_path,original_files):
    data,bundle=_prepare(tmp_path,original_files['copy'])
    assert data['status']=='expanded_static'
    assert len(data['nodes'])==198
    assert len(data['edges'])==227
    assert data['output_shape']==[1,1,64,64]
    assert sum(x['op']=='call_function' and x['target']=='cat' for x in data['nodes'])==4
    assert sum(x['op']=='call_function' and x['target']=='add' for x in data['nodes'])==16
    assert sum(x['op']=='call_function' and x['target']=='interpolate' for x in data['nodes'])==5
    assert sum(x['op']=='call_module' for x in data['nodes'])==149
    assert {x['target'] for x in data['nodes'] if 'encoder.layer4.' in x['target']}
    assert all(x['source_refs'] and x['source_ir_ids'] for x in data['nodes'])
    assert len({x['id'] for x in data['nodes']})==198
    r=present(bundle,tmp_path/'figure',view='instance',level='operation',format='all')
    assert r['status']=='partial' and r['render_allowed']
    man=json.loads((tmp_path/'figure/presentation-manifest.json').read_text())
    assert len(man['visual_elements'])==198 and len(man['visual_links'])==227
    assert (tmp_path/'figure/architecture.svg').stat().st_size>0
    assert (tmp_path/'figure/architecture.pdf').stat().st_size>0
    assert (tmp_path/'figure/architecture.tex').stat().st_size>0


def test_shared_relu_and_separate_execution_sites(tmp_path,original_files):
    data,_=_prepare(tmp_path,original_files['copy'])
    relus=[n for n in data['nodes'] if n['target']=='encoder.layer1.0.relu']
    assert len(relus)==2
    assert relus[0]['id']!=relus[1]['id']
    assert relus[0]['module_instance']==relus[1]['module_instance']
    assert relus[0]['source_refs'][0]['start_byte']!=relus[1]['source_refs'][0]['start_byte']


@pytest.mark.parametrize('change',[
    ('decoders/unet/decoder.py','features = features[::-1]','features = features[::1]'),
    ('decoders/unet/decoder.py','torch.cat([feature_map, skip_connection], dim=1)',
     'torch.cat([feature_map, skip_connection], dim=0)'),
    ('decoders/unet/decoder.py','mode=self.interpolation_mode','mode="bilinear"'),
    ('decoders/unet/model.py','(256, 128, 64, 32, 16)', '(256, 128, 64, 32, 32)'),
    ('encoders/resnet.py','"layers": [3, 4, 6, 3]', '"layers": [3, 4, 5, 3]'),
    ('base/model.py','features = self.encoder(x)', 'features = [x]'),
    ('base/modules.py','activation = nn.Identity() if is_inplaceabn else nn.ReLU(inplace=True)',
     'activation = nn.Identity()'),
    ('torchvision/resnet.py','out += identity','out -= identity'),
])
def test_mutated_source_rejected(tmp_path,original_files,change):
    data,_=_prepare(tmp_path,original_files['copy'],change=change)
    assert data['status']=='unsupported' and not data['nodes'] and not data['edges']
    assert data['issues']


@pytest.mark.parametrize('missing',[
    'base/model.py', 'base/modules.py', 'base/heads.py',
    'decoders/unet/model.py','decoders/unet/decoder.py',
    'encoders/resnet.py','torchvision/resnet.py'
])
def test_missing_required_source_rejected(tmp_path,original_files,missing):
    data,_=_prepare(tmp_path,original_files['copy'],missing=missing)
    assert data['status']=='unsupported' and data['nodes']==[]


@pytest.mark.parametrize('shape',[
    (1,4,64,64), (1,3,16,16), (1,3,48,64), (1,3,64,49),
    (1,3,0,64), (0,3,64,64), (1,3,-1,64), (1,3,64)
])
def test_unsupported_input_fails_closed(tmp_path,original_files,shape):
    data,_=_prepare(tmp_path,original_files['copy'],shape=shape)
    assert data['status']=='unsupported' and not data['nodes']


@pytest.mark.parametrize('shape',[(1,3,96,96),(1,3,224,224),(1,3,64,96),(2,3,64,64)])
def test_supported_input_shapes(tmp_path,original_files,shape):
    data,_=_prepare(tmp_path,original_files['copy'],shape=shape)
    assert data['status']=='expanded_static'
    assert len(data['nodes'])==198 and len(data['edges'])==227
    assert data['output_shape']==[shape[0],1,shape[2],shape[3]]


def test_nonmatching_entrypoint_does_not_claim_unet(tmp_path,original_files):
    data,_=_prepare(tmp_path,original_files['copy'],entrypoint='encoders.resnet:ResNetEncoder')
    assert data is None or data['status']!='expanded_static'


def test_publication_gate_remains_conservative(tmp_path,original_files):
    _,bundle=_prepare(tmp_path,original_files['copy'])
    result=present(bundle,tmp_path/'publication',view='instance',level='operation',
                   mode='publication',format='svg')
    assert result['status']=='partial' and not result['render_allowed']
    assert not (tmp_path/'publication/architecture.svg').exists()
