"""Source-only full graph gate, including conservative failure cases."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ae.sae.api import extract, validate_bundle
from ae.presenter.api import present
from ae.sae.analysis.instance_forward import expand_factory_forward


def _resnet_source():
    resnet=pytest.importorskip('torchvision.models.resnet')
    return Path(resnet.__file__).read_text(),Path(resnet.__file__).name


def _extract(tmp_path,source,*,shape=(1,3,64,64)):
    src=tmp_path/'source';src.mkdir(parents=True)
    (src/'resnet.py').write_text(source)
    bundle=tmp_path/'sae'
    extract(src,bundle,entrypoint='resnet:resnet18',input_shape=shape)
    result=json.loads((bundle/'instance-forward.json').read_text())
    assert validate_bundle(bundle)['valid']
    return result,bundle


def test_full_resnet18_static_ir_and_presentation(tmp_path):
    source,_=_resnet_source()
    data,bundle=_extract(tmp_path,source)
    assert data['status']=='expanded_static'
    assert (len(data['nodes']),len(data['edges']))==(71,78)
    assert data['output_shape']==[1,1000]
    targets=[n['target'] for n in data['nodes']]
    assert sum(t=='add' for t in targets)==8
    assert sum('.downsample.' in t for t in targets)==6
    assert sum(t=='relu' or t.endswith('.relu') for t in targets)==17 # 8 twice plus the stem
    assert len([n for n in data['nodes'] if n['op']=='call_module'])==60
    relu=[n for n in data['nodes'] if n['target']=='layer1.0.relu']
    assert len(relu)==2 and len({n['id'] for n in relu})==2
    assert len({n['module_instance'] for n in relu})==1
    ir=json.loads((bundle/'architecture.ir.json').read_text())
    assert sum(n['kind']=='instance_forward_operation' for n in ir['nodes'])==71
    assert sum(e['relation']=='STATIC_INSTANCE_TENSOR_DEPENDENCY' for e in ir['edges'])==78
    out=tmp_path/'figures'
    response=present(bundle,out,view='instance',level='operation',format='all')
    assert response['status']=='partial' # independent runtime is deliberately separate
    manifest=json.loads((out/'presentation-manifest.json').read_text())
    assert len(manifest['visual_elements'])==71
    assert len(manifest['visual_links'])==78
    assert all((out/f'architecture.{ext}').stat().st_size>0 for ext in ('svg','pdf','tex'))
    assert {n['id'] for n in data['nodes']}==set(manifest['visual_elements'])
    assert {e['id'] for e in data['edges']}==set(manifest['visual_links'])
    health=json.loads((out/'presentation-health.json').read_text())
    assert health['status']=='partial'


@pytest.mark.parametrize('old,new',[ 
    ('kernel_size=7, stride=2, padding=3','kernel_size=5, stride=2, padding=3'),
    ('if self.downsample is not None:', 'if True:'),
    ('downsample = nn.Sequential(\n                conv1x1(self.inplanes, planes * block.expansion, stride),\n                norm_layer(planes * block.expansion),\n            )',
     'downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride))'),
    ('out += identity','out *= identity'),
])
def test_source_drift_is_not_assumed_valid(tmp_path,old,new):
    source,_=_resnet_source()
    assert old in source
    data,_=_extract(tmp_path,source.replace(old,new))
    assert data['status']!='expanded_static',data


def test_invalid_input_is_not_expanded(tmp_path):
    source,_=_resnet_source()
    for shape in ((1,4,64,64),(0,3,64,64),(1,3,2),(-1,3,64,64)):
        target=tmp_path/('c'+str(abs(hash(shape))))
        data,_=_extract(target,source,shape=shape)
        assert data['status']=='unsupported'


def test_no_factory_is_not_claimed_as_complete():
    assert expand_factory_forward([],None,(1,3,64,64),[])['status']=='unsupported'
    assert expand_factory_forward([],None,None,[]) is None
