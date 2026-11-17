"""Dense feature-list expansion: complete proven default path or fail closed."""
import json
from pathlib import Path

import pytest

from ae.sae.api import extract,validate_bundle
from ae.presenter.api import present


def _source():
    mod=pytest.importorskip('torchvision.models.densenet')
    return Path(mod.__file__).read_text()


def _extract(tmp_path, content, shape=(1,3,64,64)):
    src=tmp_path/'source';src.mkdir(parents=True,exist_ok=True)
    (src/'densenet.py').write_text(content)
    bundle=tmp_path/'sae'
    extract(src,bundle,entrypoint='densenet:densenet121',input_shape=shape)
    assert validate_bundle(bundle)['valid']
    return json.loads((bundle/'instance-forward.json').read_text()),bundle


def test_densenet121_full_graph_and_render(tmp_path):
    data,bundle=_extract(tmp_path,_source())
    assert data['status']=='expanded_static'
    assert len(data['nodes'])==433
    assert len(data['edges'])==967
    assert data['output_shape']==[1,1000]
    assert sum(x['op']=='call_function' and x['target']=='cat' for x in data['nodes'])==62
    assert sum(x['op']=='call_module' for x in data['nodes'])==366
    assert sum('factory_instance_id' in x for x in data['nodes'])==58*7
    for x in data['nodes']:
        assert x['source_ir_ids'] and x['source_refs']
    ir=json.loads((bundle/'architecture.ir.json').read_text())
    assert sum(x['kind']=='instance_forward_operation' for x in ir['nodes'])==433
    assert sum(x['relation']=='STATIC_INSTANCE_TENSOR_DEPENDENCY' for x in ir['edges'])==967
    render=present(bundle,tmp_path/'figure',view='instance',level='operation',format='svg')
    assert render['status']=='partial'  # independent runtime evidence stays external
    manifest=json.loads((tmp_path/'figure/presentation-manifest.json').read_text())
    assert set(manifest['visual_elements'])=={x['id'] for x in data['nodes']}
    assert set(manifest['visual_links'])=={x['id'] for x in data['edges']}


@pytest.mark.parametrize('old,new',[
    ('(6, 12, 24, 16)', '(6, 12, 20, 16)'),
    ('for name, layer in self.items():', 'for name, layer in reversed(list(self.items())):'),
    ('torch.cat(inputs, 1)', 'torch.cat(inputs, 0)'),
    ('new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))',
     'new_features = self.conv2(self.norm2(self.relu2(bottleneck_output)))'),
    ('new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)',
     'new_features = F.dropout(new_features, p=1.0, training=self.training)'),
    ('out = F.adaptive_avg_pool2d(out, (1, 1))',
     'out = F.adaptive_avg_pool2d(out, (2, 2))'),
])
def test_source_drift_fails_closed(tmp_path,old,new):
    source=_source()
    assert old in source
    data,_=_extract(tmp_path,source.replace(old,new))
    assert data['status']=='unsupported'
    assert data['nodes']==[]
    assert data['edges']==[]


@pytest.mark.parametrize('shape',[(1,4,64,64),(1,3,0,64),(1,3,16,16)])
def test_unsupported_geometry_is_explicit(tmp_path,shape):
    data,_=_extract(tmp_path,_source(),shape)
    assert data['status']=='unsupported'
    assert data['nodes']==[]


def test_unrelated_factory_not_misclassified(tmp_path):
    source=_source()
    src=tmp_path/'src';src.mkdir()
    (src/'densenet.py').write_text(source)
    extract(src,tmp_path/'out',entrypoint='densenet:densenet169',input_shape=(1,3,64,64))
    data=json.loads((tmp_path/'out/instance-forward.json').read_text())
    assert data['status']=='unsupported'
