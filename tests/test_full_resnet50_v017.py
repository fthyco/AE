"""ResNet50 source expansion and verified construction constraints."""
from pathlib import Path
import json
import pytest
from ae.sae.api import extract,validate_bundle
from ae.presenter.api import present


def _extract(root,content,shape=(1,3,64,64)):
    source=root/'source';source.mkdir(parents=True)
    (source/'resnet.py').write_text(content)
    bundle=root/'sae'
    extract(source,bundle,entrypoint='resnet:resnet50',input_shape=shape)
    assert validate_bundle(bundle)['valid']
    return json.loads((bundle/'instance-forward.json').read_text()),bundle


def test_resnet50_full_static_nodes_and_shapes(tmp_path):
    torchvision=pytest.importorskip('torchvision.models.resnet')
    graph,bundle=_extract(tmp_path,Path(torchvision.__file__).read_text())
    assert graph['status']=='expanded_static'
    assert (len(graph['nodes']),len(graph['edges']))==(177,192)
    assert graph['output_shape']==[1,1000]
    paths=[n['target'] for n in graph['nodes']]
    assert paths.count('add')==16
    assert sum('.downsample.' in p for p in paths)==8
    assert sum(p.endswith('.relu') or p=='relu' for p in paths)==49
    assert sum(n['op']=='call_module' for n in graph['nodes'])==158
    ir=json.loads((bundle/'architecture.ir.json').read_text())
    assert sum(n['kind']=='instance_forward_operation' for n in ir['nodes'])==177
    assert sum(e['relation']=='STATIC_INSTANCE_TENSOR_DEPENDENCY' for e in ir['edges'])==192
    view=present(bundle,tmp_path/'figure',view='instance',level='operation',format='svg')
    assert view['status']=='partial'
    manifest=json.loads((tmp_path/'figure/presentation-manifest.json').read_text())
    assert len(manifest['visual_elements'])==177
    assert len(manifest['visual_links'])==192


@pytest.mark.parametrize('old,new',[
    ('width = int(planes * (base_width / 64.0)) * groups',
     'width = int(planes * (base_width / 32.0)) * groups'),
    ('out = self.conv3(out)', 'out = self.conv2(out)'),
    ('if self.downsample is not None:', 'if True:'),
    ('self.conv3 = conv1x1(width, planes * self.expansion)',
     'self.conv3 = conv3x3(width, planes * self.expansion)'),
])
def test_bottleneck_source_drift_fails_closed(tmp_path,old,new):
    torchvision=pytest.importorskip('torchvision.models.resnet')
    src=Path(torchvision.__file__).read_text()
    assert old in src
    graph,_=_extract(tmp_path,src.replace(old,new))
    assert graph['status']=='unsupported'
