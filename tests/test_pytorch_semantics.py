"""Phase 3: PyTorch call signatures and typed static configurations."""
import json
from magg.sae import extract


def run(tmp_path, files):
    root=tmp_path/'src'; root.mkdir()
    for name, value in files.items():
        path=root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(value)
    out=tmp_path/'bundle'; extract(root,out)
    return json.loads((out/'architecture.ir.json').read_text())


def specs(ir): return [n for n in ir['nodes'] if n['kind']=='pytorch_layer_spec']


def test_linear_symbolic_and_literal_dims(tmp_path):
    ir=run(tmp_path, {'m.py':'from torch import nn\nlayer = nn.Linear(dim, 16)\n'})
    assert len(specs(ir))==1
    sh=specs(ir)[0]['attributes']['shape_template']
    assert sh['input']==['...', 'dim'] and sh['output']==['...', '16']
    assert sh['rule']=='linear-last-dimension/v1'


def test_sequential_nested_modules_not_flattened(tmp_path):
    ir=run(tmp_path, {'m.py':'import torch\nx = torch.nn.Sequential(torch.nn.Linear(8,16),torch.nn.ReLU())\n'})
    assert len(specs(ir))==3
    assert len([e for e in ir['edges'] if e['relation']=='CONTAINS_NESTED_CONSTRUCTOR'])==2


def test_external_linear_not_mistaken_for_torch(tmp_path):
    ir=run(tmp_path, {'m.py':'class Linear: pass\nx = Linear(8,16)\n'})
    assert not specs(ir)


def test_json_and_toml_tree(tmp_path):
    ir=run(tmp_path, {'config.json':'{"hidden": 128, "sizes": [4, 8]}',
                      'config.toml':'[network]\nheads = 8\n'})
    config=[n for n in ir['nodes'] if n['kind']=='configuration_entry']
    assert any(n['attributes'].get('value')==128 for n in config)
    assert any(n['attributes'].get('value')==8 for n in config)
    assert len([e for e in ir['edges'] if e['relation']=='HAS_CONFIGURATION_ENTRY'])==len(config)


def test_invalid_config_is_diagnostic_not_fake_fact(tmp_path):
    root=tmp_path/'src'; root.mkdir(); (root/'config.json').write_text('{oops: 1}')
    out=tmp_path/'out'; extract(root,out)
    diags=json.loads((out/'diagnostics.json').read_text())['diagnostics']
    assert any(d['kind']=='config_parse_failure' for d in diags)


def test_function_parameter_shadow_prevents_fake_pytorch(tmp_path):
    ir=run(tmp_path, {'m.py':'''from torch import nn
class M:
    def __init__(self, nn):
        self.fc = nn.Linear(2, 4)
'''})
    assert not specs(ir)
    assert not [n for n in ir['nodes'] if n['kind']=='module_construction_site']


def test_nonfinite_json_is_diagnostic(tmp_path):
    root=tmp_path/'src'; root.mkdir(); (root/'bad.json').write_text('{"x":NaN}')
    out=tmp_path/'out'; extract(root,out)
    diag=json.loads((out/'diagnostics.json').read_text())['diagnostics']
    assert any(d['kind']=='config_parse_failure' for d in diag)
