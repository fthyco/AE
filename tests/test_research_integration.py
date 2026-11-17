"""Phase 5 multi-file source-only model smoke fixture.

This is *not* a substitute for validation against unmodified research repositories.
"""
import json
from ae.sae import extract, validate_bundle


def test_nested_attention_residual_and_shared_custom_module(tmp_path):
    root = tmp_path/'project'; root.mkdir()
    (root/'blocks.py').write_text('''from torch import nn
class Encoder(nn.Module):
    def __init__(self, dim):
        self.pipe = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.attn = nn.MultiheadAttention(dim, 4)
    def forward(self, x):
        h = self.pipe(x)
        return h + x
''')
    (root/'model.py').write_text('''from torch import nn
from blocks import Encoder
class Network(nn.Module):
    def __init__(self, d):
        self.block = Encoder(d)
    def forward(self, x, repeat):
        h = self.block(x)
        if repeat:
            h = self.block(h)
        return h
''')
    out=tmp_path/'bundle'
    extract(root,out,entrypoint='model:Network')
    assert validate_bundle(out)['valid']
    ir=json.loads((out/'architecture.ir.json').read_text())
    relations=[edge['relation'] for edge in ir['edges']]
    assert relations.count('CONSTRUCTS_LOCAL_MODULE') == 1
    assert relations.count('USES_MODULE_REFERENCE') == 3
    assert relations.count('CONTAINS_NESTED_CONSTRUCTOR') == 2
    assert relations.count('DESCRIBES_PYTORCH_CALL') == 4  # Sequential, Linear, GELU, MHA
    assert any(n['kind']=='operation_site' and n['attributes'].get('operator')=='Add' for n in ir['nodes'])
    assert ir['entrypoint_target_id']
    assert not any(e['relation']=='RUNTIME_TENSOR_FLOW' for e in ir['edges'])
