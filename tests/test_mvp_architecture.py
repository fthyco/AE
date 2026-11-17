"""MVP end-to-end validation: MLP, CNN/residual, cross-file Transformer, YAML."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ae.sae import extract, validate_bundle, inspect_bundle
from ae.sae.indexing.model_index import validate_model_index

ROOT=Path(__file__).resolve().parents[1]
EXAMPLES=ROOT/'examples'

def bundle(tmp_path,name,entry):
    out=tmp_path/'bundle'
    extract(EXAMPLES/name,out,entrypoint=entry)
    ir=json.loads((out/'architecture.ir.json').read_text())
    idx=json.loads((out/'model-index.json').read_text())
    assert validate_bundle(out)['valid']
    return ir,idx,out

def test_mlp_index_and_yaml(tmp_path):
    ir,idx,_=bundle(tmp_path,'mlp_mvp','model:MLP')
    assert idx['status']=='indexed' and len(idx['modules'])==1
    m=idx['modules'][0]
    assert [c['owner_expression'] for c in m['components']]==['self.proj','self.head']
    assert all(c['layer_spec_ids'] for c in m['components'])
    assert {c['callee_expression'] for c in m['forward_calls']}=={'F.gelu','self.proj','self.head'}
    assert sum(bool(c['module_construction_site_ids']) for c in m['forward_calls'])==2
    assert m['forward_relation_ids']
    assert any(n['kind']=='configuration_entry' and n['attributes'].get('pointer')=='/model/hidden_dim'
               and n['attributes'].get('value')==32 for n in ir['nodes'])

def test_cnn_residual_and_custom_module(tmp_path):
    ir,idx,_=bundle(tmp_path,'cnn_residual_mvp','model:CNN')
    assert {m['qualname'] for m in idx['modules']}=={'model.CNN','model.ResidualBlock'}
    main=next(m for m in idx['modules'] if m['qualname']=='model.CNN')
    block=next(m for m in idx['modules'] if m['qualname']=='model.ResidualBlock')
    assert next(c for c in main['components'] if c['owner_expression']=='self.block')['target_definition_id']==block['definition_id']
    assert any(op['operator']=='Add' for op in block['forward_operations'])
    assert any(e['relation']=='DECLARES_COMPONENT' for e in ir['edges'])

def test_transformer_cross_file_and_attention(tmp_path):
    ir,idx,_=bundle(tmp_path,'transformer_mvp','model:Transformer')
    assert {m['qualname'] for m in idx['modules']}=={'model.Transformer','model.Block','components.Attention'}
    block=next(m for m in idx['modules'] if m['qualname']=='model.Block')
    attn=next(m for m in idx['modules'] if m['qualname']=='components.Attention')
    assert next(c for c in block['components'] if c['owner_expression']=='self.attn')['target_definition_id']==attn['definition_id']
    assert {'torch.matmul','F.softmax'}.issubset({c['callee_expression'] for c in attn['forward_calls']})
    assert any(e['relation']=='CONTAINS_NESTED_CONSTRUCTOR' for e in ir['edges'])
    assert idx['unresolved']==[]
    assert any(n['kind']=='configuration_entry' and n['attributes'].get('pointer')=='/architecture/heads' for n in ir['nodes'])

def test_unknown_factory_never_becomes_guessed_module(tmp_path):
    root=tmp_path/'src';root.mkdir()
    (root/'model.py').write_text('from torch import nn\nclass Model(nn.Module):\n    def __init__(self): self.hidden=build_layer(4)\n    def forward(self,x): return self.hidden(x)\n')
    out=tmp_path/'out';extract(root,out,entrypoint='model:Model')
    idx=json.loads((out/'model-index.json').read_text())
    assert idx['modules'][0]['components']==[]
    assert any(u['kind']=='unrecognized_or_ambiguous_component' for u in idx['unresolved'])

def test_yaml_rejects_unsafe_and_duplicates(tmp_path):
    root=tmp_path/'src';root.mkdir()
    (root/'model.py').write_text('a=1\n')
    (root/'unsafe.yaml').write_text('danger: !!python/object/apply:os.system ["echo bad"]\n')
    (root/'duplicate.yml').write_text('a: 1\na: 2\n')
    (root/'multiple.yaml').write_text('a: 1\n---\nb: 2\n')
    out=tmp_path/'out';extract(root,out)
    problems=json.loads((out/'diagnostics.json').read_text())['diagnostics']
    assert {e['path'] for e in problems if e['kind']=='config_parse_failure'}=={'unsafe.yaml','duplicate.yml','multiple.yaml'}
    assert validate_bundle(out)['valid']

def test_yaml_recursive_alias_bounded(tmp_path):
    root=tmp_path/'src';root.mkdir()
    (root/'loop.yaml').write_text('cycle: &cycle [*cycle]\n')
    out=tmp_path/'out';extract(root,out)
    problems=json.loads((out/'diagnostics.json').read_text())['diagnostics']
    assert any(x['kind']=='config_expansion_limit' for x in problems)

def test_ir_index_reference_integrity(tmp_path):
    ir,idx,out=bundle(tmp_path,'mlp_mvp','model:MLP')
    idx['modules'][0]['forward_calls'][0]['call_site_id']='nonexistent'
    with pytest.raises(ValueError,match='dangling'): validate_model_index(idx,ir)
    (out/'model-index.json').write_text(json.dumps(idx))
    with pytest.raises(ValueError,match='dangling'): validate_bundle(out)

def test_deterministic_across_hash_seeds(tmp_path):
    paths=[]
    for seed in ('1','977'):
        out=tmp_path/('run'+seed)
        env={**os.environ,'PYTHONHASHSEED':seed,'PYTHONPATH':str(ROOT/'src')}
        proc=subprocess.run([sys.executable,'-m','ae.cli','extract',str(EXAMPLES/'transformer_mvp'),
                '--entrypoint','model:Transformer','--output',str(out)],env=env,capture_output=True,text=True)
        assert proc.returncode==0,proc.stderr
        paths.append(out)
    for f in ('architecture.ir.json','model-index.json','coverage.json','source-index.json'):
        assert (paths[0]/f).read_bytes()==(paths[1]/f).read_bytes()
    assert inspect_bundle(paths[0])['model_index']['module_count']==3

def test_no_entrypoint_and_unknown_entrypoint(tmp_path):
    out=tmp_path/'out'
    extract(EXAMPLES/'mlp_mvp',out)
    idx=json.loads((out/'model-index.json').read_text())
    assert idx['status']=='no_entrypoint' and idx['modules']==[]
    extract(EXAMPLES/'mlp_mvp',out,entrypoint='model:Missing')
    idx=json.loads((out/'model-index.json').read_text())
    assert idx['status']=='unresolved' and idx['unresolved']

def test_mlp_component_relations_not_just_list(tmp_path):
    ir,idx,_=bundle(tmp_path,'mlp_mvp','model:MLP')
    calls=idx['modules'][0]['forward_calls']
    sites={c['construction_site_id'] for c in idx['modules'][0]['components']}
    assert {x for c in calls for x in c['module_construction_site_ids']}==sites
    assert any(e['relation']=='DECLARES_FORWARD' for e in ir['edges'])

def test_model_index_schema(tmp_path):
    jsonschema=pytest.importorskip('jsonschema')
    _,idx,_=bundle(tmp_path,'transformer_mvp','model:Transformer')
    schema=json.loads((ROOT/'schemas/model-index.v1.schema.json').read_text())
    jsonschema.validate(idx,schema)
