"""Phase 4: graph integrity, repeatability, and selected entrypoint provenance."""
import json
import os
import subprocess
import sys
from pathlib import Path
from magg.sae import extract, validate_bundle
from magg.sae.ir.validator import ValidationError, validate_ir
import pytest


def run(tmp_path, files, entrypoint=None):
    root=tmp_path/'repo'; root.mkdir()
    for key, content in files.items():
        path=root/key; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(content)
    out=tmp_path/'out'; extract(root,out,entrypoint=entrypoint)
    return root,out,json.loads((out/'architecture.ir.json').read_text())


def test_entrypoint_is_resolved_to_actual_definition(tmp_path):
    _,out,ir=run(tmp_path,{'model.py':'class M: pass\n'},'model:M')
    assert ir['entrypoint_target_id'] in {n['id'] for n in ir['nodes'] if n['kind']=='class_definition'}
    assert validate_bundle(out)['valid']


def test_invalid_entrypoint_not_silently_accepted(tmp_path):
    _,_,ir=run(tmp_path,{'model.py':'class M: pass\n'},'model:Other')
    assert ir['entrypoint_target_id'] is None
    assert any(u['kind']=='entrypoint_resolution' for u in ir['unresolved'])


def test_phase_relations_have_provenance(tmp_path):
    _,out,ir=run(tmp_path,{'a.py':'def func(x):\n    return x\n',
                            'b.py':'from a import func\nfrom torch import nn\n\ndef run(v):\n    z = nn.Linear(2, 4)\n    return func(v)\n'})
    interesting={'CALLS_LOCAL_DEFINITION','READS_LEXICAL_DEFINITION','DESCRIBES_PYTORCH_CALL'}
    edges=[e for e in ir['edges'] if e['relation'] in interesting]
    assert interesting <= {e['relation'] for e in edges}
    assert all(e['analysis_rule'] and e['source_refs'] for e in edges)
    assert validate_bundle(out)['valid']


def test_bytes_stable_across_python_hash_seeds(tmp_path):
    root,out,ir=run(tmp_path,{'a.py':'from torch import nn\nclass A:\n    def f(self, x):\n        y = nn.Linear(2, 4)\n        if x:\n            z = 1\n        else:\n            z = 2\n        return z\n'})
    other=tmp_path/'other'
    code='from magg.sae import extract;import sys;extract(sys.argv[1],sys.argv[2])'
    env={**os.environ,'PYTHONPATH':str(Path(__file__).parents[1]/'src'),'PYTHONHASHSEED':'9876'}
    result=subprocess.run([sys.executable,'-c',code,str(root),str(other)],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    for file in out.glob('*.json'):
        assert file.read_bytes()==(other/file.name).read_bytes()


def test_unresolved_candidate_must_exist(tmp_path):
    _,out,ir=run(tmp_path,{'m.py':'class M: pass\n'},'m:missing')
    ir['unresolved'][0]['candidate_ids']=['bogus']
    with pytest.raises(ValidationError, match='Unresolved candidate'):
        validate_ir(ir)


def test_entrypoint_target_integrity(tmp_path):
    _,out,ir=run(tmp_path,{'m.py':'class M: pass\n'},'m:M')
    ir['entrypoint_target_id']='wrong'
    with pytest.raises(ValidationError, match='Entrypoint target'):
        validate_ir(ir)
