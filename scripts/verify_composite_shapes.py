"""Independent FX, runtime, and unchanged-source shape checks."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import torchvision
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from ae.sae.api import extract, validate_bundle
from ae.presenter.api import present

CASES = {
    'matmul': ('''import torch
class Model(torch.nn.Module):
 def forward(self, x):
  a=torch.ones(2,3)
  b=torch.ones(3,4)
  return torch.matmul(a,b)
''', 'torch.matmul', (2, 4)),
    'batched-matmul': ('''import torch
class Model(torch.nn.Module):
 def forward(self, x):
  a=torch.ones(2,1,3,4)
  b=torch.ones(1,5,4,6)
  return a @ b
''', 'a @ b', (2, 5, 3, 6)),
    'cat': ('''import torch
class Model(torch.nn.Module):
 def forward(self, x):
  a=torch.ones(2,3)
  b=torch.ones(2,5)
  return torch.cat([a,b],dim=-1)
''', 'torch.cat', (2, 8)),
    'reshape': ('''import torch
class Model(torch.nn.Module):
 def forward(self, x):
  a=torch.ones(2,3,4)
  return a.reshape(6,-1)
''', 'a.reshape', (6, 4)),
    'shared-linear-cat': ('''import torch
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__()
  self.a=torch.nn.Linear(4,7)
  self.b=torch.nn.Linear(4,5)
 def forward(self,x):
  a=self.a(x)
  b=self.b(x)
  return torch.cat([a,b],dim=-1)
''', 'torch.cat', ('...', 12)),
}


def check_case(case, content, call_expr, expected, root):
    folder = root / case
    src = folder / 'source'
    src.mkdir(parents=True, exist_ok=True)
    file = src / 'model.py'
    file.write_text(content)
    before = hashlib.sha256(file.read_bytes()).hexdigest()
    extract(src, folder / 'sae', entrypoint='model:Model')
    assert validate_bundle(folder / 'sae')['valid']
    ir = json.loads((folder / 'sae/architecture.ir.json').read_text())
    shape_nodes = {n['attributes']['value_id']:n['attributes']['dimensions'] for n in ir['nodes']
                   if n['kind'] == 'symbolic_shape_fact'}
    operation_nodes = [n for n in ir['nodes'] if
                       (n['kind'] == 'call_site' and n['attributes'].get('callee_expression') == call_expr) or
                       (n['kind'] == 'operation_site' and n['attributes'].get('expression') == call_expr)]
    assert len(operation_nodes) == 1, (case, len(operation_nodes))
    actual_shape = shape_nodes[operation_nodes[0]['id']]
    module_name = f'ae_shape_ref_{case.replace("-", "_")}'
    spec = importlib.util.spec_from_file_location(module_name, file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    instance = mod.Model().eval()
    sample = torch.ones(1, 4)
    with torch.inference_mode():
        result = instance(sample)
    runtime_dims = tuple(result.shape)
    assert runtime_dims == ((1, 12) if case == 'shared-linear-cat' else expected)
    traced = symbolic_trace(instance)
    ShapeProp(traced).propagate(sample)
    fx_out = next(n for n in traced.graph.nodes if n.op == 'output')
    fx_meta = fx_out.meta['tensor_meta']
    fx_dims = tuple(fx_meta.shape)
    assert fx_dims == runtime_dims
    if expected[0] == '...':
        assert actual_shape == ['...', str(expected[-1])]
        assert runtime_dims[-1] == int(actual_shape[-1])
    else:
        assert actual_shape == list(map(str, runtime_dims))
    assert hashlib.sha256(file.read_bytes()).hexdigest() == before
    return {'case':case, 'source_sha256':before, 'ae_shape':actual_shape,
            'fx_shape':list(fx_dims), 'runtime_shape':list(runtime_dims),
            'ae_fx_runtime_agreement':True}


def real_model_checks(root):
    from torchvision import models
    import torchvision.models.densenet as densenet
    import torchvision.models.resnet as resnet
    source_root=root/'original-torchvision'/'models'
    source_root.mkdir(parents=True,exist_ok=True)
    for filename in (densenet.__file__,resnet.__file__):
        data=Path(filename).read_bytes()
        (source_root/Path(filename).name).write_bytes(data)
    results=[]
    for name, source_path in [('resnet18','models.resnet:resnet18'),
                              ('densenet121','models.densenet:densenet121')]:
        base=root/'original-torchvision'/name
        extract(source_root.parent,base/'sae',entrypoint=source_path)
        assert validate_bundle(base/'sae')['valid']
        present(base/'sae',base/'figure',view='flow',level='operation',format='svg')
        ir=json.loads((base/'sae/architecture.ir.json').read_text())
        health=json.loads((base/'figure/presentation-health.json').read_text())
        model=getattr(models,name)(weights=None).eval()
        fx=symbolic_trace(model)
        reference_count=len(list(fx.graph.nodes))
        with torch.inference_mode():
            output=model(torch.ones(1,3,64,64))
        assert tuple(output.shape)==(1,1000)
        assert health['status']=='partial'
        results.append({'model':name, 'version':torchvision.__version__,
                        'fx_nodes':reference_count,'ae_ir_nodes':len(ir['nodes']),
                        'ae_composite_facts':sum(n['kind']=='symbolic_shape_fact' and
                                n.get('analysis_rule')=='concrete-torch-operations/v1' for n in ir['nodes']),
                        'health':health['status'], 'runtime_output_shape':list(output.shape),
                        'full_forward_equivalence':False})
    return results


def main():
    torch.set_num_threads(1)
    root=Path('benchmark-output/composite-v013')
    root.mkdir(parents=True,exist_ok=True)
    results=[check_case(name,*case,root) for name,case in CASES.items()]
    actual_real=real_model_checks(root)
    # Reproducibility check on a case with derived symbolic-concat fact.
    ir_hash=[]
    src=root/'shared-linear-cat'/'source'
    for seed in ('17','1009'):
        target=root/'repro'/seed
        env=os.environ.copy();env['PYTHONHASHSEED']=seed
        subprocess.run([sys.executable,'-m','ae.cli','extract',str(src),'--entrypoint','model:Model',
                        '--output',str(target)],env=env,check=True,capture_output=True,text=True)
        ir_hash.append(hashlib.sha256((target/'architecture.ir.json').read_bytes()).hexdigest())
    assert ir_hash[0]==ir_hash[1]
    report={'operations':results,'real_models':actual_real,'hash_seed_deterministic':True,
            'operation_matches':len(results),'full_model_coverage_verified':False,
            'torch':torch.__version__,'torchvision':torchvision.__version__}
    (root/'verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
