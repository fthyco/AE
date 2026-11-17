"""Compare static ResNet factory blueprints with unmodified TorchVision models."""
from __future__ import annotations

import hashlib
import inspect
import json
import operator
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import torch
import torchvision
from torch.fx import symbolic_trace
from torchvision import models
from torchvision.models.resnet import ResNet

from ae.sae.api import extract, validate_bundle
from ae.presenter.api import present


CASES = [('resnet18', models.resnet18, 'BasicBlock', [2, 2, 2, 2]),
         ('resnet50', models.resnet50, 'Bottleneck', [3, 4, 6, 3])]


def main() -> None:
    target = Path(os.environ.get('AE_FACTORY_OUTPUT', '/tmp/ae-factory-verified')).resolve()
    target.mkdir(parents=True, exist_ok=True)
    path = Path(inspect.getsourcefile(ResNet)).resolve()
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    results = []
    torch.set_num_threads(1)
    for name, builder, expected_type, expected_counts in CASES:
        root = target / name
        source = root / 'source' / 'models' / 'resnet.py'
        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, source)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
        started = time.perf_counter()
        extract(root / 'source', root / 'sae', entrypoint='models.resnet:' + name)
        elapsed = round(time.perf_counter() - started, 3)
        assert validate_bundle(root / 'sae')['valid']
        index = json.loads((root / 'sae' / 'model-index.json').read_text())
        ir = json.loads((root / 'sae' / 'architecture.ir.json').read_text())
        spec = index['factory_specialization']
        assert index['status'] == 'indexed'
        assert spec['block_type'] == expected_type
        assert spec['stage_counts'] == expected_counts
        assert spec['total_blocks'] == sum(expected_counts)
        assert len([n for n in ir['nodes'] if n['kind'] == 'factory_block_instance']) == sum(expected_counts)
        present(root / 'sae', root / 'overview', view='structure', level='overview', format='svg')
        present(root / 'sae', root / 'blocks', view='structure', level='block', format='all')
        flow_result = present(root / 'sae', root / 'flow', view='flow', level='block', format='svg')
        assert flow_result['status'] == 'partial'
        model = builder(weights=None).eval()
        stages = [getattr(model, f'layer{i}') for i in range(1, 5)]
        runtime_counts = [len(stage) for stage in stages]
        runtime_types = [[type(x).__name__ for x in stage] for stage in stages]
        assert runtime_counts == spec['stage_counts']
        assert all(x == expected_type for row in runtime_types for x in row)
        # FX acts as an independent residual-operation reference, not an AE graph-equivalence proof.
        graph = symbolic_trace(model).graph
        fx_residual_adds = sum(n.op == 'call_function' and n.target == operator.add for n in graph.nodes)
        assert fx_residual_adds == sum(expected_counts)
        serial_a = (root / 'sae' / 'architecture.ir.json').read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [sys.executable, '-m', 'ae.cli', 'extract', str(root / 'source'),
                   '--entrypoint', 'models.resnet:' + name, '--output', tmp]
            env = os.environ.copy()
            env['PYTHONHASHSEED'] = '779'
            subprocess.run(cmd, env=env, capture_output=True, check=True)
            assert (Path(tmp) / 'architecture.ir.json').read_bytes() == serial_a
        report = {
            'model': name, 'source_sha256': source_sha,
            'torchvision_version': torchvision.__version__,
            'source_modified': False,
            'model_weights': 'random',
            'extract_seconds': elapsed,
            'factory': spec['block_type'],
            'stage_counts': spec['stage_counts'],
            'total_blocks': spec['total_blocks'],
            'runtime_stage_counts': runtime_counts,
            'runtime_block_types': runtime_types,
            'fx_residual_adds': fx_residual_adds,
            'ir_nodes': len(ir['nodes']), 'ir_edges': len(ir['edges']),
            'health': flow_result['status'],
            'determinism_cross_seed': True,
            'full_runtime_tensor_flow_equivalence': 'not_verified',
        }
        (root / 'verification.json').write_text(json.dumps(report, indent=2, sort_keys=True))
        results.append(report)
        print('FACTORY_RESULT ' + json.dumps(report, sort_keys=True), flush=True)
    assert results[0]['stage_counts'] != results[1]['stage_counts']
    assert results[0]['factory'] != results[1]['factory']
    (target / 'summary.json').write_text(json.dumps(results, indent=2, sort_keys=True))
    print('Factory specialization: 2/2 verified; complete runtime equivalence: 0/2')


if __name__ == '__main__':
    main()
