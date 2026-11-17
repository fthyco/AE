"""Independent TorchVision DenseNet registration verification."""
from __future__ import annotations

import hashlib
import inspect
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torchvision
from torchvision import models
from torch.fx import symbolic_trace
from ae.sae.api import extract, validate_bundle
from ae.presenter.api import present


CASES = {
    'densenet121': (6, 12, 24, 16),
    'densenet161': (6, 12, 36, 24),
    'densenet169': (6, 12, 32, 32),
    'densenet201': (6, 12, 48, 32),
}


def main():
    torch.set_num_threads(1)
    origin = Path(importlib.import_module("torchvision.models.densenet").__file__).resolve()
    source = Path('benchmark-output/torchvision-source/models')
    source.mkdir(parents=True, exist_ok=True)
    dest = source / origin.name
    raw = origin.read_bytes()
    dest.write_bytes(raw)
    assert dest.read_bytes() == raw
    checks = []
    for name, expected in CASES.items():
        bundle = Path('benchmark-output/containers') / name / 'sae'
        start = time.perf_counter()
        extract(source.parent, bundle, entrypoint=f'models.densenet:{name}')
        elapsed = round(time.perf_counter() - start, 5)
        assert validate_bundle(bundle)['valid']
        data = json.loads((bundle / 'model-index.json').read_text())
        spec = data['container_specialization']
        assert tuple(spec['stage_counts']) == expected
        assert spec['total_members'] == sum(expected)
        with torch.inference_mode():
            model = getattr(models, name)(weights=None)
            model.eval()
            modules = dict(model.named_modules())
            observed = tuple(sum(key.startswith(f'features.denseblock{i}.denselayer')
                                 and key.count('.') == 2 for key in modules)
                             for i in range(1, 5))
            assert observed == expected, (name, observed, expected)
            assert sum(key.startswith('features.transition') and key.count('.') == 1
                       for key in modules) == 3
            fx = symbolic_trace(model)
            count_cat = sum(n.op == 'call_function' and getattr(n.target, '__name__', '') == 'cat'
                            for n in fx.graph.nodes)
            assert count_cat == sum(expected) + len(expected)
            if name == 'densenet121':
                for side in (64, 96):
                    output = model(torch.zeros(1, 3, side, side))
                    assert list(output.shape) == [1, 1000]
        fig_root = bundle.parent / 'figure'
        present(bundle, fig_root, view='structure', level='operation', format='svg')
        manifest = json.loads((fig_root / 'presentation-manifest.json').read_text())
        indexed = {x for item in manifest['visual_elements'].values() for x in item['ir_node_ids']}
        for stage in spec['stages']:
            assert set(stage['member_instance_ids']).issubset(indexed)
        checks.append({'model': name, 'source_version': torchvision.__version__,
                       'sha256': hashlib.sha256(raw).hexdigest(), 'ae_counts': list(expected),
                       'runtime_counts': list(observed), 'members': sum(expected),
                       'fx_cat': count_cat, 'extract_seconds': elapsed,
                       'shape_64_96_verified': name == 'densenet121',
                       'full_forward_equivalence': False})
    repro = Path('benchmark-output/determinism')
    digests = []
    for seed in (13, 997):
        target = repro / str(seed)
        env = os.environ.copy()
        env['PYTHONHASHSEED'] = str(seed)
        subprocess.run([sys.executable, '-m', 'ae.cli', 'extract', str(source.parent),
                        '--entrypoint', 'models.densenet:densenet121', '--output', str(target)],
                       env=env, check=True, capture_output=True, text=True)
        digests.append((hashlib.sha256((target / 'architecture.ir.json').read_bytes()).hexdigest(),
                        hashlib.sha256((target / 'model-index.json').read_bytes()).hexdigest()))
    assert digests[0] == digests[1], 'IR or model-index nondeterministic across hash seeds'
    output = Path('benchmark-output/container-verification.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
