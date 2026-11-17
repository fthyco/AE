"""Check AE inherited forwards and nested templates against original packages."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

ROOT = Path(os.environ.get('AE_NESTED_OUTPUT', 'nested-benchmark-output')).resolve()


def checkout_exact(package, relative_files: list[str], destination: Path) -> dict[str, str]:
    """Copy original package source bytes without rewriting them."""
    base = Path(package.__file__).parent.resolve()
    hashes: dict[str, str] = {}
    for relative in relative_files:
        source = base / relative
        raw = source.read_bytes()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        assert target.read_bytes() == raw
        hashes[relative] = hashlib.sha256(raw).hexdigest()
    return hashes


def extract_deterministic(source: Path, entrypoint: str, out: Path) -> bool:
    for seed in ('17', '999'):
        command = [sys.executable, '-m', 'ae.cli', 'extract', str(source), '--entrypoint', entrypoint,
                   '--output', str(out / ('sae-' + seed))]
        env = dict(os.environ, PYTHONHASHSEED=seed)
        subprocess.run(command, check=True, env=env, stdout=subprocess.DEVNULL)
    a = (out / 'sae-17/architecture.ir.json').read_bytes()
    b = (out / 'sae-999/architecture.ir.json').read_bytes()
    if a != b:
        raise AssertionError('IR differs across hash seeds')
    return True


def baseline(model):
    model.eval()
    torch.set_num_threads(1)
    tracer = symbolic_trace(model)
    nodes = list(tracer.graph.nodes)
    fx_categories = {}
    for n in nodes:
        fx_categories[n.op] = fx_categories.get(n.op, 0) + 1
    fx_calls = {}
    for n in nodes:
        if n.op == 'call_function':
            name = getattr(n.target, '__name__', str(n.target))
            fx_calls[name] = fx_calls.get(name, 0) + 1
    ShapeProp(tracer).propagate(torch.zeros(1, 3, 64, 64))
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 64, 64))
    return {'fx_nodes': len(nodes), 'fx_edges': sum(len(n.all_input_nodes) for n in nodes),
            'fx_categories': fx_categories,
            'fx_concat': fx_calls.get('cat', 0), 'fx_add': fx_calls.get('add', 0),
            'output_shape': list(output.shape)}


def verify(name: str, model, package, files: list[str], entrypoint: str):
    from ae.sae.api import validate_bundle
    from ae.presenter.api import present

    path = ROOT / name
    hashes = checkout_exact(package, files, path / 'source')
    assert extract_deterministic(path / 'source', entrypoint, path)
    sae = path / 'sae-17'
    assert validate_bundle(sae)['valid']
    index = json.loads((sae / 'model-index.json').read_text())
    ir = json.loads((sae / 'architecture.ir.json').read_text())
    root = next(x for x in index['modules'] if x['definition_id'] == index['root_definition_id'])
    result = present(sae, path / 'nested', view='nested', level='block', format='all')
    assert result['render_allowed'] and result['status'] == 'partial'
    present(sae, path / 'structure', view='structure', level='block', format='svg')
    view = json.loads((path / 'nested/presentation-manifest.json').read_text())
    nested_edges = [v for v in view['visual_links'].values()
                    if v['relation_kind'] == 'STATIC_NESTED_FORWARD_INVOCATION']
    templates = [(m['qualname'].split('.')[-1], index['modules'], t)
                 for m in index['modules'] for t in m.get('nested_module_templates', [])]
    modules = {x['definition_id']: x['qualname'].split('.')[-1] for x in index['modules']}
    listed = sorted((src, modules[t['target_definition_id']], t['count']) for src, _, t in templates)
    runtime = baseline(model)
    if name == 'densenet121':
        assert root['forward_origin'] == 'declared'
        assert ('DenseNet', '_DenseBlock', None) in listed
        assert ('DenseNet', '_Transition', None) in listed
        assert ('_DenseBlock', '_DenseLayer', None) in listed
        assert [m['qualname'].split('.')[-1] for m in index['modules']].count('_DenseLayer') == 1
        assert len([n for n, _ in model.features.named_children() if n.startswith('denseblock')]) == 4
        assert sum(type(m).__name__ == '_DenseLayer' for m in model.modules()) == 58
    else:
        assert root['forward_origin'] == 'inherited'
        assert root['forward_owner_definition_id'] is not None
        assert root['forward_owner_definition_id'] != root['definition_id']
        assert 'base/model.py' in [r['path'] for e in ir['edges']
                                   if e['relation'] == 'STATIC_NESTED_FORWARD_INVOCATION'
                                   for r in e['source_refs']]
        assert ('UnetDecoder', 'UnetDecoderBlock', None) in listed
        assert type(model).forward is type(model).__mro__[1].forward
        assert Path(inspect.getsourcefile(type(model).forward)).name == 'model.py'
        assert len(model.decoder.blocks) == 5
        assert len(nested_edges) >= 1
    record = {
        'model': name, 'package_version': getattr(package, '__version__', None),
        'entrypoint': entrypoint, 'source_sha256': hashes,
        'source_bytes_unchanged': True, 'cross_seed_ir_identical': True,
        'ir_nodes': len(ir['nodes']), 'ir_edges': len(ir['edges']),
        'indexed_modules': len(index['modules']), 'root_forward_origin': root['forward_origin'],
        'nested_templates': listed, 'nested_invocation_edges': len(nested_edges),
        'ae_health': result['status'], 'independent': runtime,
        'full_forward_equivalence': 'not_verified',
        'runtime_instance_count_inferred_by_ae': False,
    }
    (path / 'verified.json').write_text(json.dumps(record, indent=2, sort_keys=True), encoding='utf-8')
    print('NESTED_RESULT', json.dumps({k: record[k] for k in [
        'model', 'indexed_modules', 'root_forward_origin', 'nested_templates',
        'nested_invocation_edges', 'ae_health', 'cross_seed_ir_identical']}, sort_keys=True), flush=True)


def main():
    import torchvision
    from torchvision import models
    import segmentation_models_pytorch as smp

    verify('densenet121', models.densenet121(weights=None), torchvision,
           ['models/densenet.py', 'models/__init__.py'], 'models.densenet:DenseNet')
    verify('unet_resnet34', smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1), smp,
           ['base/__init__.py', 'base/model.py', 'base/heads.py', 'base/modules.py',
            'base/hub_mixin.py', 'decoders/__init__.py', 'decoders/unet/__init__.py',
            'decoders/unet/model.py', 'decoders/unet/decoder.py',
            'encoders/__init__.py', 'encoders/_base.py', 'encoders/resnet.py'],
           'decoders.unet.model:Unet')


if __name__ == '__main__':
    main()
