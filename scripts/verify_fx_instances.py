"""Verify per-block FX identities against original TorchVision ResNet source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from verify_fx_correspondence import run


def verify(output: Path) -> dict:
    verified = []
    for name, expected, matched, total, edges in (
        ('resnet18', 8, 56, 62, 48),
        ('resnet50', 16, 160, 168, 144),
    ):
        report = run(name, output/name)
        instance = report['instance_correspondence']
        assert instance['expected_block_instances'] == expected
        assert instance['fx_observed_block_instances'] == expected
        assert instance['block_sequence_matches']
        assert len(instance['runtime_module_type_checks']) == expected
        assert all(x['matched'] for x in instance['runtime_module_type_checks'])
        assert instance['observed_in_scope_fx_operations'] == total
        assert instance['matched_instance_operations'] == matched
        assert instance['direct_edge_correspondence']['true_positive'] == edges
        assert instance['direct_edge_correspondence']['false_negative'] == 0
        assert instance['direct_edge_correspondence']['false_positive'] == 0
        assert instance['status'] == 'partial'
        assert instance['issues'] == ['AMBIGUOUS_INSTANCE_EVENT']
        assert not instance['shape_checks']
        assert instance['full_equivalence'] == 'not_verified'
        ir = json.loads((output/name/'sae/architecture.ir.json').read_text())
        static = {node['id']: node for node in ir['nodes']}
        adds = [e for e in instance['matched_events']
                if static[e['source_site_id']]['kind'] == 'forward_augassign']
        assert len(adds) == expected
        assert len({x['instance_id'] for x in adds}) == expected
        verified.append({
            'model': name, 'source_sha256': report['source_hashes'],
            'expected_blocks': expected, 'matched_block_identities': len(adds),
            'matched_instance_operations': matched,
            'in_scope_fx_operations': total,
            'operation_match_fraction': matched / total,
            'verified_direct_intrablock_edges': edges,
            'fx_direct_intrablock_edges': instance['direct_edge_correspondence']['fx_edges'],
            'unsupported_composite_leaf_events': len(instance['ambiguous_operations']),
            'nested_forward_graph_equivalence': 'not_verified',
            'full_tensor_shape_equivalence': 'not_verified',
            'architectural_equivalence': 'not_verified',
        })
    summary = {'scope': 'original_torchvision_factory_instances_to_fx_events',
               'torchvision_version': report['tool_versions']['torchvision'],
               'status': 'partial', 'models': verified}
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
    print('INSTANCE_AUDIT ' + json.dumps(summary, sort_keys=True), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('benchmark-output/fx-instances'))
    options = parser.parse_args()
    verify(options.output)
