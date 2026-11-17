"""Instance-aware FX matching regression tests."""
import copy

import pytest

from ae.verification.instance_correspondence import compare_factory_instances


def case():
    source = 'resnet.py'
    ir = {
        'nodes': [
            {'id': 'block0', 'kind': 'factory_block_instance', 'attributes':
                {'stage': 'layer1', 'position': 0, 'block_type': 'BasicBlock'},
             'source_refs': [{'path': source, 'start_line': 20}]},
            {'id': 'block1', 'kind': 'factory_block_instance', 'attributes':
                {'stage': 'layer1', 'position': 1, 'block_type': 'BasicBlock'},
             'source_refs': [{'path': source, 'start_line': 20}]},
            {'id': 'conv', 'kind': 'call_site', 'attributes': {'callee_expression': 'self.conv1'},
             'source_refs': [{'path': source, 'start_line': 90, 'start_column': 4,
                              'end_column': 22}]},
            {'id': 'relu', 'kind': 'call_site', 'attributes': {'callee_expression': 'self.relu'},
             'source_refs': [{'path': source, 'start_line': 91, 'start_column': 4,
                              'end_column': 20}]},
        ],
        'edges': [{'source': 'conv', 'target': 'relu', 'relation': 'STATIC_VALUE_DEPENDENCY'}],
    }
    index = {'factory_specialization': {
        'resolution': 'conditional', 'factory_definition_id': 'f',
        'block_type': 'BasicBlock', 'stages': [
            {'name': 'layer1', 'block_type': 'BasicBlock',
             'block_instance_ids': ['block0', 'block1']}]}}

    def event(name, position, operation, inputs=()):
        return {'name': name, 'op': 'call_module', 'target_name': f'layer1.{position}.{operation}',
                'source': {'path': source, 'line': 90 if operation == 'conv1' else 91,
                           'column': 4, 'end_column': 22 if operation == 'conv1' else 20},
                'module_stack': [{'path': f'layer1.{position}', 'type_name': 'BasicBlock'}],
                'input_nodes': list(inputs)}

    fx = [event('conv0', 0, 'conv1'), event('relu0', 0, 'relu', ['conv0']),
          event('conv1', 1, 'conv1', ['relu0']), event('relu1', 1, 'relu', ['conv1'])]
    runtime = {'layer1.0': 'BasicBlock', 'layer1.1': 'BasicBlock'}
    return ir, index, fx, runtime


def audit(ir, index, fx, runtime=None):
    return compare_factory_instances(ir, index, fx, runtime_modules=runtime)


def test_matches_distinct_instances_and_intrablock_edges():
    ir, index, fx, runtime = case()
    r = audit(ir, index, fx, runtime)
    assert r['status'] == 'verified_in_scope'
    assert r['matched_instance_operations'] == r['observed_in_scope_fx_operations'] == 4
    assert r['block_sequence_matches']
    assert len(set(x['instance_id'] for x in r['matched_events'])) == 2
    assert len(set(x['source_site_id'] for x in r['matched_events'])) == 2
    assert r['direct_edge_correspondence']['true_positive'] == 2
    assert r['direct_edge_correspondence']['recall'] == 1
    assert r['direct_edge_correspondence']['cross_block_fx_edges_excluded'] == 1
    assert r['ae_to_fx_edge_equivalence'] == 'not_verified'
    assert r['full_equivalence'] == 'not_verified'


def test_determinism_does_not_depend_on_ir_node_order():
    ir, index, fx, runtime = case()
    original = audit(ir, index, fx, runtime)
    ir['nodes'].reverse()
    ir['edges'].reverse()
    assert audit(ir, index, fx, runtime) == original


def test_missing_instance_stays_partial():
    ir, index, fx, runtime = case()
    r = audit(ir, index, fx[:2], runtime)
    assert r['status'] == 'partial'
    assert r['missing_instance_paths'] == ['layer1.1']
    assert 'MISSING_BLOCK_OCCURRENCES' in r['issues']


def test_wrong_order_stays_partial():
    ir, index, fx, runtime = case()
    r = audit(ir, index, fx[2:] + fx[:2], runtime)
    assert not r['block_sequence_matches']
    assert r['status'] == 'partial'


def test_wrong_fx_class_is_not_attached_to_blueprint():
    ir, index, fx, runtime = case()
    fx[0]['module_stack'][0]['type_name'] = 'Bottleneck'
    r = audit(ir, index, fx, runtime)
    assert 'BLOCK_TYPE_MISMATCH' in r['issues']
    assert 'conv0' not in {m['fx_event'] for m in r['matched_events']}


def test_wrong_runtime_type_rejected():
    ir, index, fx, runtime = case()
    runtime['layer1.0'] = 'Bottleneck'
    r = audit(ir, index, fx, runtime)
    assert 'MODULE_IDENTITY_MISMATCH' in r['issues']


def test_repeated_site_in_one_instance_not_duplicated():
    ir, index, fx, runtime = case()
    copy_event = copy.deepcopy(fx[0]);copy_event['name'] = 'conv0_again'
    fx.insert(1, copy_event)
    r = audit(ir, index, fx, runtime)
    assert 'AMBIGUOUS_INSTANCE_EVENT' in r['issues']
    assert 'conv0' not in {m['fx_event'] for m in r['matched_events']}
    assert 'conv0_again' not in {m['fx_event'] for m in r['matched_events']}


def test_unknown_static_site_does_not_count_as_match():
    ir, index, fx, runtime = case()
    fx[0]['source']['line'] = 999
    r = audit(ir, index, fx, runtime)
    assert r['unmatched_operations']
    assert r['status'] == 'partial'


def test_wrong_span_rejected():
    ir, index, fx, runtime = case()
    fx[0]['source']['column'] = 99
    r = audit(ir, index, fx, runtime)
    assert r['matched_instance_operations'] == 3


def test_direct_edge_disagreement_is_reported():
    ir, index, fx, runtime = case()
    ir['edges'].clear()
    r = audit(ir, index, fx, runtime)
    assert r['direct_edge_correspondence']['false_negative'] == 2
    assert r['direct_edge_correspondence']['recall'] == 0
    assert 'DIRECT_INTRA_BLOCK_EDGE_MISMATCH' in r['issues']


def test_no_factory_requires_no_guesses():
    ir, index, fx, runtime = case()
    index.pop('factory_specialization')
    r = audit(ir, index, fx, runtime)
    assert r['status'] == 'unsupported'
    assert r['full_equivalence'] == 'not_verified'


def test_invalid_factory_instance_record_rejected():
    ir, index, fx, runtime = case()
    ir['nodes'][0]['attributes']['position'] = 42
    r = audit(ir, index, fx, runtime)
    assert 'INVALID_BLOCK_BLUEPRINT' in r['issues']
    assert r['status'] == 'partial'


def test_duplicate_fx_names_rejected():
    ir, index, fx, runtime = case()
    fx[1]['name'] = fx[0]['name']
    with pytest.raises(ValueError, match='Duplicate FX event'):
        audit(ir, index, fx, runtime)


def test_no_runtime_or_shape_proof_claimed():
    ir, index, fx, runtime = case()
    r = audit(ir, index, fx)
    assert not r['runtime_module_type_checks']
    assert r['shape_checks'] == []
    assert r['shape_mismatches'] == 0
    assert r['full_equivalence'] == 'not_verified'
