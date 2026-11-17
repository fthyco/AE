"""Model composition completeness tests."""
from types import SimpleNamespace
from ae.presenter.health import evaluate


def check(modules, unresolved=(), focus=None):
    bundle = SimpleNamespace(
        ir={'entrypoint': 'model:Model'},
        model_index={'status': 'indexed', 'root_definition_id': 'r',
                     'modules': modules, 'unresolved': list(unresolved)},
    )
    graph = {'view': 'flow', 'focus': focus, 'nodes': [{'id': 'a'}, {'id': 'b'}],
             'links': [], 'issues': []}
    return evaluate(bundle, graph, mode='exploration', level='operation')


def module(components=None, calls=None):
    return {'components': components or [], 'forward_calls': calls or []}


def test_simple_model_complete():
    result = check([module([{'declared_type': 'torch.nn.Linear',
                             'construction_site_id': 'linear'}])])
    assert result['status'] == 'complete'


def test_unresolved_construction_partial():
    result = check([module()], unresolved=[{'kind': 'unrecognized_component_assignment'}])
    assert result['status'] == 'partial'
    assert 'MODEL_COMPOSITION_INCOMPLETE' in [i['code'] for i in result['issues']]


def test_opaque_sequential_partial():
    result = check([module([{'declared_type': 'torch.nn.Sequential',
                             'construction_site_id': 'seq'}])])
    assert result['status'] == 'partial'
    assert 'OPAQUE_COMPOSITE_MODULES' in [i['code'] for i in result['issues']]


def test_unexpanded_helper_partial():
    result = check([module(calls=[{'call_site_id': 'helper',
                                  'local_definition_ids': ['method'],
                                  'module_construction_site_ids': []}])])
    assert result['status'] == 'partial'
    assert 'FORWARD_HELPER_NOT_EXPANDED' in [i['code'] for i in result['issues']]


def test_focus_keeps_local_scope():
    result = check([module([{'declared_type': 'torch.nn.Sequential',
                             'construction_site_id': 'seq'}])], focus='Leaf')
    assert result['status'] == 'complete'
