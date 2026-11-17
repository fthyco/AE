"""Source-site and edge matching tests without a PyTorch dependency."""
from ae.verification.fx_correspondence import compare_source_sites


def node(ident, line, expression):
    return {'id':ident,'kind':'call_site','attributes':{'callee_expression':expression},
            'source_refs':[{'path':'model.py','start_line':line}]}


def event(ident,line,op,target,inputs=()):
    return {'name':ident,'op':op,'target_name':target,
            'source':{'path':'model.py','line':line},'input_nodes':list(inputs)}


def test_unique_sites_and_dependency_edges():
    ir={'nodes':[node('a',1,'torch.ones'),node('b',2,'torch.ones'),node('m',3,'torch.matmul')],
        'edges':[{'source':'a','target':'m','relation':'STATIC_VALUE_DEPENDENCY'},
                 {'source':'b','target':'m','relation':'STATIC_VALUE_DEPENDENCY'}]}
    fx=[event('a1',1,'call_function','ones'),event('b1',2,'call_function','ones'),
        event('mm',3,'call_function','matmul',['a1','b1'])]
    report=compare_source_sites(ir,fx)
    assert report['matched_sites']==3
    assert report['observed_site_recall']==1
    assert report['edge_true_positives']==2
    assert report['edge_precision']==report['edge_recall']==1
    assert report['full_architecture_equivalence']=='not_verified'


def test_repeated_site_preserves_multiplicity():
    ir={'nodes':[node('add',1,'torch.matmul')],'edges':[]}
    fx=[event('m1',1,'call_function','matmul'),event('m2',1,'call_function','matmul')]
    report=compare_source_sites(ir,fx)
    assert report['matched_sites']==1
    assert report['observed_fx_occurrences']==2
    assert report['repeated_sites'][0]['fx_occurrences']==2
    assert report['one_to_one_events']==0
    assert report['edge_precision'] is None


def test_unrecognized_fx_event_and_source_missing():
    ir={'nodes':[node('x',1,'torch.ones')],'edges':[]}
    fx=[event('m',2,'call_function','matmul'),
        {'name':'x','op':'call_function','target_name':'cat','source':None,'input_nodes':[]},
        event('z',3,'placeholder','x')]
    report=compare_source_sites(ir,fx)
    assert report['unmatched_fx_sites']==1
    assert report['source_location_unknown_count']==1
    assert report['observed_site_recall']==0


def test_ambiguous_static_site_not_assigned_to_fx():
    ir={'nodes':[node('a',1,'torch.ones'),node('b',1,'torch.ones')],'edges':[]}
    report=compare_source_sites(ir,[event('o',1,'call_function','ones')])
    assert report['ambiguous_sites']==1
    assert report['matched_sites']==0
    assert report['one_to_one_events']==0


def test_missing_fx_edge_is_reported():
    ir={'nodes':[node('a',1,'torch.ones'),node('m',2,'torch.matmul')],
        'edges':[{'source':'a','target':'m','relation':'STATIC_VALUE_DEPENDENCY'}]}
    fx=[event('a1',1,'call_function','ones'),event('mm',2,'call_function','matmul')]
    report=compare_source_sites(ir,fx)
    assert report['edge_false_positives']==1
    assert report['edge_precision']==0
    assert report['edge_recall'] is None


def test_omitted_source_sites_not_labelled_false_positive():
    ir={'nodes':[node('a',1,'torch.ones'),node('unused',99,'torch.ones')],'edges':[]}
    report=compare_source_sites(ir,[event('a1',1,'call_function','ones')])
    assert report['static_site_count']==2
    assert report['matched_sites']==1
    assert report['edge_false_positives']==0


def test_non_supported_name_is_excluded_from_scope():
    ir={'nodes':[node('x',1,'torch.special.dynamic_op')],'edges':[]}
    fx=[event('z',1,'call_function','dynamic_op')]
    report=compare_source_sites(ir,fx)
    assert report['observed_fx_site_count']==0
    assert report['observed_site_recall'] is None


def test_same_line_distinct_call_columns():
    a=node('a',1,'self.first');a['source_refs'][0].update(start_column=12,end_column=42,end_line=1)
    b=node('b',1,'self.second');b['source_refs'][0].update(start_column=3,end_column=43,end_line=1)
    fx=[event('a1',1,'call_module','first'),event('b1',1,'call_module','second')]
    fx[0]['source'].update(column=12,end_column=42)
    fx[1]['source'].update(column=3,end_column=43)
    report=compare_source_sites({'nodes':[a,b],'edges':[]},fx)
    assert report['matched_sites']==2
    assert report['ambiguous_sites']==0
    assert report['one_to_one_events']==2
