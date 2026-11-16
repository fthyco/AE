"""Phase 2 lexical dataflow, ambiguous branches, conservative non-execution."""
import json
from magg.sae import extract


def run(tmp_path, code):
    root=tmp_path/'src'; root.mkdir(); (root/'model.py').write_text(code)
    out=tmp_path/'out'; extract(root,out)
    return json.loads((out/'architecture.ir.json').read_text())


def reads(ir): return [e for e in ir['edges'] if e['relation']=='READS_LEXICAL_DEFINITION']


def test_parameters_and_linear_assignments(tmp_path):
    ir=run(tmp_path,'def forward(x):\n    a = x + 1\n    b = a * 2\n    return b\n')
    assert len(reads(ir))==3
    defs=[n for n in ir['nodes'] if n['kind']=='lexical_definition']
    assert {n['attributes']['name'] for n in defs}=={'x','a','b'}


def test_reassignments_link_latest_definition(tmp_path):
    ir=run(tmp_path,'def f(x):\n    a = x\n    a = a + 2\n    return a\n')
    assert len(reads(ir))==3
    a_defs=[n['id'] for n in ir['nodes'] if n['kind']=='lexical_definition' and n['attributes']['name']=='a']
    assert len(a_defs)==2
    assert {r['target'] for r in reads(ir) if r['target'] in a_defs}==set(a_defs)


def test_conditional_values_are_ambiguous(tmp_path):
    ir=run(tmp_path,'def f(flag):\n    if flag:\n        a = 2\n    else:\n        a = 3\n    return a\n')
    assert any(u['kind']=='ambiguous_reaching_definition' and len(u['candidate_ids'])==2 for u in ir['unresolved'])


def test_loop_may_not_execute(tmp_path):
    ir=run(tmp_path,'def f(xs):\n    for item in xs:\n        y = item\n    return y\n')
    assert any(u['kind']=='ambiguous_reaching_definition' and 'may be unbound=True' in u['reason'] for u in ir['unresolved'])


def test_nested_scope_does_not_bind_outer(tmp_path):
    ir=run(tmp_path,'def f(x):\n    def g():\n        y = 1\n        return y\n    return x\n')
    assert any(n['kind']=='lexical_definition' and n['attributes']['name']=='y' for n in ir['nodes'])
    assert len(reads(ir))==2


def test_global_declaration_not_linked_to_parameter(tmp_path):
    ir=run(tmp_path,'def f(x):\n    global v\n    return v + x\n')
    assert len(reads(ir))==1
    assert any(u['kind']=='global_or_nonlocal_read' for u in ir['unresolved'])
