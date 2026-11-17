"""Entrypoint-scoped, source-backed index. No code execution or rendering."""
from __future__ import annotations
import ast
from ..ir.canonical import stable_id
from ..linking.project import derived_edge, span, syntax_ids

RULE = 'model-architecture-index/v1'

def build_model_index(graph, units, entrypoint_id, entrypoint):
    result = {'schema_version':'1.0', 'entrypoint':entrypoint,
              'root_definition_id':entrypoint_id,
              'status':'unresolved' if entrypoint else 'no_entrypoint',
              'modules':[], 'unresolved':[],
              'limitations':['Lexical call order is not execution order',
                  'Module construction sites do not prove runtime instantiation',
                  'Unresolved local constructors are not guessed']}
    if not entrypoint_id:
        if entrypoint:
            result['unresolved'].append({'kind':'model_entrypoint','reason':'No unique entrypoint'})
        return result
    definitions = {}
    for unit in units:
        ids = syntax_ids(unit)
        for cls in unit.tree.body:
            if isinstance(cls, ast.ClassDef):
                key = stable_id(unit.path,'semantic','class_definition',ids[cls],'')
                definitions[key] = (unit,cls,ids)
    if entrypoint_id not in definitions:
        result['status']='unsupported_entrypoint'
        result['unresolved'].append({'kind':'model_entrypoint','reason':'Entrypoint must be a top-level class'})
        return result
    nodes={n['id']:n for n in graph.nodes}
    old_edges=list(graph.edges)
    links={}
    for e in old_edges:
        links.setdefault((e['source'],e['relation']),[]).append(e['target'])
    for values in links.values(): values.sort()
    visited=set()
    def walk_body(method):
        def sub(node):
            yield node
            for child in ast.iter_child_nodes(node):
                if isinstance(child,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef,ast.Lambda)): continue
                yield from sub(child)
        for statement in method.body: yield from sub(statement)
    def visit(def_id, active=()):
        if def_id in active:
            result['unresolved'].append({'kind':'recursive_module_reference','definition_id':def_id})
            return
        if def_id in visited: return
        if def_id not in definitions: return
        visited.add(def_id)
        unit,cls,ids=definitions[def_id]
        methods=[n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
        inits=[n for n in methods if n.name=='__init__']
        forwards=[n for n in methods if n.name=='forward']
        forward_id=stable_id(unit.path,'semantic','method_definition',ids[forwards[0]],'') if len(forwards)==1 else None
        if forward_id:
            derived_edge(graph,unit,forwards[0],def_id,forward_id,'DECLARES_FORWARD',RULE)
        else:
            result['unresolved'].append({'kind':'forward_resolution','definition_id':def_id,'reason':'No unique forward method'})
        components=[]
        child_ids=set()
        for init in inits:
            for statement in walk_body(init):
                if not isinstance(statement,(ast.Assign,ast.AnnAssign)): continue
                targets=statement.targets if isinstance(statement,ast.Assign) else [statement.target]
                for target in targets:
                    if not (isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id=='self'): continue
                    value=statement.value
                    if not isinstance(value,ast.Call): continue
                    ref=span(unit,value)
                    sites=[n for n in graph.nodes if n['kind']=='module_construction_site' and
                           n['source_refs'][0]['path']==unit.path and
                           n['source_refs'][0]['start_byte']==ref['start_byte'] and
                           n['source_refs'][0]['end_byte']==ref['end_byte']]
                    if len(sites)!=1:
                        result['unresolved'].append({'kind':'unrecognized_or_ambiguous_component',
                            'definition_id':def_id,'assignment_ast_id':ids[target],
                            'expression':ast.unparse(value),'candidate_ids':sorted(n['id'] for n in sites)})
                        continue
                    site=sites[0]
                    sid=site['id']
                    target_defs=links.get((sid,'CONSTRUCTS_LOCAL_MODULE'),[])
                    dst=target_defs[0] if len(target_defs)==1 else None
                    call=stable_id(unit.path,'semantic','call_site',ids[value],'')
                    layer_specs=sorted(e['source'] for e in old_edges
                              if e['relation']=='DESCRIBES_PYTORCH_CALL' and e['target']==call)
                    components.append({'construction_site_id':sid,'layer_spec_ids':layer_specs,
                        'owner_expression':ast.unparse(target),
                        'declared_type':site['attributes'].get('declared_type'),
                        'target_definition_id':dst,'source_refs':[ref]})
                    derived_edge(graph,unit,value,def_id,sid,'DECLARES_COMPONENT',RULE)
                    if dst: child_ids.add(dst)
        calls=[]; operations=[]; relations=[]
        if forward_id:
            forward=forwards[0]
            for node in walk_body(forward):
                if isinstance(node,ast.Call):
                    call=stable_id(unit.path,'semantic','call_site',ids[node],'')
                    if call not in nodes: continue
                    calls.append({'call_site_id':call,'callee_expression':nodes[call]['attributes']['callee_expression'],
                        'module_construction_site_ids':links.get((call,'USES_MODULE_REFERENCE'),[]),
                        'local_definition_ids':links.get((call,'CALLS_LOCAL_DEFINITION'),[]),
                        'source_refs':[span(unit,node)]})
                    derived_edge(graph,unit,node,forward_id,call,'HAS_FORWARD_CALL',RULE)
                elif isinstance(node,(ast.BinOp,ast.UnaryOp,ast.BoolOp,ast.Compare,ast.Subscript)):
                    op=stable_id(unit.path,'semantic','operation_site',ids[node],'')
                    if op not in nodes: continue
                    operations.append({'operation_site_id':op,'operator':nodes[op]['attributes']['operator'],
                                       'source_refs':[span(unit,node)]})
                    derived_edge(graph,unit,node,forward_id,op,'HAS_FORWARD_OPERATION',RULE)
            syntax_members={ids[n] for n in ast.walk(forward)}
            semantic_members={e['source'] for e in old_edges if e['relation']=='APPEARS_AT' and e['target'] in syntax_members}
            members=syntax_members|semantic_members
            kinds={'EXPRESSION_DEPENDS_ON','VALUE_EXPRESSION','READS_LEXICAL_DEFINITION',
                   'RETURNS_EXPRESSION','HAS_ARGUMENT','HAS_KEYWORD','CALLEE_EXPRESSION',
                   'CONDITION_EXPRESSION','HAS_BRANCH_STATEMENT'}
            relations=sorted(e['id'] for e in old_edges if e['relation'] in kinds
                             and e['source'] in members and e['target'] in members)
        order=lambda d:(d['source_refs'][0]['path'],d['source_refs'][0]['start_byte'],
                        d['source_refs'][0]['end_byte'])
        result['modules'].append({'definition_id':def_id,'qualname':f'{unit.module}.{cls.name}',
               'source_refs':[span(unit,cls)],'forward_method_id':forward_id,
               'components':sorted(components,key=order),'forward_calls':sorted(calls,key=order),
               'forward_operations':sorted(operations,key=order),
               'forward_relation_ids':relations})
        for child in sorted(child_ids): visit(child,(*active,def_id))
    visit(entrypoint_id)
    result['modules'].sort(key=lambda m:(m['qualname'],m['definition_id']))
    result['unresolved'].sort(key=lambda x:(x['kind'],x.get('definition_id',''),x.get('assignment_ast_id','')))
    result['status']='indexed'
    return result

def validate_model_index(index,ir):
    if index.get('schema_version')!='1.0' or index.get('root_definition_id')!=ir.get('entrypoint_target_id'):
        raise ValueError('Invalid model-index contract')
    nodes={n['id']:n['kind'] for n in ir['nodes']}
    edges={e['id'] for e in ir['edges']}
    def check(ref,kind=None):
        if ref is None: return
        if ref not in nodes or (kind and nodes[ref]!=kind):
            raise ValueError(f'Model index dangling or mistyped reference: {ref}')
    check(index.get('root_definition_id'))
    factory=index.get('factory_specialization')
    if factory:
        check(factory['factory_definition_id'],'function_definition')
        check(factory['factory_configuration_id'],'factory_configuration')
        check(factory['model_definition_id'],'class_definition')
        check(factory['block_definition_id'],'class_definition')
        if factory['model_definition_id'] != index.get('root_definition_id'):
            raise ValueError('Factory model root mismatch')
        if sum(factory['stage_counts']) != factory['total_blocks']:
            raise ValueError('Factory block count mismatch')
        if len(factory['stages']) != len(factory['stage_counts']):
            raise ValueError('Factory stage count mismatch')
        for i, stage in enumerate(factory['stages']):
            check(stage['stage_id'],'factory_stage')
            check(stage['block_definition_id'],'class_definition')
            if stage['position'] != i or stage['count'] != factory['stage_counts'][i]:
                raise ValueError('Factory stage order mismatch')
            if len(stage['block_instance_ids']) != stage['count']:
                raise ValueError('Factory instance count mismatch')
            for ref in stage['block_instance_ids']:
                check(ref,'factory_block_instance')
    for m in index.get('modules',[]):
        check(m['definition_id'],'class_definition')
        check(m['forward_method_id'],'method_definition')
        for c in m['components']:
            check(c['construction_site_id'],'module_construction_site')
            check(c['target_definition_id'],'class_definition')
            for x in c['layer_spec_ids']: check(x,'pytorch_layer_spec')
        for c in m['forward_calls']:
            check(c['call_site_id'],'call_site')
            for x in c['module_construction_site_ids']: check(x,'module_construction_site')
            for x in c['local_definition_ids']: check(x)
        for op in m['forward_operations']: check(op['operation_site_id'],'operation_site')
        for rid in m['forward_relation_ids']:
            if rid not in edges: raise ValueError(f'Model index dangling relation: {rid}')
