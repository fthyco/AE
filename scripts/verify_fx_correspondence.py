"""Independent FX source-site, dependency, and concrete-shape audit."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from collections import defaultdict
from pathlib import Path

import torch
import torchvision
from torch.fx import GraphModule, Tracer
from torch.fx.passes.shape_prop import ShapeProp

from ae.sae.api import extract, validate_bundle
from ae.verification.fx_correspondence import compare_source_sites, fx_operation


class SourceTracer(Tracer):
    def __init__(self, source_root: Path):
        super().__init__()
        self.source_root = source_root.resolve()

    def create_proxy(self, kind, target, args, kwargs, name=None,
                     type_expr=None, proxy_factory_fn=None):
        proxy = super().create_proxy(kind, target, args, kwargs,
                                    name, type_expr, proxy_factory_fn)
        if kind in ('call_module', 'call_function', 'call_method'):
            frame = inspect.currentframe()
            try:
                frame = frame.f_back
                while frame is not None:
                    location = Path(frame.f_code.co_filename).resolve()
                    try:
                        relative = location.relative_to(self.source_root)
                    except ValueError:
                        frame = frame.f_back
                        continue
                    positions = inspect.getframeinfo(frame, context=0).positions
                    proxy.node.meta['ae_source'] = {
                        'path':relative.as_posix(), 'line':frame.f_lineno,
                        'column':positions.col_offset if positions else None,
                        'end_column':positions.end_col_offset if positions else None,
                    }
                    break
            finally:
                del frame
        return proxy


def fx_events(graph: GraphModule) -> list[dict]:
    events = []
    for node in graph.graph.nodes:
        target_name = (node.target if isinstance(node.target, str)
                       else getattr(node.target, '__name__', str(node.target)))
        record = {
            'name':node.name, 'op':node.op,
            'target_name':str(target_name),
            'target':str(node.target),
            'source':node.meta.get('ae_source'),
            'input_nodes':[source.name for source in node.all_input_nodes],
        }
        meta = node.meta.get('tensor_meta')
        if isinstance(meta, torch.fx.passes.shape_prop.TensorMetadata):
            record['runtime_shape'] = [int(dim) for dim in meta.shape]
        events.append(record)
    return events


def compare_shapes(ir: dict, fx: list[dict], report: dict) -> dict:
    shape_map = defaultdict(set)
    for fact in ir['nodes']:
        if fact['kind'] == 'symbolic_shape_fact':
            dims = fact['attributes']['dimensions']
            if dims and all(isinstance(d,str) and (d.isdecimal() and int(d)>0 or d == '...') for d in dims):
                shape_map[fact['attributes']['value_id']].add(tuple(int(d) if d.isdecimal() else d for d in dims))
    events = {e['name']:e for e in fx}
    checked = []
    for match in report['matched_site_details']:
        if match['fx_occurrences'] != 1 or len(match['ae_sites']) != 1:
            continue
        fact_shapes = shape_map.get(match['ae_sites'][0], set())
        if len(fact_shapes) != 1:
            continue
        runtime_shape = events[match['fx_names'][0]].get('runtime_shape')
        if runtime_shape is None:
            continue
        static = next(iter(fact_shapes))
        concrete = '...' not in static
        suffix = static[1:] if static and static[0] == '...' else static
        agrees = (list(static)==runtime_shape if concrete else
                  len(runtime_shape)>=len(suffix) and list(suffix)==runtime_shape[-len(suffix):])
        checked.append({'path':match['path'],'line':match['line'],
                        'operation':match['operation'], 'kind':'concrete' if concrete else 'suffix',
                        'static':list(static),'fx':runtime_shape, 'match':agrees})
    return {
        'shape_comparisons':len(checked),
        'shape_matches':sum(e['match'] for e in checked),
        'shape_mismatches':sum(not e['match'] for e in checked),
        'concrete_shape_checks':sum(e['kind']=='concrete' for e in checked),
        'suffix_shape_checks':sum(e['kind']=='suffix' for e in checked),
        'details':checked,
    }


def create_source(model: torch.nn.Module, mode: str, out: Path) -> tuple[str, Path, dict]:
    if mode in {'fixture', 'nested_composite'}:
        filename = 'audit_model.py' if mode == 'fixture' else 'model.py'
        original=Path(__file__).parent.parent/'tests'/'architecture_cases'/'15_nested_shape_fx'/filename
        src=out/'source';src.mkdir(parents=True,exist_ok=True)
        destination=src/filename
        data=original.read_bytes();destination.write_bytes(data)
        entry = 'audit_model:Model' if mode == 'fixture' else 'model:Outer'
        return entry,original.parent,{filename:hashlib.sha256(data).hexdigest()}
    from torchvision.models import resnet, densenet
    original=Path(resnet.__file__ if mode.startswith('resnet') else densenet.__file__).resolve()
    src=out/'source';src.mkdir(parents=True,exist_ok=True)
    data=original.read_bytes();(src/original.name).write_bytes(data)
    factory=mode
    assert mode in {'resnet18','resnet50','densenet121'}
    return f'{original.stem}:{factory}',original.parent,{original.name:hashlib.sha256(data).hexdigest()}


def run(mode: str, output: Path) -> dict:
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    if mode in {'fixture','nested_composite'}:
        import importlib.util
        filename='audit_model.py' if mode=='fixture' else 'model.py'
        target=Path(__file__).parent.parent/'tests'/'architecture_cases'/'15_nested_shape_fx'/filename
        spec=importlib.util.spec_from_file_location('ae_shape_fixture',target)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        model=(module.Model() if mode=='fixture' else module.Outer()).eval()
        size=(2,4) if mode=='fixture' else (1,3,8,8)
    else:
        model=getattr(torchvision.models,mode)(weights=None).eval()
        size=(1,3,64,64)
    entry,original_root,hashes=create_source(model,mode,output)
    src=output/'source';out=output/'sae'
    extract(src,out,entrypoint=entry)
    assert validate_bundle(out)['valid']
    ir=json.loads((out/'architecture.ir.json').read_text())
    tracer=SourceTracer(original_root)
    graph=GraphModule(model,tracer.trace(model))
    ShapeProp(graph).propagate(torch.zeros(*size))
    fx=fx_events(graph)
    report=compare_source_sites(ir,fx)
    report['shapes']=compare_shapes(ir,fx,report)
    index=json.loads((out/'model-index.json').read_text())
    root_fid=next((m.get('forward_method_id') for m in index.get('modules', [])
                   if m['definition_id']==index.get('root_definition_id')),None)
    root_out={n['id'] for n in ir['nodes']
              if n['kind']=='forward_output' and n['attributes'].get('forward_id')==root_fid}
    inferred={tuple(n['attributes']['dimensions']) for n in ir['nodes']
              if n['kind']=='symbolic_shape_fact' and n['attributes']['value_id'] in root_out}
    output_meta=next((node.meta.get('tensor_meta') for node in graph.graph.nodes
                      if node.op=='output'),None)
    output_shape=(list(output_meta.shape)
                  if isinstance(output_meta,torch.fx.passes.shape_prop.TensorMetadata) else None)
    root_checks=[]
    for template in sorted(inferred):
        if output_shape is None: continue
        suffix=template[1:] if template and template[0]=='...' else template
        expected=[int(d) for d in suffix] if all(d.isdecimal() for d in suffix) else None
        agree=(expected is not None and len(output_shape)>=len(suffix) and
               expected==output_shape[-len(suffix):]) if template and template[0]=='...' else (
               expected is not None and expected==output_shape)
        root_checks.append({'static':list(template),'fx_output':output_shape,
                            'match':agree, 'kind':'suffix' if template[0]=='...' else 'concrete'})
    report['root_forward_output_shape_checks']=root_checks
    report['model']=mode
    report['tool_versions']={'torch':torch.__version__,'torchvision':torchvision.__version__}
    report['source_hashes']=hashes
    report['fx_total_nodes']=len(fx)
    report['fx_unscoped_nodes']=sum(e['source'] is None for e in fx if fx_operation(e) is not None)
    report['static_graph_nodes']=len(ir['nodes'])
    report['static_graph_edges']=len(ir['edges'])
    report['fx_supported_event_count']=sum(1 for e in fx if fx_operation(e) is not None)
    report['fx_total_events_excluded_by_scope']=len(fx)-sum(1 for e in fx if fx_operation(e) is not None)
    report['excluded_from_coverage']='FX events with unknown source or unsupported operation names'
    report['full_operation_coverage']=None
    report['full_dependency_correctness']=None
    (output/'correspondence.json').write_text(json.dumps(report,indent=2,sort_keys=True))
    (output/'fx_events.json').write_text(json.dumps(fx,indent=2,sort_keys=True))
    brief={k:report[k] for k in ['model','fx_total_nodes','static_graph_nodes',
        'observed_fx_site_count','matched_sites','unmatched_fx_sites','ambiguous_sites',
        'observed_site_recall','one_to_one_events','edge_true_positives',
        'edge_false_positives','edge_false_negatives','edge_precision','edge_recall']}
    brief['shapes']=report['shapes']
    brief['root_output']=report['root_forward_output_shape_checks']
    print('FX_AE_MATCH '+json.dumps(brief,sort_keys=True),flush=True)
    if mode in {'resnet18','resnet50'}:
        expected={'resnet18':8,'resnet50':16}[mode]
        adds=sum(e['fx_occurrences'] for e in report['matched_site_details']
                 if e['operation']=='add')
        assert adds == expected, (mode, adds)
        assert report['observed_site_recall']==1
    if mode == 'densenet121':
        cats=sum(e['fx_occurrences'] for e in report['matched_site_details']
                 if e['operation']=='cat')
        assert cats == 62, cats
        assert report['observed_site_recall']==1
    if mode == 'nested_composite':
        assert report['root_forward_output_shape_checks']
        assert all(x['match'] for x in report['root_forward_output_shape_checks'])
    if mode == 'fixture':
        assert report['matched_sites'] >= 3
        assert report['edge_true_positives'] >= 2
        assert report['shapes']['shape_mismatches'] == 0
        assert report['shapes']['shape_matches'] >= 2
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('model',choices=['fixture','nested_composite','resnet18','resnet50','densenet121'])
    parser.add_argument('--output',type=Path,default=Path('benchmark-output/fx-ae'))
    args=parser.parse_args()
    run(args.model,args.output/args.model)
