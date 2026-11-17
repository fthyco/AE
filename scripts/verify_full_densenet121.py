"""Independent FX, shapes, runtime hooks, and rendered-graph gate for DenseNet121."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torchvision
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from ae.sae.api import extract, validate_bundle
from ae.presenter.api import present


def verify(out: Path, sides=(64,96,224)):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    source=Path(torchvision.models.densenet.__file__).read_bytes()
    model=torchvision.models.densenet121(weights=None).eval()
    module_names=dict(model.named_modules())
    fx=symbolic_trace(model)
    expected_count=len(list(fx.graph.nodes))
    expected_edges=sum(len(n.all_input_nodes) for n in fx.graph.nodes)
    assert (expected_count,expected_edges)==(433,967)
    results=[]
    for side in sides:
        folder=out/f'input_{side}';src=folder/'source';src.mkdir(parents=True,exist_ok=True)
        (src/'densenet.py').write_bytes(source)
        extract(src,folder/'sae',entrypoint='densenet:densenet121',input_shape=(1,3,side,side))
        assert validate_bundle(folder/'sae')['valid']
        static=json.loads((folder/'sae/instance-forward.json').read_text())
        assert static['status']=='expanded_static',static
        ShapeProp(fx).propagate(torch.zeros(1,3,side,side))
        fx_nodes=list(fx.graph.nodes)
        a_nodes=static['nodes']
        assert len(a_nodes)==len(fx_nodes),(len(a_nodes),len(fx_nodes))
        by_id={n['id']:n for n in a_nodes}
        assert len(by_id)==len(a_nodes)
        names=dict(zip([n['id'] for n in a_nodes],[n.name for n in fx_nodes]))
        op_matches=edge_matches=shape_matches=0
        errors=[]
        for idx,(a,b) in enumerate(zip(a_nodes,fx_nodes)):
            if b.op in ('call_module','placeholder','output'):
                target=str(b.target)
            elif b.op=='call_function':
                target=getattr(b.target,'__name__',str(b.target))
            else:target='UNKNOWN'
            if (a['op'],a['target'])!=(b.op,target):
                errors.append({'index':idx,'case':'op', 'got':(a['op'],a['target']),'expected':(b.op,target)})
            else:op_matches+=1
            got=[names[i] for i in a['input_ids']]
            expected=[n.name for n in b.all_input_nodes]
            if got!=expected:errors.append({'index':idx,'case':'dependencies','got':got,'expected':expected})
            else:edge_matches+=len(got)
            meta=b.meta.get('tensor_meta')
            if hasattr(meta,'shape'):
                if a['shape']!=list(meta.shape):
                    errors.append({'index':idx,'case':'shape','got':a['shape'],'expected':list(meta.shape)})
                else:shape_matches+=1
            if a['op']=='call_module' and a['target'] not in module_names:
                errors.append({'index':idx,'case':'unknown_module'})
            if not a['source_refs'] or not a['source_ir_ids']:
                errors.append({'index':idx,'case':'source_provenance'})
        visits=[];hooks=[]
        for path,module in model.named_modules():
            if path and not any(module.children()):
                hooks.append(module.register_forward_hook(
                    lambda obj,args,output,path=path:visits.append((path,list(output.shape)))))
        try:
            with torch.inference_mode(): result=model(torch.zeros(1,3,side,side))
        finally:
            for hook in hooks:hook.remove()
        expected_calls=[(n['target'],n['shape']) for n in a_nodes if n['op']=='call_module']
        if visits!=expected_calls:
            diff=next((i for i,(x,y) in enumerate(zip(visits,expected_calls)) if x!=y),None)
            errors.append({'case':'runtime_sequence_and_shapes','index':diff,
                           'got':visits[diff] if diff is not None else len(visits),
                           'expected':expected_calls[diff] if diff is not None else len(expected_calls)})
        if list(result.shape)!=static['output_shape']:
            errors.append({'case':'runtime_output'})
        expected_edge_records={(node['input_ids'][slot],node['id'],slot)
                                for node in a_nodes for slot in range(len(node['input_ids']))}
        actual_edge_records={(e['source'],e['target'],e['input_slot']) for e in static['edges']}
        if actual_edge_records!=expected_edge_records:
            errors.append({'case':'ir_operand_edges'})
        visual=present(folder/'sae',folder/'figure',view='instance',level='operation',format='all')
        manifest=json.loads((folder/'figure/presentation-manifest.json').read_text())
        if visual['status']!='partial' or not visual['render_allowed']:
            errors.append({'case':'view_health'})
        if set(manifest['visual_elements'])!={n['id'] for n in a_nodes} or \
           set(manifest['visual_links'])!={e['id'] for e in static['edges']}:
            errors.append({'case':'visual_projection_completeness'})
        for fmt in ('svg','pdf','tex'):
            if not (folder/'figure'/f'architecture.{fmt}').is_file():
                errors.append({'case':'render_missing','format':fmt})
        if errors:
            (folder/'failures.json').write_text(json.dumps(errors[:30],indent=2))
            raise AssertionError(errors[:10])
        row={'input_shape':[1,3,side,side], 'operations':op_matches,
             'dependencies':edge_matches,'shapes':shape_matches,
             'hook_invocations':len(visits),'unique_module_targets':len({x[0] for x in visits}),
             'figure_elements':len(manifest['visual_elements']),
             'figure_links':len(manifest['visual_links']),
             'final_shape':list(result.shape),'status':'verified_for_input'}
        results.append(row)
        (folder/'result.json').write_text(json.dumps(row,indent=2))
    summary={'model':'densenet121','status':'verified_for_selected_inputs',
             'source_sha256':hashlib.sha256(source).hexdigest(),
             'torch':torch.__version__,'torchvision':torchvision.__version__,
             'results':results,'full_equivalence_all_dynamic_inputs':False,
             'static_extractor_imports_target':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('benchmark-output/full-densenet121'))
    args=parser.parse_args()
    verify(args.output)
