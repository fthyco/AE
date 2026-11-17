"""Independent full-graph gate for unmodified TorchVision ResNet18 defaults."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch
import torchvision
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from ae.sae.api import extract, validate_bundle


def verify(output: Path, sides=(64,96,224)) -> dict:
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    rows=[]
    original=Path(torchvision.models.resnet.__file__).resolve()
    data=original.read_bytes()
    model=torchvision.models.resnet18(weights=None).eval()
    runtime_names=dict(model.named_modules())
    for side in sides:
        folder=output / f'input_{side}'
        source=folder/'source';source.mkdir(parents=True,exist_ok=True)
        (source/'resnet.py').write_bytes(data)
        extract(source, folder/'sae', entrypoint='resnet:resnet18',input_shape=(1,3,side,side))
        valid=validate_bundle(folder/'sae')
        assert valid['valid'],valid
        static=json.loads((folder/'sae/instance-forward.json').read_text())
        assert static['status']=='expanded_static',static.get('issues')
        graph=symbolic_trace(model)
        ShapeProp(graph).propagate(torch.zeros(1,3,side,side))
        fx=list(graph.graph.nodes)
        assert len(static['nodes'])==len(fx), (len(static['nodes']),len(fx))
        by_id={n['id']:n for n in static['nodes']}
        assert len(by_id)==len(static['nodes'])
        mapping=dict(zip((n['id'] for n in static['nodes']), (n.name for n in fx)))
        failures=[]; matched_ops=0;matched_edges=0;matched_shapes=0
        for idx,(a,b) in enumerate(zip(static['nodes'],fx)):
            op=b.op
            target=(str(b.target) if op=='call_module' else
                    'add' if op=='call_function' and getattr(b.target,'__name__',None)=='add' else
                    'flatten' if op=='call_function' and getattr(b.target,'__name__',None)=='flatten' else
                    'x' if op=='placeholder' else 'output' if op=='output' else 'UNKNOWN')
            if (a['op'],a['target'])!=(op,target):
                failures.append({'index':idx,'error':'operation_or_module_mismatch',
                                 'ae':(a['op'],a['target']),'fx':(op,target)})
            else:matched_ops+=1
            incoming=[mapping[ident] for ident in a['input_ids']]
            actual=[node.name for node in b.all_input_nodes]
            if incoming!=actual:
                failures.append({'index':idx,'error':'dependency_mismatch','ae':incoming,'fx':actual})
            else:matched_edges+=len(incoming)
            meta=b.meta.get('tensor_meta')
            if hasattr(meta,'shape'):
                target_shape=list(meta.shape)
                if a['shape']!=target_shape:
                    failures.append({'index':idx,'error':'shape_mismatch','ae':a['shape'],'fx':target_shape})
                else:matched_shapes+=1
            if a['op']=='call_module' and a['target'] not in runtime_names:
                failures.append({'index':idx,'error':'unknown_module_identity','path':a['target']})
            if not a['source_refs']:
                failures.append({'index':idx,'error':'missing_source_provenance'})
        # Runtime hooks independently confirm each actual leaf-module call and shape.
        visits=[];handles=[]
        def _hook(path):
            def listener(_module, _inputs, output):
                assert isinstance(output,torch.Tensor),path
                visits.append((path,list(output.shape)))
            return listener
        for path,module in model.named_modules():
            if path and not any(module.children()):
                handles.append(module.register_forward_hook(_hook(path)))
        try:
            with torch.inference_mode():
                actual_output=model(torch.zeros(1,3,side,side))
        finally:
            for handle in handles:
                handle.remove()
        expected_visits=[(n['target'],n['shape']) for n in static['nodes']
                         if n['op']=='call_module']
        if visits!=expected_visits:
            failures.append({'error':'runtime_hook_sequence_or_shape_mismatch',
                             'expected':len(expected_visits),'observed':len(visits),
                             'first_difference':next((i for i,(a,b) in enumerate(zip(visits,expected_visits))
                                                     if a!=b),None)})
        if list(actual_output.shape)!=static['output_shape']:
            failures.append({'error':'runtime_final_shape_mismatch'})
        reported_edges={(e['source'],e['target'],e['input_slot']) for e in static['edges']}
        expected_edges={(n['input_ids'][i],n['id'],i) for n in static['nodes']
                        for i in range(len(n['input_ids']))}
        if reported_edges!=expected_edges:
            failures.append({'error':'graph_edges_do_not_match_node_operands'})
        assert not failures,failures[:15]
        row={'input_shape':[1,3,side,side], 'node_count':len(fx),
             'ae_edges':len(static['edges']),'fx_edges':sum(len(n.all_input_nodes) for n in fx),
             'operation_matches':matched_ops,'edge_matches':matched_edges,
             'shape_matches':matched_shapes,'output_shape':static['output_shape'],
             'modules_verified':len({a['target'] for a in static['nodes'] if a['op']=='call_module'}),
             'runtime_leaf_invocations_verified':len(visits),
             'status':'verified_for_input'}
        rows.append(row)
        (folder/'result.json').write_text(json.dumps(row,indent=2))
    result={'model':'resnet18','source_sha256':hashlib.sha256(data).hexdigest(),
            'torch_version':torch.__version__,'torchvision_version':torchvision.__version__,
            'verified_inputs':rows,
            'scope':'default_factory, selected concrete input shapes; not arbitrary dynamic paths',
            'full_operation_matching':True,'full_dependency_matching':True,
            'full_shape_matching_for_selected_inputs':True,
            'next_model_unlocked':True}
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('benchmark-output/full-resnet18'))
    args=parser.parse_args()
    verify(args.output)
