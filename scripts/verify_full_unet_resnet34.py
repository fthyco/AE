"""Independent SMP 0.6.0 U-Net ResNet34 source/FX/runtime/rendering gate."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections import Counter
from pathlib import Path

import torch
import torchvision
import segmentation_models_pytorch as smp
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from ae.sae.api import extract, validate_bundle
from ae.presenter.api import present

SOURCE_MODULES = {
    'base/model.py': 'segmentation_models_pytorch.base.model',
    'base/modules.py': 'segmentation_models_pytorch.base.modules',
    'base/heads.py': 'segmentation_models_pytorch.base.heads',
    'decoders/unet/model.py': 'segmentation_models_pytorch.decoders.unet.model',
    'decoders/unet/decoder.py': 'segmentation_models_pytorch.decoders.unet.decoder',
    'encoders/resnet.py': 'segmentation_models_pytorch.encoders.resnet',
    'torchvision/resnet.py': 'torchvision.models.resnet',
}


def snapshot_sources(destination: Path) -> dict[str, str]:
    result = {}
    for rel, module in SOURCE_MODULES.items():
        original = Path(importlib.import_module(module).__file__)
        raw = original.read_bytes()
        path = destination / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        assert path.read_bytes() == raw
        result[rel] = hashlib.sha256(raw).hexdigest()
    return result


def _reference_target(node):
    return (str(node.target) if node.op in ('placeholder','output','call_module')
            else getattr(node.target,'__name__',str(node.target)))


def _hook_sequence(model, input_):
    events=[]; handles=[]
    for path,module in model.named_modules():
        if path and not list(module.children()):
            handles.append(module.register_forward_hook(
                lambda obj,args,result,path=path:
                events.append((path,list(result.shape) if isinstance(result,torch.Tensor) else None))))
    try:
        with torch.inference_mode():
            result=model(input_)
    finally:
        for handle in handles:
            handle.remove()
    return events,list(result.shape)


def verify(output: Path, shapes=((1,3,64,64), (1,3,96,96),
                                (1,3,224,224), (1,3,64,96))):
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    # No network access or pretrained weight download is used.
    model=smp.Unet(encoder_name='resnet34', encoder_weights=None,
                   in_channels=3,classes=1).eval()
    reference=symbolic_trace(model)
    fx_nodes=list(reference.graph.nodes)
    expected_edges=sum(len(x.all_input_nodes) for x in fx_nodes)
    assert len(fx_nodes)==198 and expected_edges==227,(len(fx_nodes),expected_edges)
    expected_modules=dict(model.named_modules())
    summaries=[]
    hashes=None
    for batch,ch,h,w in shapes:
        label=f'{batch}x{ch}x{h}x{w}'
        folder=output/label
        src=folder/'source'
        source_hashes=snapshot_sources(src)
        if hashes is None: hashes=source_hashes
        assert hashes==source_hashes,'Installed upstream source changed between runs'
        bundle=folder/'sae'
        extract(src,bundle,entrypoint='decoders.unet.model:Unet',input_shape=(batch,ch,h,w))
        assert validate_bundle(bundle)['valid']
        static=json.loads((bundle/'instance-forward.json').read_text())
        assert static['status']=='expanded_static',static
        assert len(static['nodes'])==198 and len(static['edges'])==227
        node_ids={n['id'] for n in static['nodes']}
        assert len(node_ids)==198
        ShapeProp(reference).propagate(torch.zeros(batch,ch,h,w))
        lookup=dict(zip((x['id'] for x in static['nodes']),(n.name for n in fx_nodes)))
        op_matches=dep_matches=shape_matches=0
        tensor_shapes=0
        mismatches=[]
        for idx,(got,ref) in enumerate(zip(static['nodes'],fx_nodes)):
            expected_op=(ref.op,_reference_target(ref))
            if (got['op'],got['target'])!=expected_op:
                mismatches.append({'index':idx,'kind':'op','ae':[got['op'],got['target']],
                                   'fx':list(expected_op)})
            else:op_matches+=1
            actual_dependencies=[lookup[i] for i in got['input_ids']]
            wanted=[node.name for node in ref.all_input_nodes]
            if actual_dependencies!=wanted:
                mismatches.append({'index':idx,'kind':'dependency','ae':actual_dependencies,
                                   'fx':wanted})
            else:dep_matches+=len(wanted)
            meta=ref.meta.get('tensor_meta')
            expected_shape=list(meta.shape) if hasattr(meta,'shape') else None
            if expected_shape is not None:tensor_shapes+=1
            if got['shape']!=expected_shape:
                mismatches.append({'index':idx,'kind':'shape','ae':got['shape'],
                                   'fx':expected_shape})
            else:shape_matches+=1
            if not got['source_refs'] or not got['source_ir_ids']:
                mismatches.append({'index':idx,'kind':'source_provenance'})
            if got['op']=='call_module':
                if got['target'] not in expected_modules or got['module_instance']!=got['target']:
                    mismatches.append({'index':idx,'kind':'module_identity','target':got['target']})
        actual_edges={(x['source'],x['target'],x['input_slot']) for x in static['edges']}
        expected_edge_set={(a['input_ids'][pos],a['id'],pos) for a in static['nodes']
                           for pos in range(len(a['input_ids']))}
        if actual_edges!=expected_edge_set or len(actual_edges)!=len(static['edges']):
            mismatches.append({'kind':'IR edge inconsistency'})
        by_id={a['id']:a for a in static['nodes']}
        for edge in static['edges']:
            pred=by_id[edge['source']];succ=by_id[edge['target']]
            want=('STATIC_INSTANCE_SHAPE_METADATA_DEPENDENCY'
                  if pred['shape'] is None or succ['shape'] is None
                  else 'STATIC_INSTANCE_TENSOR_DEPENDENCY')
            if edge.get('relation')!=want:
                mismatches.append({'kind':'tensor/metadata edge classification',
                                   'source':edge['source'],'target':edge['target']})
        hooks,outshape=_hook_sequence(model,torch.zeros(batch,ch,h,w))
        expected_hooks=[(n['target'],n['shape']) for n in static['nodes']
                        if n['op']=='call_module']
        if hooks!=expected_hooks:
            first=next((i for i,(a,b) in enumerate(zip(hooks,expected_hooks))
                        if a!=b),None)
            mismatches.append({'kind':'runtime leaf hook order or shape','at':first,
                               'counts':[len(hooks),len(expected_hooks)],
                               'got':hooks[first] if first is not None else None,
                               'wanted':expected_hooks[first] if first is not None else None})
        if outshape!=static['output_shape']:
            mismatches.append({'kind':'final output','got':outshape,
                               'wanted':static['output_shape']})
        if len(hooks)!=149:
            mismatches.append({'kind':'hook count','got':len(hooks),'wanted':149})
        rendered=present(bundle,folder/'figure',view='instance',level='operation',format='all')
        manifest=json.loads((folder/'figure/presentation-manifest.json').read_text())
        if rendered['status']!='partial' or not rendered['render_allowed']:
            mismatches.append({'kind':'health gate','value':rendered})
        if (set(manifest['visual_elements'])!=node_ids or
            set(manifest['visual_links'])!={x['id'] for x in static['edges']}):
            mismatches.append({'kind':'projection completeness'})
        for extension in ('svg','pdf','tex'):
            path=folder/'figure'/f'architecture.{extension}'
            if not path.is_file() or path.stat().st_size==0:
                mismatches.append({'kind':'missing diagram','extension':extension})
        if mismatches:
            (folder/'failures.json').write_text(json.dumps(mismatches[:40],indent=2))
            raise AssertionError(f'{label}: {mismatches[:10]}')
        row={'shape':[batch,ch,h,w],'operations_matched':op_matches,
             'dependencies_matched':dep_matches,'tensor_shapes_matched':tensor_shapes,
             'all_shape_or_metadata_matches':shape_matches,
             'hook_invocations':len(hooks),'distinct_module_instances':len({x[0] for x in hooks}),
             'figure_elements':len(manifest['visual_elements']),
             'figure_links':len(manifest['visual_links']),
             'final_output_shape':outshape,'status':'verified_for_input'}
        summaries.append(row)
        (folder/'result.json').write_text(json.dumps(row,indent=2))
    report={'model':'smp.Unet(resnet34)','status':'verified_for_selected_inputs',
            'torch':torch.__version__,'torchvision':torchvision.__version__,
            'smp':smp.__version__,'source_sha256':hashes,
            'models_imported_only_in_independent_verifier':True,
            'full_equivalence_for_arbitrary_sources_and_kwargs':False,
            'results':summaries}
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('benchmark-output/full-unet-resnet34'))
    args=parser.parse_args()
    verify(args.output)
