"""One-stage Presenter API: verified bundle -> provenance-backed scientific outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ae.sae.ir.canonical import canonical_bytes, sha256
from .reader import load_bundle
from .projection import project
from .flow import project_flow
from .flow_layout import make_flow_scene
from .health import evaluate
from .layout import make_scene
from .render import svg, pdf, tikz
from .themes import select_theme


FORMATS={'svg': ('svg',), 'pdf': ('pdf',), 'tikz': ('tex',),
         'all': ('svg','pdf','tex')}


def present(bundle_path: str | Path, output: str | Path, *, level: str = 'block',
            mode: str = 'exploration', format: str = 'svg',
            allow_partial: bool = False, view: str = 'structure',
            focus: str | None = None, theme: str = 'scientific-latex') -> dict[str, Any]:
    """Render architecture views and write health reports."""
    style = select_theme(theme)
    if view not in ('structure','flow'):
        raise ValueError(f'Invalid view: {view}')
    if focus and view != 'flow':
        raise ValueError('--focus currently supports the flow view only')
    if level not in ('overview','block','operation'):
        raise ValueError(f'Invalid level: {level}')
    if mode not in ('exploration','publication'):
        raise ValueError(f'Invalid mode: {mode}')
    if format not in FORMATS:
        raise ValueError(f'Invalid format: {format}')
    if allow_partial and mode != 'publication':
        raise ValueError('--allow-partial applies to publication mode only')
    output_dir=Path(output)
    output_dir.mkdir(parents=True,exist_ok=True)
    # No stale figures should survive a failed or subsequently blocked invocation.
    for ext in ('svg','pdf','tex'):
        (output_dir/f'architecture.{ext}').unlink(missing_ok=True)
    bundle=None; graph=None; problem=None
    try:
        bundle=load_bundle(bundle_path)
        graph=project_flow(bundle,level,focus=focus) if view=='flow' else project(bundle,level)
    except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
        problem=f'{type(exc).__name__}: {exc}'
    health=evaluate(bundle,graph,mode=mode,level=level,input_error=problem)
    if view == 'flow':
        health['scope']='Source-backed static forward dependencies, supported call summaries and conditional type-level shapes; runtime behavior unverified'
    health['view']=view
    health['focus']=focus
    allowed=health['status'] != 'blocked' and (mode=='exploration' or
           health['status']=='complete' or allow_partial)
    health['render_allowed']=allowed
    health['publication_gate']='withheld_partial' if health['status']=='partial' and mode=='publication' and not allow_partial else 'ok'
    if bundle is not None:
        input_hashes=bundle.metadata.get('artifact_sha256',{})
    else:
        input_hashes={}
    manifest={'schema_version':'1.0','style_version':style['name'],
              'mode':mode,'level':level,'view':view,'focus':focus,'entrypoint':bundle.ir.get('entrypoint') if bundle else None,
              'input_artifact_hashes':input_hashes,'status':health['status'],
              'rendered':allowed,'visual_elements':{},'visual_links':{},
              'omitted_indexed_ir_ids':[],'output_sha256':{}}
    if graph is not None:
        manifest['visual_elements']={n['id']:{'kind':n['kind'],
               'ir_node_ids':n['ir_node_ids'],'ir_edge_ids':n['ir_edge_ids'],
               'source_refs':n['source_refs']} for n in graph['nodes']}
        manifest['visual_links']={e['id']:{'relation_kind':e['relation_kind'],
               'ir_edge_ids':e['ir_edge_ids'],'source_refs':e['source_refs']}
               for e in graph['links']}
        if bundle:
            indexed=set()
            for m in bundle.model_index.get('modules',[]):
                if m['definition_id'] not in {x for n in graph['nodes'] for x in n['ir_node_ids']}:
                    continue
                indexed.update(m.get('forward_relation_ids',[]))
                indexed.update(c['call_site_id'] for c in m['forward_calls'])
                indexed.update(op['operation_site_id'] for op in m['forward_operations'])
            mapped={z for n in graph['nodes'] for z in n['ir_node_ids']}
            mapped|={z for e in graph['links'] for z in e['ir_edge_ids']}
            manifest['omitted_indexed_ir_ids']=[{'id':i,'reason':'hidden_by_selected_view_or_unsupported_projection'}
                                                for i in sorted(indexed-mapped)]
    if allowed:
        scene=(make_flow_scene if view=='flow' else make_scene)(
            graph, status=health['status'],
            entrypoint=(f"{health['entrypoint']} / {focus}"
                        if focus and focus != (health['entrypoint'] or '').split(':')[-1]
                        else health['entrypoint']))
        scene['style'] = style
        exporters={'svg':svg,'pdf':pdf,'tex':tikz}
        for extension in FORMATS[format]:
            content=exporters[extension](scene)
            (output_dir/f'architecture.{extension}').write_bytes(content)
            manifest['output_sha256'][f'architecture.{extension}']=sha256(content)
    (output_dir/'presentation-health.json').write_bytes(canonical_bytes(health))
    (output_dir/'presentation-manifest.json').write_bytes(canonical_bytes(manifest))
    return {'status':health['status'],'render_allowed':allowed,'mode':mode,'level':level,'view':view,'focus':focus,'theme':theme,
            'formats':list(FORMATS[format]) if allowed else [],
            'issues':len(health['issues']),'output':str(output_dir)}
