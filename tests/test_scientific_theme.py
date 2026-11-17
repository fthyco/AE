"""Publication typography, orthogonal geometry, and evidence-preserving themes."""
from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from ae.sae.api import extract
from ae.presenter.api import present
from ae.presenter.reader import load_bundle
from ae.presenter.flow import project_flow
from ae.presenter.flow_layout import make_flow_scene
from ae.presenter.themes import select_theme

ROOT = Path(__file__).resolve().parents[1]


def bundle(tmp_path, model='cnn_residual_mvp', entrypoint='model:CNN'):
    directory=tmp_path/'sae'
    extract(ROOT/'examples'/model,directory,entrypoint=entrypoint)
    return directory


def test_scientific_theme_has_referenced_monochrome_style():
    t=select_theme('scientific-latex')
    assert t['name']=='scientific-latex/v1'
    assert 'Latin Modern Roman' in t['font_family']
    assert t['background']=='#ffffff'
    assert t['ink']=='#181818'
    assert t['stroke']=='#181818'
    assert t['corner_radius']<=1
    with pytest.raises(ValueError):
        select_theme('unsupported-theme')


def test_theme_only_changes_presentation_not_provenance(tmp_path):
    b=bundle(tmp_path)
    root=tmp_path/'a'
    legacy=tmp_path/'b'
    r1=present(b,root,view='flow',format='all',theme='scientific-latex')
    r2=present(b,legacy,view='flow',format='all',theme='legacy')
    assert r1['status']==r2['status']
    a=json.loads((root/'presentation-manifest.json').read_text())
    b_=json.loads((legacy/'presentation-manifest.json').read_text())
    assert a['style_version']=='scientific-latex/v1'
    assert b_['style_version']=='monochrome-scientific/v1'
    assert a['visual_elements']==b_['visual_elements']
    assert a['visual_links']==b_['visual_links']
    assert a['input_artifact_hashes']==b_['input_artifact_hashes']
    assert (root/'presentation-health.json').read_bytes()==(legacy/'presentation-health.json').read_bytes()
    assert b'Latin Modern Roman' in (root/'architecture.svg').read_bytes()
    tree=ET.parse(root/'architecture.svg')
    ns='{http://www.w3.org/2000/svg}'
    assert any(' L ' in (p.attrib.get('d') or '') for p in tree.findall(f'.//{ns}path') if p.attrib.get('data-ir-edge-ids'))
    assert b'.. controls' not in (root/'architecture.tex').read_bytes()


def test_orthogonal_paths_avoid_nonendpoint_block_interiors(tmp_path):
    b=load_bundle(bundle(tmp_path))
    g=project_flow(b,level='block',focus='ResidualBlock')
    # The fixture's component name may be present only in its real definition.
    assert not any(i['severity']=='error' for i in g['issues'])
    scene=make_flow_scene(g,status='partial',entrypoint='ResidualBlock')
    leaves=[r for r in scene['rects'] if r.get('id') and r['kind']!='flow-module']
    for path in scene['paths']:
        pts=path['points']
        assert len(pts)>=2
        for (x1,y1),(x2,y2) in zip(pts,pts[1:]):
            assert x1==x2 or y1==y2
            for r in leaves:
                x,y,w,h=r['x'],r['y'],r['w'],r['h']
                if x1==x2 and x+0.5<x1<x+w-0.5:
                    assert max(y1,y2)<=y+0.5 or min(y1,y2)>=y+h-0.5
                if y1==y2 and y+0.5<y1<y+h-0.5:
                    assert max(x1,x2)<=x+0.5 or min(x1,x2)>=x+w-0.5


def test_repeatable_scientific_pdf_and_tikz(tmp_path):
    b=bundle(tmp_path,model='mlp_mvp',entrypoint='model:MLP')
    present(b,tmp_path/'one',view='flow',format='all',theme='scientific-latex')
    present(b,tmp_path/'two',view='flow',format='all',theme='scientific-latex')
    for name in ('architecture.svg','architecture.pdf','architecture.tex',
                 'presentation-health.json','presentation-manifest.json'):
        assert (tmp_path/'one'/name).read_bytes()==(tmp_path/'two'/name).read_bytes(),name
    assert (tmp_path/'one'/'architecture.pdf').read_bytes().startswith(b'%PDF')
    assert b'\\usepackage{lmodern}' in (tmp_path/'one'/'architecture.tex').read_bytes()
