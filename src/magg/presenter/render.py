"""Monochrome publication-style renderers sharing one geometric scene."""
from __future__ import annotations

from xml.sax.saxutils import escape as xml_escape


def _f(n: float) -> str:
    return f'{n:.1f}'


def svg(scene: dict) -> bytes:
    w,h=scene['width'],scene['height']
    a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img">',
       f'<title>{xml_escape(scene["title"])}</title>',
       '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#777"/></marker></defs>',
       '<rect width="100%" height="100%" fill="white"/>']
    for r in scene['rects']:
        dash='' if r['dash']=='none' else f' stroke-dasharray="{r["dash"]}"'
        vid=f' id="v-{r["id"]}" data-ir-ids="{",".join(r["ir_ids"])}"' if r.get('id') else ''
        a.append(f'<rect{vid} x="{_f(r["x"])}" y="{_f(r["y"])}" width="{_f(r["w"])}" height="{_f(r["h"])}" rx="4" fill="{r["fill"]}" stroke="#555" stroke-width="1"{dash}/>')
    for p in scene['paths']:
        dash='4,4' if p['kind']=='reference' else '2,5'
        d=(f'M {_f(p["x1"])} {_f(p["y1"])} C {_f(p["cx1"])} {_f(p["cy1"])} '
           f'{_f(p["cx2"])} {_f(p["cy2"])} {_f(p["x2"])} {_f(p["y2"])}')
        a.append(f'<path id="l-{p["id"]}" data-ir-edge-ids="{",".join(p["ir_edge_ids"])}" d="{d}" fill="none" stroke="#777" stroke-width="1.15" stroke-dasharray="{dash}" marker-end="url(#arrow)"/>')
    for t in scene['texts']:
        weight='bold' if t['bold'] else 'normal'
        a.append(f'<text x="{_f(t["x"])}" y="{_f(t["y"])}" fill="#111" font-family="DejaVu Sans, Arial, sans-serif" font-size="{t["size"]}" font-weight="{weight}">{xml_escape(t["text"])}</text>')
    a.append('</svg>')
    return ('\n'.join(a)+'\n').encode('utf-8')


def pdf(scene: dict) -> bytes:
    try:
        import cairosvg
    except ImportError as exc:
        raise ValueError('PDF export requires CairoSVG; install magg-sae[render]') from exc
    return cairosvg.svg2pdf(bytestring=svg(scene))


def _tex_escape(text: str) -> str:
    mapping={'\\':r'\textbackslash{}','{':r'\{','}':r'\}','#':r'\#','$':r'\$',
             '%':r'\%','&':r'\&','_':r'\_','^':r'\textasciicircum{}',
             '~':r'\textasciitilde{}'}
    return ''.join(mapping.get(c,c) for c in text)


def tikz(scene: dict) -> bytes:
    """Use exact layout coordinates from the same scene; no second graph extraction."""
    a=[r'\documentclass[tikz,border=0pt]{standalone}',
       r'\usepackage[T1]{fontenc}',r'\usepackage{lmodern}',
       r'\begin{document}',r'\begin{tikzpicture}[x=1pt,y=-1pt,>=stealth]',
       f'\\path[use as bounding box] (0,0) rectangle ({scene["width"]},{scene["height"]});',
       f'\\fill[white] (0,0) rectangle ({scene["width"]},{scene["height"]});']
    for r in scene['rects']:
        fill='white' if r['fill']=='#ffffff' else 'black!9' if r['fill']=='#e9e9e9' else 'black!4'
        line='dashed' if r['dash']!='none' else 'solid'
        x,y,w,h=(_f(r[z]) for z in ('x','y','w','h'))
        a.append(f'\\draw[draw=black!65,fill={fill},{line},line width=0.7pt,rounded corners=4pt] ({x},{y}) rectangle ({_f(r["x"]+r["w"])},{_f(r["y"]+r["h"])});')
    for p in scene['paths']:
        dash='dashed' if p['kind']=='reference' else 'dotted'
        pts=[f'({_f(p[z])},{_f(p[t])})' for z,t in [('x1','y1'),('cx1','cy1'),('cx2','cy2'),('x2','y2')]]
        a.append(f'\\draw[->,{dash},draw=black!55,line width=0.8pt] {pts[0]} .. controls {pts[1]} and {pts[2]} .. {pts[3]};')
    for t in scene['texts']:
        txt=_tex_escape(t['text']).replace('…','...')
        font='\\bfseries' if t['bold'] else ''
        a.append(f'\\node[anchor=base west,inner sep=0,outer sep=0,text=black,font=\\fontsize{{{t["size"]}}}{{{t["size"]+2}}}\\selectfont{font}] at ({_f(t["x"])},{_f(t["y"])}) {{{txt}}};')
    a.extend([r'\end{tikzpicture}',r'\end{document}',''])
    return ('\n'.join(a)).encode('utf-8')
