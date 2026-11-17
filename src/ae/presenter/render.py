"""One geometrical scene, three vector exports, shared scientific visual tokens."""
from __future__ import annotations

from math import atan2, cos, sin
from xml.sax.saxutils import escape as xml_escape

from .themes import select_theme


def _f(n: float) -> str:
    return f'{n:.2f}'


def _style(scene: dict) -> dict:
    return scene.get('style') or select_theme('scientific-latex')


def _polyline(path: dict) -> list[tuple[float,float]]:
    if 'points' in path:
        return [tuple(p) for p in path['points']]
    # This fallback is for the historical Structure view. The old curved
    # control points are NOT semantic evidence; only endpoints are.
    return [(path['x1'],path['y1']),
            ((path['x1']+path['x2'])/2,path['y1']),
            ((path['x1']+path['x2'])/2,path['y2']),
            (path['x2'],path['y2'])]


def _dash(kind: str) -> tuple[int, ...]:
    if kind == 'static-value':
        return ()
    if kind == 'reference':
        return (5, 3)
    return (2, 4)


def svg(scene: dict) -> bytes:
    w,h = scene['width'],scene['height']
    s = _style(scene)
    family = xml_escape(s['font_family'])
    size = s['arrow_size']
    a = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img">',
         f'<title>{xml_escape(scene["title"])}</title>',
         f'<defs><marker id="arch-arrow" viewBox="0 0 8 8" refX="7.5" refY="4" markerWidth="{_f(size)}" markerHeight="{_f(size)}" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="{s["connection"]}"/></marker></defs>',
         f'<rect width="100%" height="100%" fill="{s["background"]}"/>']
    for r in scene['rects']:
        dash = '' if r['dash']=='none' else f' stroke-dasharray="{r["dash"]}"'
        vid = f' id="v-{r["id"]}" data-ir-ids="{",".join(r["ir_ids"])}"' if r.get('id') else ''
        color = s['header_fill'] if r['kind']=='header' else s['block_fill'] if r['fill'] not in ('#ffffff', '#e9e9e9') else s['module_fill'] if r['fill']=='#ffffff' else s['header_fill']
        strength = s['emphasis_width'] if r['kind'] in ('flow-module','model','module') else s['border_width']
        a.append(f'<rect{vid} x="{_f(r["x"])}" y="{_f(r["y"])}" width="{_f(r["w"])}" height="{_f(r["h"])}" rx="{_f(s["corner_radius"])}" fill="{color}" stroke="{s["stroke"]}" stroke-width="{_f(strength)}"{dash}/>')
    for p in scene['paths']:
        pts = _polyline(p)
        if len(pts)<2:
            continue
        d='M '+ ' L '.join(f'{_f(x)} {_f(y)}' for x,y in pts)
        dash=_dash(p['kind'])
        dash_attr= f' stroke-dasharray="{",".join(map(str,dash))}"' if dash else ''
        a.append(f'<path id="l-{p["id"]}" data-ir-edge-ids="{",".join(p["ir_edge_ids"])}" d="{d}" fill="none" stroke="{s["connection"]}" stroke-width="{_f(s["edge_width"])}" stroke-linejoin="miter"{dash_attr} marker-end="url(#arch-arrow)"/>')
    for t in scene['texts']:
        weight = 'bold' if t['bold'] else 'normal'
        a.append(f'<text x="{_f(t["x"])}" y="{_f(t["y"])}" fill="{s["ink"]}" font-family="{family}" font-size="{_f(t["size"])}" font-weight="{weight}">{xml_escape(t["text"])}</text>')
    a.append('</svg>')
    return ('\n'.join(a)+'\n').encode('utf-8')


def _pdf_font_pair() -> tuple[str,str]:
    """Embed the actual Latin Modern Type1 faces where TeX is installed."""
    try:
        from pathlib import Path
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.pdfmetrics import EmbeddedType1Face, Font
        common = Path('/usr/share/texmf/fonts')
        for logical, stem in (('AELatinModern','lmr10'),('AELatinModernBold','lmbx10')):
            afm = common/'afm/public/lm'/f'{stem}.afm'
            pfb = common/'type1/public/lm'/f'{stem}.pfb'
            if not (afm.is_file() and pfb.is_file()):
                return ('Times-Roman','Times-Bold')
            if logical not in pdfmetrics.getRegisteredFontNames():
                face=EmbeddedType1Face(str(afm),str(pfb))
                pdfmetrics.registerTypeFace(face)
                pdfmetrics.registerFont(Font(logical,face.name,'WinAnsiEncoding'))
        return ('AELatinModern','AELatinModernBold')
    except (OSError,ValueError):
        return ('Times-Roman','Times-Bold')


def pdf(scene: dict) -> bytes:
    """Invariant vector PDF, same rectangles, coordinates, and orthogonal paths."""
    try:
        from io import BytesIO
        from reportlab.pdfgen import canvas
        from reportlab.lib.colors import HexColor
    except ImportError as exc:
        raise ValueError('PDF export requires ReportLab; install architecture-executer[render]') from exc
    s = _style(scene)
    font, font_bold = _pdf_font_pair() if s['name']=='scientific-latex/v1' else ('Helvetica','Helvetica-Bold')
    memory = BytesIO()
    c=canvas.Canvas(memory,pagesize=(scene['width'],scene['height']),bottomup=0,invariant=1,pageCompression=1)
    c.setTitle(scene['title'])
    c.setFillColor(HexColor(s['background']))
    c.rect(0,0,scene['width'],scene['height'],stroke=0,fill=1)
    for r in scene['rects']:
        fill = s['header_fill'] if r['kind']=='header' else s['block_fill'] if r['fill'] not in ('#ffffff','#e9e9e9') else s['module_fill'] if r['fill']=='#ffffff' else s['header_fill']
        c.setStrokeColor(HexColor(s['stroke']))
        c.setFillColor(HexColor(fill))
        c.setLineWidth(s['emphasis_width'] if r['kind'] in ('flow-module','model','module') else s['border_width'])
        c.setDash([] if r['dash']=='none' else [4,3])
        c.roundRect(r['x'],r['y'],r['w'],r['h'],s['corner_radius'],stroke=1,fill=1)
    c.setDash([])
    for p in scene['paths']:
        pts=_polyline(p)
        if len(pts)<2:
            continue
        c.setStrokeColor(HexColor(s['connection']))
        c.setFillColor(HexColor(s['connection']))
        c.setLineWidth(s['edge_width'])
        c.setDash(list(_dash(p['kind'])))
        path=c.beginPath(); path.moveTo(*pts[0])
        for point in pts[1:]:
            path.lineTo(*point)
        c.drawPath(path,stroke=1,fill=0)
        x2,y2=pts[-1]
        previous=next((pt for pt in reversed(pts[:-1]) if pt!=(x2,y2)),None)
        if previous:
            angle=atan2(y2-previous[1],x2-previous[0])
            sz=s['arrow_size']
            triangle=c.beginPath()
            triangle.moveTo(x2,y2)
            triangle.lineTo(x2-sz*cos(angle-0.45),y2-sz*sin(angle-0.45))
            triangle.lineTo(x2-sz*cos(angle+0.45),y2-sz*sin(angle+0.45))
            triangle.close()
            c.setDash([]);c.drawPath(triangle,stroke=0,fill=1)
    c.setDash([])
    c.setFillColor(HexColor(s['ink']))
    for t in scene['texts']:
        c.setFont(font_bold if t['bold'] else font,t['size'])
        c.drawString(t['x'],t['y'],t['text'].replace('…','...'))
    c.save()
    return memory.getvalue()


def _tex_escape(text: str) -> str:
    mapping={'\\':r'\textbackslash{}','{':r'\{','}':r'\}','#':r'\#','$':r'\$',
             '%':r'\%','&':r'\&','_':r'\_','^':r'\textasciicircum{}',
             '~':r'\textasciitilde{}'}
    return ''.join(mapping.get(c,c) for c in text)


def tikz(scene: dict) -> bytes:
    """TikZ uses Latin Modern (lmodern), the source document's LaTeX family."""
    s=_style(scene)
    a=[r'\documentclass[tikz,border=0pt]{standalone}',
       r'\usepackage[T1]{fontenc}',r'\usepackage{lmodern}',r'\usetikzlibrary{arrows.meta}',
       r'\begin{document}',r'\begin{tikzpicture}[x=1pt,y=-1pt,>=stealth]',
       f'\\path[use as bounding box] (0,0) rectangle ({scene["width"]},{scene["height"]});',
       f'\\fill[white] (0,0) rectangle ({scene["width"]},{scene["height"]});']
    for r in scene['rects']:
        shade='white' if r['fill']=='#ffffff' else 'black!9' if r['kind']=='header' or r['fill']=='#e9e9e9' else 'black!4'
        line='dashed' if r['dash']!='none' else 'solid'
        x,y,w,h=(_f(r[z]) for z in ('x','y','w','h'))
        strength=s['emphasis_width'] if r['kind'] in ('flow-module','model','module') else s['border_width']
        a.append(f'\\draw[draw=black!90,fill={shade},{line},line width={_f(strength)}pt,rounded corners={_f(s["corner_radius"])}pt] ({x},{y}) rectangle ({_f(r["x"]+r["w"])},{_f(r["y"]+r["h"])});')
    for p in scene['paths']:
        dash='solid' if p['kind']=='static-value' else 'dashed' if p['kind']=='reference' else 'dotted'
        pts=' -- '.join(f'({_f(x)},{_f(y)})' for x,y in _polyline(p))
        a.append(f'\\draw[-{{Stealth[length={_f(s["arrow_size"])}pt,width=3.2pt]}},{dash},draw=black!90,line width={_f(s["edge_width"])}pt] {pts};')
    for t in scene['texts']:
        txt=_tex_escape(t['text']).replace('…','...')
        font=r'\bfseries' if t['bold'] else ''
        a.append(f'\\node[anchor=base west,inner sep=0,outer sep=0,text=black,font=\\fontsize{{{_f(t["size"])}}}{{{_f(t["size"]+2)}}}\\selectfont{font}] at ({_f(t["x"])},{_f(t["y"])}) {{{txt}}};')
    a.extend([r'\end{tikzpicture}',r'\end{document}',''])
    return ('\n'.join(a)).encode('utf-8')
