"""A small immutable visual style modeled on Latin Modern research figures.

Font files are resolved by the PDF renderer from the *local* TeX installation
and are never packaged or redistributed by MAGG.
"""
from __future__ import annotations

from types import MappingProxyType

SCIENTIFIC_LATEX = MappingProxyType({
    'name': 'scientific-latex/v1',
    'background': '#ffffff',
    'ink': '#181818',
    'stroke': '#181818',
    'connection': '#181818',
    'header_fill': '#e9e9e9',
    'block_fill': '#f4f4f4',
    'module_fill': '#ffffff',
    'font_family': 'Latin Modern Roman, CMU Serif, Computer Modern, serif',
    'border_width': 0.60,
    'emphasis_width': 1.15,
    'edge_width': 0.85,
    'corner_radius': 1,
    'arrow_size': 5.5,
})
LEGACY = MappingProxyType({
    'name': 'monochrome-scientific/v1',
    'background': '#ffffff',
    'ink': '#111111',
    'stroke': '#555555',
    'connection': '#777777',
    'header_fill': '#e9e9e9',
    'block_fill': '#f5f5f5',
    'module_fill': '#ffffff',
    'font_family': 'DejaVu Sans, Arial, sans-serif',
    'border_width': 1.0,
    'emphasis_width': 1.0,
    'edge_width': 1.15,
    'corner_radius': 4,
    'arrow_size': 7.0,
})


def select_theme(name: str = 'scientific-latex') -> dict:
    if name == 'scientific-latex':
        return dict(SCIENTIFIC_LATEX)
    if name == 'legacy':
        return dict(LEGACY)
    raise ValueError(f'Unknown presenter theme: {name}')
