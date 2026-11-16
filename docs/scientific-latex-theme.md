# MAGG scientific-latex theme — v0.7

## Purpose

Render **the same source-backed MAGG architecture information** using the visual conventions of the supplied Chessformer architecture report (a ten-page LaTeX/PDF reference): Latin Modern serif typography, grayscale blocks, thin outlines, small dark arrowheads, and right-angle connector routing.

This is a **Presenter-only** change. SAE source analysis, `architecture.ir.json`, `model-index.json`, diagnostic health status, and source evidence are not altered.

## Usage

```bash
magg extract examples/cnn_residual_mvp --entrypoint model:CNN --output sae-output
magg present sae-output --view flow --focus ResidualBlock \
  --theme scientific-latex --format all --output paper-figure
```

`scientific-latex` is the default. `--theme legacy` retains the former gray/sans-serif token style for comparison.

## Design tokens

| Feature | scientific-latex/v1 |
| --- | --- |
| Background | white `#FFFFFF` |
| Text, borders and arrows | near-black `#181818` |
| Header backgrounds | light gray `#E9E9E9` |
| Operation backgrounds | light gray `#F4F4F4` |
| Serif | Latin Modern Roman, medium/bold variants |
| Main border | 0.60 pt |
| Emphasized border | 1.15 pt |
| Arrow stroke | 0.85 pt |
| Corner radius | 1 pt |

All fonts are read **only from the user's local installed environment** and are never embedded in the MAGG source release. SVG requests Latin Modern by family name, with serif fallbacks. TikZ selects `lmodern`. The deterministic PDF exporter embeds genuine local Latin Modern Type 1 fonts when installed at the conventional TeX location; otherwise it uses the standard PDF Times fonts. Exact glyph shaping can differ between exporter/backends on machines without Latin Modern.

## Routing and truthfulness

- Forward dependencies are sorted deterministically into layered columns.
- Adjacent relationships route horizontally, then vertically through a gutter, then horizontally into the target.
- Skip dependencies use reserved upper channels between the header and node blocks; this exposes residual-like shortcuts without pretending that source-level evidence is verified runtime Tensor Flow.
- Each link retains its original IR edge IDs and source references. No link is added to make a drawing prettier.
- Solid = static value dependency. Dashed = conditional value. Dotted = guarded call or other conditional link.
- Syntax and Module references in `structure` mode remain distinct from the `flow` view.
- Unsupported/ambiguous execution remains `partial`/`unresolved`, even if diagram styling looks publication-ready.

## Export contract

The exact same geometric scene feeds SVG, PDF, and TikZ. No separate layout computations happen in exporters. SVG exposes `data-ir-ids` and `data-ir-edge-ids`; the manifest maps every visual item to original SAE entities. `presentation-manifest.json` declares `style_version: scientific-latex/v1`. Output files are byte-stable for fixed source, tool versions, and environment.

## Validation and known constraints

- Existing extraction and Presenter tests must pass unchanged.
- New regression tests check thematic tokens, immutability of visual evidence under theme switch, right-angle segments, lack of arrows cutting through unrelated node interiors, and byte-identical repeat SVG/PDF/TikZ exports.
- TikZ must compile with `pdflatex` where installed, and the PDF should visually render with Latin Modern Type1 fonts when available.
- Large graphs may remain physically wide. Focused module views provide readable individual figures; automatic multi-page routing and complete verified runtime Tensor Flow remain outside this theme work.

## Minimal code location

```text
src/magg/presenter/
├── themes/
│   ├── __init__.py
│   └── scientific_latex.py
├── flow_layout.py   # orthogonal channels
├── layout.py        # structure reference gutters
├── render.py        # consistent SVG / PDF / TikZ style
├── api.py           # --theme selection and manifest
└── ...
tests/test_scientific_theme.py
```
