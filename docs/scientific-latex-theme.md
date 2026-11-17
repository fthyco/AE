# Architecture Extractor — Scientific LaTeX Theme

Theme: `scientific-latex` (default). Alternative: `legacy`.

## Style

| Element | Value |
| --- | --- |
| Background | White `#FFFFFF` |
| Text / border / arrows | `#181818` |
| Header | `#E9E9E9` |
| Operation | `#F4F4F4` |
| Font | Latin Modern Roman; serif fallback |
| Border | 0.60 pt; emphasized 1.15 pt |
| Arrow | 0.85 pt; orthogonal routes |
| Corners | 1 pt |

## Routing

Layered columns with orthogonal gutters and reserved skip channels. Solid links show static value dependencies; dashed links show conditional relations; dotted links show guarded or reference relations. Every connection retains its source evidence.

## Export

SVG, PDF, and TikZ share one geometric scene. SVG includes source IDs; `presentation-manifest.json` retains all visual-to-IR references. PDF uses installed Latin Modern fonts when available; TikZ uses `lmodern`.

## Example

```bash
ae present ae-bundle --view flow --focus ResidualBlock --theme scientific-latex --format all --output figures
```
