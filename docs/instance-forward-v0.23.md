# AE v0.23 — Source-proven VGG16

Original TorchVision 0.25.0 vgg16(weights=None).eval().
Source pin: torchvision/models/vgg.py SHA-256 214a5a421d7e5693315bb91c5087b47431660f39c905c97a806cc54840dfb04f.

Model scope: 42 FX operations, 41 ordered operand edges, 39 original leaf calls.
Audited RGB input shapes are NCHW (1,3,64,64), (1,3,128,128), (1,3,224,224), and (2,3,224,224).
Default 13-conv VGG-D blueprint plus five pooling boundaries is read from the pinned original annotated cfgs AST; default classifier uses three Linear modules.

AE never imports or executes original model source. A separate independent reference script uses torch.fx, ShapeProp and forward hooks to verify ordered nodes, edges, shapes and vector figure mappings.

Two-level rendering: Architecture overview four groups; block view eight groups including five original convolution stages; operation detail 42 nodes. Focus stage.2 preserves original source references and records omitted operation/edge IDs.

CLI:
    ae extract SOURCE --entrypoint torchvision.models.vgg:vgg16 --input-shape 1,3,224,224 --output BUNDLE
    ae present BUNDLE --view instance --level overview --format all --output OVERVIEW
    ae present BUNDLE --view instance --level block --format all --output BLOCKS
    ae present BUNDLE --view instance --level operation --focus stage.2 --format all --output DETAIL

Unsupported source changes, weights/configuration variants, and untested geometries fail closed. This is a bounded audited specialization, not universal VGG tracing.

## Unified semantic visual cues

Only source-proven operation classes receive colored 3-point accent rules,
shared by SVG, PDF and TikZ. Labels and original IR IDs remain authoritative.
The scientific white/black baseline is preserved; each function category
uses a muted blue/teal/amber/violet/slate/graphite accent, and unresolved
or unclassified types remain neutral. Grouped Architecture blocks use
neutral accents because they aggregate multiple operation categories.

Color is never evidence of execution; exact original node/edge provenance
and tensor shapes remain in the canonical graph and figure manifest.
