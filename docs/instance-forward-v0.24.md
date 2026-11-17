# AE v0.24 — Source-proven EfficientNet-B0

Reference: original torchvision 0.25.0 efficientnet_b0(weights=None).eval().
Source pinned by exact SHA-256: torchvision/models/efficientnet.py, torchvision/ops/misc.py, torchvision/ops/stochastic_depth.py and torchvision/models/_utils.py.
Only default width_mult=1.0, depth_mult=1.0, dropout 0.2 and the audited RGB square shapes are proven.

Independent original execution and FX reference:
- 251 forward FX operations; 275 ordered operand dependencies.
- 214 traced leaf-module calls, verified against runtime hooks.
- 16 MBConv blocks, 16 Squeeze-and-Excitation gates, nine stochastic-depth functional calls, nine residual additions.
- 5,288,548 parameters and 1,000-class output.
- Audited inputs (1,3,64,64), (1,3,128,128), (1,3,224,224), (2,3,224,224).

The source-only extractor reads AST of original model construction, stages and forward branches and never imports the target model. The separate verifier executes TorchVision to produce independent torch.fx graphs, ShapeProp and real module hooks, and checks every node, shape, operand edge, source reference, and Presenter visual identity.

Architecture overview is six groups; block view is twelve groups with seven MBConv stages. Operation view shows all 251 original proven operations. A stage focus such as features.4 displays only the selected stage and immediate connection context; all hidden node/edge IDs are recorded.

CLI:
    ae extract SOURCE --entrypoint torchvision.models.efficientnet:efficientnet_b0 --input-shape 1,3,224,224 --output BUNDLE
    ae present BUNDLE --view instance --level overview --format all --output OVERVIEW
    ae present BUNDLE --view instance --level block --format all --output BLOCKS
    ae present BUNDLE --view instance --level operation --focus features.4 --format all --output DETAIL

Bounded scope: original pinned files, known factory and selected input geometries only. Unsupported dynamic construction, changed sources, variants or unproven tensor operators must fail closed, not invent edges.
