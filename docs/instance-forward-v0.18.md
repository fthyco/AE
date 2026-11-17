# DenseNet121 — Bounded full forward expansion

AE 0.18 extends source-only `instance-forward.json` to the unmodified TorchVision 0.25.0 DenseNet121 factory. The extractor never imports or executes that implementation.

## Scope

Factory: `densenet121(weights=None)` with default constructor arguments. Input: concrete positive NCHW RGB shape with a spatial extent sufficient for three transition pools. The current source assertion is pinned to the audited SHA-256 of the complete original TorchVision `densenet.py` file: `76a71797b7873088c27ca0458588861c197528e86421f57525dc8e3ad771c59a`. Any source changes fail closed. This is a bounded specialization, not general inference of arbitrary DenseNet forks or configurations.

The factory's existing `container_specialization` supplies 4 DenseBlocks, `[6, 12, 24, 16]` registered DenseLayer instances, and three transition modules. For each layer, the expander represents `torch.cat(inputs, 1)` with a separate dependency on **every previously accumulated feature** followed by BatchNorm, ReLU and Convolution chains. It emits the final per-block `cat`, the transition and stem modules, the final normalization, functional pooling and classifier. IDs are instance-aware, deterministic, and tied to exact source bytes and source/registration IR IDs.

## Independent verification

`python scripts/verify_full_densenet121.py` is a separate program allowed to import and run the upstream model. It verifies operation identities, ordered operand edges, complete FX tensor shapes, ordered leaf runtime hook invocations and shapes, final output, IR provenance, and 1:1 projection to SVG/PDF/TikZ. It tests three input shapes `(1, 3, 64, 64)`, `(1, 3, 96, 96)`, `(1, 3, 224, 224)`.

Acceptance for this pinned source and selected inputs: **433 operation nodes**, **967 dependency edges**, **433 shape matches**, **366 ordered runtime module calls**, and **433 diagram nodes / 967 diagram edges**. Cross-hash-seed IR determinism and conservative source-mutation tests are required.

Static Publisher health remains `Partial` without embedded runtime verification. The separate verification report explicitly records the verified input cases. Arbitrary dynamic branches, modified source, nondefault kwargs, and other DenseNet variants are not asserted complete.
