# AE v0.16 — Bounded instance-forward graph

Source-only expansion is available for the default `resnet18` factory backed by the verified TorchVision `BasicBlock` / `ResNet` constructor and forward patterns.

```bash
ae extract <source-dir> --entrypoint resnet:resnet18 --input-shape 1,3,64,64 --output bundle
ae validate bundle
ae present bundle --view instance --level operation --format all --output figures
```

`instance-forward.json` and `architecture.ir.json` contain per-call instance identities, source references, concrete shapes conditional on the supplied input, and explicit ordered operand edges. In-place ReLU reuse retains one module identity and distinct invocation sites. Each projection `downsample` is expanded into its two registered leaves.

Unsupported or modified source patterns return `unsupported` with an issue, not a guessed graph. No model import or execution happens during extraction. The `instance` Presenter reports `partial` until independent execution verification is supplied outside the bundle; its output manifest still maps every derived node and edge to the instance IR.

## Independent gate

```bash
python scripts/verify_full_resnet18.py --output benchmark-output/full-resnet18
```

The verifier imports unmodified TorchVision **only in the test process**, traces it with FX, runs ShapeProp, and independently invokes the real `forward()` with leaf-module hooks. For each of three input sizes (`64×64`, `96×96`, `224×224`), it checks every expanded node's operation and module identity, every operand dependency and every shape. Runtime hooks check the actual leaf-module invocation order and outputs. Source snapshot SHA-256 and an audit JSON are recorded.

Current pinned reference: `torchvision 0.25.0+cpu`, `torch 2.10.0+cpu`, original `resnet.py` SHA-256 `9faf2f321fdaf7edbd791aedc6e3610c7edf9c892112cbd75861beae1e9518d1`.

**Pass criteria** (per tested input): 71/71 operation identities, 78/78 dependencies, 71/71 concrete shapes, 60/60 leaf runtime invocations, and all 52 distinct leaf module targets present. The verified scope is the default factory, selected input shapes, and the tested upstream implementation. It does not imply arbitrary kwargs, runtime branches, or other models are fully supported.
