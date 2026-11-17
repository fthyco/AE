# AE v0.19 — SMP U-Net (ResNet34) full instance-forward graph

## Scope

The `instance` view supports the **unchanged** `segmentation-models-pytorch==0.6.0` U-Net default topology, with a ResNet34 encoder from `torchvision==0.25.0`. The extractor reads only source files and an explicit concrete NCHW shape; it does not import or execute either model package. The independent verifier instantiates the original model with `encoder_weights=None` to avoid network downloads. Encoder weights do not change this verified topology.

The bounded source proof requires all seven upstream files at their individually audited SHA-256 digests. Each file is byte-for-byte identical to the installed originals. Unsupported source changes, missing implementations, nondefault topology, and unsupported input geometry produce `status=unsupported` with zero speculative forward nodes or edges.

Supported input: RGB NCHW batch with positive dimensions, H/W at least 32 and divisible by 32. This restriction is an analysis precondition, not a claim that the upstream model rejects other sizes.

## Forward graph

- Resolves `Unet` → inherited `SegmentationModel.forward`, `ResNetEncoder.forward`, `UnetDecoder.forward`, and five `UnetDecoderBlock.forward` invocations
- Unrolls 16 ResNet34 BasicBlocks in `[3, 4, 6, 3]`, including residual additions and all 3 projection branches
- Retains the six feature maps and reverses their selection order, then connects **four multi-scale concatenation skips**
- Emits five target-size reads and interpolations, decoder Conv2d/BatchNorm/ReLU leaves, Identity attentions, and SegmentationHead leaves
- Separates module identity from call identity (e.g. two calls to one encoder BasicBlock ReLU)
- Includes FX-style source-backed non-Tensor shape-metadata operations; these nodes have `shape=null`, not fabricated Tensor shapes. Metadata operand edges are explicitly distinguished from Tensor operand edges

The graph has **198 operation/metadata nodes**, **227 ordered operand edges**, **149 leaf `call_module` events**, **16 residual adds**, **4 concats**, and **5 interpolate operations** in the audited topology.

## Independent acceptance gate

```bash
python scripts/verify_full_unet_resnet34.py --output benchmark-output/full-unet-resnet34
```

The verifier runs outside the extractor, using original SMP/TorchVision model packages. For the selected input shapes `1×3×64×64`, `1×3×96×96`, `1×3×224×224`, and `1×3×64×96`, it checks **every FX operation and ordered dependency**, each FX Tensor shape (plus the 22 non-Tensor metadata nodes), runtime invocation order and output shape for every leaf module, shared-module identities, exact source hashes and provenance, and 1:1 graphical projection to SVG/PDF/TikZ. A separate hash-seed test asserts byte-identical IR. CI also blocks imports of the target packages during extraction and records an isolated extractor process's elapsed time and peak RSS.

These are model-and-input-scoped acceptance gates, not proof of completeness for arbitrary SMP configurations, other encoder families, modified forks, training-only branches, or general dynamic Python. Publisher health remains `Partial`; independent execution evidence lives in the verifier report, not in the static bundle.

## Usage

Stage the seven unchanged source files into the paths documented by `scripts/verify_full_unet_resnet34.py` and extract:

```bash
ae extract source-snapshot --entrypoint decoders.unet.model:Unet \
  --input-shape 1,3,64,64 --output ae-bundle
ae validate ae-bundle
ae present ae-bundle --view instance --level operation \
  --format all --output figures
```
