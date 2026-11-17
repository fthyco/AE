# Architecture Extractor (AE)

Static Python/PyTorch architecture extraction and vector diagrams. Models are not imported or executed.

## Install

```bash
pip install -e '.[dev,render]'
```

## Commands

```bash
ae extract examples/cnn_residual_mvp --entrypoint model:CNN --output ae-bundle
ae validate ae-bundle
ae inspect ae-bundle
ae present ae-bundle --view structure --level operation --format all --output figures
```

## Output

SAE: source-backed IR, model index, source snapshots, hashes and unresolved records. Presenter: SVG, PDF, TikZ and health/provenance manifests.

Static factory specialization: ResNet block configuration; supported fixed-count nested containers (including DenseNet factory stages). Counts are conditional construction blueprints, not runtime tensor paths.

Nested shape propagation, FX source-site audit, and [instance-aware FX verification](docs/instance-fx-v0.15.md).

Bounded full forward expansion for the verified default ResNet18 and ResNet50 factories:

```bash
ae extract path/to/torchvision/models --entrypoint resnet:resnet18 --input-shape 1,3,64,64 --output ae-bundle
ae present ae-bundle --view instance --level operation --format all --output figures
```

The static instance graph is conditional on the input and source proof. [Scope and independent gate](docs/instance-forward-v0.16.md).

## Tests

```bash
python -m pytest -q
python scripts/verify_container_specialization.py
```

See docs/limitations.md and docs/container-specialization-v0.12.md.

ResNet50 Bottleneck support: [independent stage gate](docs/instance-forward-v0.17.md).

DenseNet121 (TorchVision 0.25.0 default factory): bounded, source-verified instance forward graph with complete feature-list concatenation dependencies for the audited input shapes. See [v0.18 scope and independent verification](docs/instance-forward-v0.18.md). Other variants and modified source remain unsupported.

SMP 0.6.0 U-Net/ResNet34 bounded, source-proven instance graph and external FX/hooks validation: [v0.19](docs/instance-forward-v0.19.md). Non-Tensor shape metadata is labeled separately; its nodes do not have Tensor shapes.

timm ViT-Tiny (1.0.22): bounded 12-block source-only instance-forward expansion with explicit `--attention-mode fused|unfused` and independent FX/hooks validation. See [v0.20](docs/instance-forward-v0.20.md). Other ViTs and modified sources remain unsupported.

Presenter v0.21: source-backed hierarchical instance views for the exact pinned ViT-Tiny reference. `ae present ae-bundle --view instance --level overview|block|operation` and `--level operation --focus blocks.0`. Overview/block group operation and edge provenance without modifying the underlying instance graph; the detail view remains 1:1. [Scope and evidence policy](docs/instance-hierarchy-v0.21.md).

MobileNetV3 Small: audited TorchVision 0.25.0 default `mobilenet_v3_small(weights=None)` source-only full forward extraction for 64/128/224 square RGB inputs. Independent FX, ShapeProp, runtime hooks, ordered edges and manifest checks; 159 operations / 173 dependencies. Original-source-only `overview`, `block` and `operation` views, with `--focus features.4`. [Scope and limitations](docs/instance-forward-v0.22.md). Arbitrary variants are unsupported.

VGG16 source-only (v0.23) default torchvision 0.25.0: verified 42 operations / 41 ordered dependencies / 39 leaf calls. Two-level source-evidenced overview/block/operation visualizations and stage focus. See docs/instance-forward-v0.23.md.
