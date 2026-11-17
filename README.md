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
