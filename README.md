# Architecture Extractor (AE)

AE extracts source-backed neural-network architecture data from Python/PyTorch repositories and renders static architecture diagrams. Target code is not executed.

## Install

Python 3.11+.

```bash
pip install -e '.[dev,render]'
```

## Commands

```bash
ae extract examples/cnn_residual_mvp --entrypoint model:CNN --output ae-bundle
ae validate ae-bundle
ae inspect ae-bundle
ae present ae-bundle --view flow --level block --theme scientific-latex --format all --output figures
```

Views: `structure` (module/call relationships), `flow` (supported static value dependencies). Levels: `overview`, `block`, `operation`. Use `--focus ResidualBlock` for an individual module.

## Data

**Extractor:** `architecture.ir.json`, `model-index.json`, source and token indexes, file hashes, coverage, diagnostics, and source snapshots.

**Presenter:** SVG, PDF, TikZ, `presentation-health.json`, and `presentation-manifest.json`. Visual elements reference source-backed IR IDs.

Health statuses: `complete`, `partial`, `blocked`. Publication mode rejects partial diagrams unless `--allow-partial` is supplied.

## Scope

Supported: static Python analysis, PyTorch module discovery, local call/value relationships, conditional branches, fixed-position collections, and conditional symbolic shapes.

Unsupported: arbitrary dynamic execution, complete runtime tensor flow, general shape inference, and unresolved imports or operations.

## Tests

```bash
python -m pytest -q
```

Technical references: [architecture](docs/architecture.md), [IR](docs/ir-schema.md), [limitations](docs/limitations.md), [theme](docs/scientific-latex-theme.md).
