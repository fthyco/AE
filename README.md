# Architecture executer (AE)

AE generates **source-grounded neural-network architecture diagrams** from Python/PyTorch repositories. It analyzes source code without importing or executing the target model.

## Install

Requires Python 3.11+.

```bash
pip install -e '.[dev,render]'
```

## Use

```bash
# Extract the selected model into a verifiable intermediate representation.
ae extract examples/cnn_residual_mvp --entrypoint model:CNN --output ae-bundle

# Check source snapshots, graph references, and artifact hashes.
ae validate ae-bundle

# Render the architecture. Each figure retains links to source evidence.
ae present ae-bundle --view flow --level block \
  --theme scientific-latex --format all --output figures
```

`ae inspect ae-bundle` summarizes extraction coverage. To isolate a block, add `--focus ResidualBlock`. Choose `--view structure` for the module/call hierarchy, or `--view flow` for supported static value dependencies. Detail levels: `overview`, `block`, `operation`.

## Technical design

- **Static Architecture Extractor (SAE):** preserves exact source bytes, Python AST/tokens, definitions, PyTorch construction sites, imports, calls, operations, supported intra-/interprocedural value relations, conditional symbolic shapes, and unresolved facts. Produces deterministic, versioned JSON with source locations and evidence IDs.
- **Presenter:** reads validated SAE output only. Projects modules/operations into a visual graph, applies source-aware health checks (`complete`, `partial`, `blocked`), and renders SVG, PDF, and TikZ. Its default `scientific-latex` theme uses monochrome vector blocks and orthogonal connectors.

The extractor never fabricates runtime tensor flow, unknown dimensions, or missing definitions. The Presenter does not reanalyze the target code.

## Output

Extraction writes `architecture.ir.json`, `model-index.json`, `source-index.json`, `source-manifest.json`, `coverage.json`, `diagnostics.json`, `extraction-meta.json`, and byte-exact `sources/` snapshots.

Rendering writes `architecture.svg` / `architecture.pdf` / `architecture.tex` (according to `--format`), `presentation-health.json`, and `presentation-manifest.json` with visual-element-to-IR mappings. A partial diagram is withheld in `--mode publication` unless `--allow-partial` is explicit.

## Verification

```bash
python -m pytest -q
```

See [architecture and evidence contract](docs/architecture.md), [IR schema](docs/ir-schema.md), and [limitations](docs/limitations.md).