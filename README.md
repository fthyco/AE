# MAGG — Model Architecture Graph Generator

MAGG is designed as **two independent components**:

1. **SAE (Static Architecture Extractor):** reads source code *without running it*, captures maximal source-grounded architecture facts into a deterministic intermediate representation (IR), and preserves exact source bytes.
2. **Presenter (future):** converts the IR into a unified research-paper-style model architecture figure. It may group and simplify, but must retain links to the original facts.

> **Extract at maximum fidelity. Abstract only when presenting.**

## SAE v0.1.0 — working first implementation

Supports Python 3.11+. PyTorch is **not** required: models are never imported, initialized, trained or traced.

```bash
python -m pip install -e '.[dev]'
magg extract examples/simple_mlp --entrypoint model:MLP --output magg-output
magg inspect magg-output
magg validate magg-output
pytest -q
```

Or run without installing using `PYTHONPATH=src python -m magg.cli ...`.

`magg extract` writes:

```text
magg-output/
├── architecture.ir.json    # Full linked AST + supported semantic facts
├── source-index.json       # Tokens, comments and original byte offsets
├── source-manifest.json    # Included/excluded file inventory and hashes
├── coverage.json           # Source/parse and unresolved counts
├── diagnostics.json        # Parse and analysis diagnostics
├── extraction-meta.json   # Artifact checksums and verification summary
└── sources/                # Byte-exact copy of in-scope files
```

### Scope and accuracy

SAE captures every AST node and lexical token of every parsable included `.py` file; it also records supported definitions, `torch.nn` construction sites, argument expressions, operations, conditional syntax and module-use links. Every fact has a source position. Unresolved symbolic constructor arguments and ambiguous module-use links remain explicit. The analyzer makes **no claim** of complete Python semantic inference or runtime behavior. Only `.py` and common text configurations are included by default; everything else is recorded as excluded. See [implemented features and limitations](docs/limitations.md).

The `--entrypoint` is a metadata selector in v0.1.0; it does not filter the syntax graph. All included Python sources are analyzed.

### Project hierarchy

```text
AE/
├── README.md, LICENSE, pyproject.toml
├── docs/
│   ├── SAE_Technical_Specification.md
│   ├── ir-schema.md
│   └── limitations.md
├── schemas/
│   └── architecture-ir.v1.schema.json
├── src/magg/
│   ├── cli.py
│   └── sae/
│       ├── api.py              # Extract, validate, inspect
│       ├── settings.py
│       ├── inventory/walker.py # Source inventory, exclusions
│       ├── parsing/            # Python AST/tokens and source spans
│       ├── analysis/analyzer.py# Semantic facts and links
│       ├── ir/                 # Graph, stable IDs, validation
│       └── output/bundle.py   # Canonical bundle and source snapshot
├── examples/simple_mlp/
└── tests/
```

A future `presenter/` remains deliberately unimplemented: the output is data, not a drawing.

### Design documentation

- [Full SAE specification](docs/SAE_Technical_Specification.md)
- [IR data contract](docs/ir-schema.md)
- [Implemented coverage and limitations](docs/limitations.md)

See the technical specification for the intended expanded package hierarchy, rules, milestones and acceptance criteria. Modules that are still planned have not been represented by fake empty implementations.
