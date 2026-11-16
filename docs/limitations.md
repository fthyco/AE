# Implementation status and analysis limitations

## Available in v0.1.0

- Deterministic source inventory with excludes, size/file limits and symlink refusal.
- Byte-exact source snapshots; original bytes and SHA-256 in manifests.
- Python tokenizer, comments, every AST node, exact/enclosing source spans.
- Definitions, imports, method scopes, assignments, calls and their arguments, operators, returns, guards, branches and loops.
- Static hints for `torch.nn` constructor sites, `self.<attribute>` bindings, repeated module call sites, symbolic arguments and unresolvable module references.
- Canonical JSON, referential integrity validation, checksums, `validate` and `inspect` commands.

## Not yet available

- Complete cross-file symbol resolution and shadow-sensitive lexical scope analysis.
- Precise SSA/data-flow reaching definitions, alias analysis, nontrivial symbolic shape inference, all Python metaprogramming patterns.
- Configuration semantic evaluation (config files are copied without execution).
- PyTorch third-party extension constructor recognition beyond `torch.nn.*` import-resolution hints.
- Presenter or paper-style drawing.

**Accuracy boundary:** a syntax graph is comprehensive for parsable Python files within scope; semantic extraction is *not* complete or proven to capture all runtime relationships. `EXPRESSION_DEPENDS_ON` is syntactic operand nesting, not a claim about runtime tensor flow. An unresolved relation stays explicitly unresolved.

## Safety

Target Python is never imported or executed. The extractor does not use `eval`, `exec`, `torch.load`, or pickle. Files are read as untrusted data. The inventory does not follow symlinks. Large or unsupported files are reported as exclusions.

## Next milestones

M2: true lexical scopes and cross-file imports. M3: conservative def-use and alias analysis. M4: versioned symbolic shape rules. M5: full verification coverage and Presenter contract tests.
