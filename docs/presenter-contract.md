# Presenter handoff contract (design-only)

The future MAGG Presenter receives the **validated output bundle** as data:

- `architecture.ir.json`: graph nodes and relations, with typed evidence and provenance;
- `model-index.json`: entrypoint-scoped modules, components, forward calls and IR relation IDs;
- `source-index.json`: complete addressable token index;
- `source-manifest.json`, `sources/`: original bytes and hashes;
- `coverage.json`, `diagnostics.json`, `extraction-meta.json`: confidence boundaries and reproducibility data.

Presenter may group nodes, hide detail, choose levels of abstraction, lay out blocks and render research-paper architecture figures. It **must** link every rendered block and arrow to contributing IR IDs and distinguish unresolved or conditional semantic links from proven structure. No execution, source parsing or reinterpretation of ambiguous links belongs in Presenter.

Presenter implementation is intentionally out of scope for the SAE v0.1 development branch.

The index exposes lexical order, not runtime execution order. See docs/mvp-acceptance.md.
