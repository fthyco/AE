# MAGG v0.9 — fixed-position collection provenance

**Status:** implemented and tested as a conservative static analysis extension.  
**Purpose:** make explicit tuple/list element identities traceable through supported returns and destructuring, without executing target code.

## 1. Contract

- SAE remains the only producer of architectural evidence; Presenter consumes immutable SAE IR.
- Preserve every original source snapshot, AST/token and the complete v1 IR. New facts are additive.
- Nodes and edges added by this rule include stable IDs, original source spans and named versioned derivation rules.
- A tuple/list **aggregate is not a tensor**. Propagation of the shape of an element into the aggregate is forbidden.
- `Complete` in a flow view means supported **static source-dependency coverage**, not verified runtime behavior, true output ranks or full shape correctness.

## 2. Supported cases

1. An explicit, unstarred tuple/list expression, including `return logits, score`, yields a `forward_collection` and indexed `forward_collection_element` objects. `STATIC_COLLECTION_ELEMENT` links a proven source to each element. `STATIC_COLLECTION_MEMBER` links each element to its collection.
2. A local exact-length unpack of a known explicit collection, such as `pair = (x, y); a, b = pair`, creates indexed `forward_unpack` elements with `STATIC_COLLECTION_ELEMENT_BINDING` edges.
3. A uniquely resolved, nonrecursive, single-use local function or `Module.forward` returning a single, explicit fixed-length collection can bind returned positions to `a,b = module(x)` **only after** validating its arity and call arguments. Otherwise no indexed cross-function edge is emitted.
4. Conditional symbolic dimensions propagate through proven element-source and element-binding edges, never through collection containment itself. Existing unsupported shape operations remain unproven.
5. Presenter can display cross-module positional bindings using dotted, source-backed orthogonal paths on outer channels, without inventing runtime tensor flow. `--focus` intentionally hides links whose other endpoint is outside the focus view.

## 3. New IR relations

| Relation | Meaning | Proof boundary |
|---|---|---|
| `STATIC_COLLECTION_ELEMENT` | One supported source expression is the value of an explicit positional member | Only when that expression has a statically tracked source |
| `STATIC_COLLECTION_MEMBER` | An indexed member belongs to a source-syntax fixed collection | Does **not** transfer tensor shape to aggregate |
| `STATIC_COLLECTION_ELEMENT_BINDING` | An explicitly indexed member reaches an exact local/cross-function unpack position | Cross-call requires unique callee, one supported return, matching position count |

## 4. Tested regressions (flow / operation)

| Fixture | v0.8 elements / links | v0.9 elements / links | v0.9 health |
|---|---:|---:|---|
| Linear Chain | 10 / 8 | 10 / 8 | complete |
| Nested Blocks | 24 / 16 | 24 / 16 | complete |
| Residual CNN | 24 / 20 | 27 / 23 | complete |
| U-Net Mini | 23 / 15 | 29 / 27 | complete* |
| Attention QKV | 19 / 17 | 19 / 17 | partial |
| Multi-head Output | 12 / 7 | 15 / 12 | complete* |
| Conditional Router | 11 / 14 | 11 / 14 | partial |
| Shared Layer Reuse | 10 / 8 | 10 / 8 | complete |
| Shape Conflict | 5 / 3 | 5 / 3 | partial |
| Missing Dependency | 5 / 3 | 5 / 3 | partial |
| Existing MLP | 6 / 4 | 6 / 4 | complete |
| Existing Residual | 14 / 11 | 17 / 14 | complete |
| Transformer Fixture | 27 / 25 | 27 / 25 | partial |

*Complete only for the declared supported source-level dependency view; spatial upsampling, `torch.cat` output dimensions and runtime shape constraints are not proven.*

The U-Net's new positional cross-module links are **two distinct edges**, `encoded -> position 0` and `skip -> position 1`. The Multi-head's two return positions keep distinct symbolic last dimensions `7` and `1`. `torch.chunk(3)` remains unresolved: PyTorch may return fewer chunks than requested; static syntax cannot prove the final length or the successful runtime unpack.

## 5. Negative and invariant tests

`tests/test_collection_provenance_v09.py` covers source-backed element IDs, known local unpack, repeated expression construction, cross-module returned positions, independent output shapes, unresolved `chunk`, mismatched return arity, starred/dynamic collections, unsupported concatenation shape, six view combinations, and repeatable output across two hash seeds. Existing full regression remains required.

## 6. Intentionally unsupported

- Dynamic `*` expansion and nested/nonconstant unpacking where structure cannot be proven.
- Inferred return arity from dynamic calls (`torch.chunk`, `split` with runtime sizes, etc.).
- Multi-return branches, generators, recursion, polymorphic dispatch or repeated context-sensitive calls.
- General shape semantics for concatenate, reshape, arbitrary matmul and broadcasting.
- Source-level facts do not prove the model's forward pass executes, succeeds or matches observed tensors.

## 7. Reproduce

```bash
python -m pip install -e '.[dev]'
pytest -q
magg extract tests/architecture_cases/04_unet_skip_concat --entrypoint model:UNetMini --output sae-unet
magg present sae-unet --view flow --level operation --format all --output fig-unet
magg present sae-unet --view flow --focus Encoder --format svg --output fig-encoder
```

The full static source evidence, `presentation-health.json`, `presentation-manifest.json`, and diagram exporters are preserved.
