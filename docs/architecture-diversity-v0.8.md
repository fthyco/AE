# MAGG v0.8 architecture diversity acceptance

This is a **static source-evidence** benchmark. No target model was imported or executed.

## Scope and artifacts

- 14 architecture scenarios examined locally: ten new regression fixture types, three existing MLP/CNN/Transformer fixtures, plus original AWN BasicBlock tested separately from its source snapshot.
- Four explicit-depth stress cases: 2, 4, 8 and 16 Linear layers.
- 108 view combinations in both `structure` and `flow`, at `overview`, `block` and `operation`, plus 26 focused module figures.
- 157 manifests inspected (includes repeated render/publication checks); all 1,801 referenced visual-node occurrences and 1,773 link occurrences had real IR IDs and source refs.
- Six of fourteen architecture cases were `complete` in the bounded static flow operation view; eight were `partial`. No extraction or renderer exceptions were seen in these fourteen cases.
- Invalid root syntax blocked architecture rendering.
- Three representative cases reproduced identical IR, model-index and SVG bytes across distinct PYTHONHASHSEED values.
- 18 model geometry scenes (including depth stress) exhibited zero nonorthogonal segments and no observed paths intersecting unrelated member boxes.

## Detail behavior

| Family | Representative behavior | Unresolved |
| --- | --- | --- |
| Deep MLP / nested custom blocks | Supported Linear shape labels and model-to-block references | No general runtime tensor tracing |
| Residual CNN | Declared skip branches and Add operations represented | Spatial tensor dimensions conditional |
| U-Net | Encoder, Decoder, interpolation and concat detected | Tuple-return/element-level provenance |
| Attention | QKV, chunk, transpose, matmul, softmax visible | QKV element identities and matmul shapes |
| Multiple outputs | Both heads displayed | Tuple return fan-in to Output |
| Conditional router | Both branch alternatives displayed | Executed branch never assumed |
| Shared/reused layer | Two use sites; one declared layer | General alias semantics |
| Shape mismatch | Linear expects 20, prior template yields 12 | Diagnostic, not runtime proof |
| Missing Encoder | Supported head plus unresolved reference | Missing external implementation |
| Original AWN BasicBlock | Downsample, inplace +=, branch join | Aliasing and dynamic conditions |

On a 16-layer explicit chain the flow scene became about 4,527 pixels wide. The remaining visual bottleneck is paginated/compact panels, not a fabricated tensor path.

## Committed CI regression

`tests/test_architecture_diversity.py` covers ten source fixtures in `tests/architecture_cases`, the three existing model examples, all six combinations of view and detail level, 2/4/8/16 depth scaling, source provenance, shape conflict, unresolved dependency, and syntax blocking.

Run: `python -m pytest -q tests/test_architecture_diversity.py`.

The measured standalone dataset, full exact sources, vector diagrams and source-backed interactive inspector are provided as a separate downloadable benchmark bundle. Regression fixtures in this repository may be normalized variants of the archived measured sources; their own CI results must be checked independently.

**Important:** `complete` means supported source-backed view completeness only, not executable or fully reconstructed PyTorch model dataflow.
