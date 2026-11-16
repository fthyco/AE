# Real-repository static architecture acceptance — MAGG v0.5

## Verified sources (GitHub Actions)

Workflow: Real-model static acceptance (.github/workflows/real-models-smoke.yml).

- fthyco/maia3, branch CF-ACT, entrypoint maia3.models_cf_act:MAIA3ActModel
- fthyco/Adaptive-Width-Neural-Networks-WITH-ORDERING-GUIDENCE, branch V(1-3)BT, entrypoint resnet:BasicBlock

The workflow checks out the original repositories and selects the relevant source packages/files, then runs SAE and Presenter without importing or executing the target model. Full model runtime equivalence is not claimed.

Successful run: https://github.com/fthyco/AE/actions/runs/43966962283

| Selected source | IR nodes | IR edges | Visible static-value links | Health |
| --- | ---: | ---: | ---: | --- |
| Chessformer CF-ACT model + local package | 14,639 | 19,850 | 55 | partial (11 issues) |
| AWN resnet.BasicBlock | 318 | 458 | 6 | partial (4 issues) |

All rendered arrows and blocks were checked against real IR IDs and source evidence. SVG/PDF/TikZ, health reports and mapping manifests are available as workflow artifacts.

## Limits observed in real code

- Chessformer contains For, If, attribute parameters, destructuring and nested method calls. These are reported as unresolved rather than fictional tensor flow.
- AWN BasicBlock has conditional downsample and in-place out += identity. The source-backed view remains partial and must not claim verified residual flow.
- Lexical call order is not runtime data flow. These outputs are research-oriented static value-dependency diagrams, not complete runtime graphs.

## Verified acceptance

- Full local regression: 83 passed on Python 3.13.
- Published GitHub CI: Python 3.11 and Python 3.13 passed on 86873a4 and 0c2fd09.
- Real-repository smoke: passed while correctly classifying both views as partial.

## Further work for complete paper-ready real-model diagrams

1. Conservative branch merge and augmented assignment handling.
2. Attribute parameter references and nested call resolution.
3. Tuple/collection element provenance.
4. Better pagination, grouping and sizing for complex architecture figures.
5. Model-by-model review against original source and dimension contracts.

These are future improvements, not features claimed complete in v0.5.


## v0.6 — guarded forward alternatives and module focus (verified)

[Successful real-repository smoke run](https://github.com/fthyco/AE/actions/runs/43971554892) at commit `a4bb564` checked out the original Chessformer CF-ACT package and AWN `resnet.py` (not an entire training execution). It passed provenance assertions for every shown node and link, plus explicit AWN conditional-merge and AugAssign assertions.

| Original selected source | IR nodes | IR edges | Visible source-backed links | Status |
| --- | ---: | ---: | ---: | --- |
| Chessformer CF-ACT + maia3 package | 14,661 | 19,950 | 98 | partial (30 issues) |
| AWN BasicBlock | 321 | 469 | 14 | partial (3 issues) |

Changes relative to v0.5 include conditional `forward_choice`, guarded call relationships and `forward_augassign` operand dependencies. The AWN diagram now represents both alternatives to `identity` and the ensuing `out += identity` result dependency, **without** asserting which downsample path executes or whether in-place alias effects are safe. `--view flow --focus <module>` exports one module in its own figure with source mappings; invalid or ambiguous focus blocks rendering.

Status is intentionally `partial` for both real models. Remaining limitations include complex interprocedural paths, some dynamic forms of control flow, unverified tuple arity, and scalable publication layout. A successful workflow is **not** a claim of complete, execution-equivalent paper architecture.
