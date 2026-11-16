# SAE v0.3.0 MVP — acceptance and Presenter handoff

Goal: deterministic entrypoint-focused architecture index from source-backed Python/PyTorch facts without executing user code.

## Outputs

- architecture.ir.json remains the complete source-grounded graph, with unmodified purpose and depth.
- model-index.json adds source-backed selections: modules, constructors, linked local submodules, forward-call sites, operation sites and references to IR dependency relation IDs.
- Ambiguous/unsupported component assignments are explicitly unresolved, never guessed.
- YAML, JSON and TOML are parsed as typed configuration trees; no model code, interpolation or YAML Python-object tags are executed.
- validate_bundle checks index references and artifact checksums.

## Acceptance matrix

| Fixture | Evidence |
| --- | --- |
| MLP | Two Linear constructors, module references, forward calls, YAML metadata |
| CNN + ResidualBlock | Unique nested component, Conv2d, residual Add |
| Transformer + Attention | Cross-file modules, nested Sequential, matmul/softmax |
| Unsafe YAML | Reject object tags, duplicate keys, multiple documents |
| Unknown factory | Explicit unresolved, never fabricate a module |
| Repeatability | Byte-identical IR and model index under different hash seeds |

Run: python -m pip install -e '.[dev]' && python -m pytest -q

## Presenter boundary

Every rendered block or arrow must retain supporting IR IDs. The model index exposes lexical source order, not execution order. Forward relation IDs are source-level, not proof of runtime tensor-flow.

## Deferred

Advanced alias/parameter sharing, interprocedural data flow, full SSA, deep symbolic tensor propagation, Hydra/OmegaConf semantic resolution and runtime-dependent paths. Scientific diagram generation remains a separate Presenter.
