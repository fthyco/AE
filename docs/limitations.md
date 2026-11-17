# Architecture Extractor — Limitations

AE analyzes source code without importing or executing the target model.

**Supported:** Python AST/tokens, local module links, selected PyTorch layer signatures, intra-function value dependencies, uniquely resolved call bindings, conditional branches, fixed-arity tuple/list provenance, YAML/JSON/TOML, and conditional symbolic shapes.

**Limited or unsupported:** dynamic imports/dispatch, parameter aliasing/sharing, recursion, multi-use call sensitivity, dynamic unpacking, arbitrary runtime paths, general broadcasting, MatMul/Concat/Reshape shape rules, complete Conv spatial dimensions, Hydra/OmegaConf runtime evaluation, and observed tensor flow.

`torch.chunk(n)` does not guarantee exactly `n` elements. Unknown relationships and dimensions remain unresolved.

`complete` applies only to the selected supported static view. `partial` indicates relevant unknowns. `blocked` indicates invalid or insufficient evidence.
