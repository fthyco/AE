# Composite tensor shapes — v0.13

The extractor adds conditional type-level facts for source-backed PyTorch operations. The target model is never executed.

| Operation | Supported evidence |
| --- | --- |
| `torch.ones`, `zeros`, `empty`, `rand`, `randn` | Positive integer literal shape arguments |
| `torch.matmul` and `@` | Two known numeric tensor shapes, including batched broadcasting |
| `torch.cat` | Fixed list/tuple, numeric dimensions, literal axis |
| `torch.cat` with two or more `Linear` results | Same proven input value, literal `dim=-1`, fixed output widths |
| `torch.reshape`, `Tensor.reshape` / `Tensor.view` | Exact known element count, literal target dimensions, at most one `-1` |

Unknown symbols, incompatible broadcasting, dynamic collection expansion, dynamic axes, and unconstrained tensor dimensions remain unresolved. Shape facts are conditional source-based constraints, not runtime observations.

Verification: `python scripts/verify_composite_shapes.py`. Compare the derived operation outputs with PyTorch FX ShapeProp and runtime tensors. Real TorchVision ResNet18 and DenseNet121 are checked separately for unchanged upstream sources and `Partial` status.
