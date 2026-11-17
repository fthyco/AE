# AE v0.22 — MobileNetV3 Small (Stage 6, work in progress)

Reference: unchanged TorchVision 0.25.0 `mobilenet_v3_small(weights=None)` with Torch 2.10.0, evaluated using independent `torch.fx`, ShapeProp and runtime hooks.

Local original-model preflight for 64×64, 128×128 and 224×224 RGB inputs measured **159 FX operations, 173 ordered operand dependencies, 141 leaf-module calls**, including **11 InvertedResidual blocks, 9 Squeeze-and-Excitation gates, and 6 residual additions**.

This reference count is **not evidence that Architecture Extractor already handles the model**. The pending work is a source-only SHA-256-pinned static expansion, a full operator-by-operator and edge-by-edge independent comparison, fail-closed source mutation tests, render manifest parity, reproducibility and CI. Model import/execution must remain confined to independent verifier code.

Keep `main` on v0.21 until dedicated and repository-wide checks pass.
