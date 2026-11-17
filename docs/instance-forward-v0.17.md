# AE v0.17 — ResNet50 Bottleneck expansion

ResNet50 extends the bounded source-only instance graph to the original TorchVision `Bottleneck` constructor, including Conv1x1, strided Conv3x3, Conv1x1, three BatchNorm layers, three ReLU calls per block, residual Add, and optional two-leaf projection.

```bash
ae extract <source-dir> --entrypoint resnet:resnet50 --input-shape 1,3,64,64 --output bundle
ae present bundle --view instance --level operation --format all --output figures
python scripts/verify_full_resnet18.py --model resnet50 --output benchmark-output/full-resnet50
```

## Independent gate

Default factory with unmodified TorchVision 0.25.0+cpu, PyTorch 2.10.0+cpu, explicit inputs `1×3×64×64`, `1×3×96×96`, and `1×3×224×224`.

- 16/16 Bottleneck instances and 4 projection branches
- **177/177** FX operation/module identities
- **192/192** operand dependencies
- **177/177** tensor shapes (FX ShapeProp)
- **158/158** Runtime Hook leaf invocations
- **126** distinct leaf module identities
- Determinism across hash seeds
- Unsupported changes to width, projections or forward steps fail closed

The original `resnet.py` SHA-256 is `9faf2f321fdaf7edbd791aedc6e3610c7edf9c892112cbd75861beae1e9518d1`. No target code runs during AE extraction. Independent tests execute the original model only in the verifier. Other models and dynamic kwargs are not claimed complete.
