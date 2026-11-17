# Instance-aware FX verification — v0.15

`ae.verification.instance_correspondence.compare_factory_instances` matches source-backed factory Block instances to independently traced FX events.

## Inputs

- `model-index.json`: factory stage names, Block types, ordered instance IR IDs
- `architecture.ir.json`: operation sites, source spans, static value dependencies
- FX event trace: operation, original file/line/column, `nn_module_stack`, input-node identities, optional runtime shape
- Optional independent `named_modules()` type map

## Matching

1. Verify factory Block instances and ordered module paths (`layer1.0`, etc.).
2. Associate each FX operation with its owning Block using the traced module stack and concrete module type.
3. Match each event to an original source IR operation site and verify file/line/column.
4. Require unique `(Block instance, source operation site)` matches; report multiplicity and missing sites.
5. Compare direct `STATIC_VALUE_DEPENDENCY` edges only where both FX endpoints are matched within the same Block.
6. Compare concrete or suffix shapes only where source facts exist.

## TorchVision 0.25.0 (CPU) results

| Model | Verified Block identities | Matched FX block operations | Direct intrablock edges (matched/FX) | Out of scope |
| --- | ---: | ---: | ---: | --- |
| ResNet18 | 8/8 | 56/62 | 48/48 | 6 composite downsample leaf events |
| ResNet50 | 16/16 | 160/168 | 144/144 | 8 composite downsample leaf events |

`matched FX block operations` is scoped to recognized operations under the Blocks, not all operations in FX or the model. FX leaf operations inside one `self.downsample(x)` call are not invented as distinct AE source sites. Cross-Block connections, general tensor shapes, graph paths, and full runtime equivalence remain unverified.

Original `torchvision/models/resnet.py` SHA-256: `9faf2f321fdaf7edbd791aedc6e3610c7edf9c892112cbd75861beae1e9518d1`.

## Run

```bash
python scripts/verify_fx_instances.py --output benchmark-output/fx-instances
```

Outputs include `summary.json`, per-model `correspondence.json`, `fx_events.json`, and source-backed SAE bundles. The static extractor never imports or executes the target models. PyTorch execution occurs only in the independent verifier.
