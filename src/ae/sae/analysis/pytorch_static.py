"""Source-only PyTorch constructor semantics and symbolic layer signatures.

A signature is a type-level formula, never a claim about observed tensor sizes
or executed modules. This pass does not import PyTorch or the analyzed project.
"""
from __future__ import annotations

import ast
from typing import Any

from .analyzer import expression, value_descriptor
from .scopes import alias_is_shadowed, parent_links
from ..ir.canonical import stable_id
from ..ir.models import Graph
from ..linking.project import Unit, derived_edge, span, syntax_ids, unresolved

# Fixed, versioned recognition rule; unknown torch.nn names are not guessed.
LAYER_NAMES = {
    'Linear', 'Bilinear', 'Embedding', 'EmbeddingBag', 'Conv1d', 'Conv2d', 'Conv3d',
    'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
    'LayerNorm', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d',
    'GroupNorm', 'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d',
    'MultiheadAttention', 'Transformer', 'TransformerEncoder',
    'TransformerEncoderLayer', 'TransformerDecoder', 'TransformerDecoderLayer',
    'RNN', 'GRU', 'LSTM', 'RNNCell', 'GRUCell', 'LSTMCell',
    'Sequential', 'ModuleList', 'ModuleDict', 'ParameterList', 'ParameterDict',
    'ReLU', 'GELU', 'SiLU', 'Sigmoid', 'Tanh', 'Softmax', 'LogSoftmax', 'Dropout',
    'Dropout1d', 'Dropout2d', 'Dropout3d', 'Identity', 'Flatten', 'Unflatten',
    'MaxPool1d', 'MaxPool2d', 'MaxPool3d', 'AvgPool1d', 'AvgPool2d', 'AvgPool3d',
    'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d', 'Upsample',
}

SHAPE_PRESERVING = {'ReLU', 'GELU', 'SiLU', 'Sigmoid', 'Tanh', 'Softmax', 'LogSoftmax',
                    'Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d', 'Identity',
                    'LayerNorm', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d',
                    'GroupNorm', 'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d'}


def _symbolic(node: ast.AST | None) -> str:
    if node is None:
        return '?'
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str)):
        return str(node.value)
    return expression(node)


def _arg(call: ast.Call, index: int, key: str) -> str:
    for kw in call.keywords:
        if kw.arg == key:
            return _symbolic(kw.value)
    if index < len(call.args):
        return _symbolic(call.args[index])
    return '?'


def signature(type_: str, call: ast.Call) -> dict[str, Any]:
    """Conservative shape templates, represented symbolically, not evaluated."""
    base = type_.split('.')[-1]
    if base == 'Linear':
        a, b = _arg(call, 0, 'in_features'), _arg(call, 1, 'out_features')
        return {'input': ['...', a], 'output': ['...', b],
                'rule': 'linear-last-dimension/v1'}
    if base == 'Embedding':
        dim = _arg(call, 1, 'embedding_dim')
        return {'input': ['...'], 'output': ['...', dim],
                'rule': 'embedding-append-dimension/v1'}
    if base in SHAPE_PRESERVING:
        return {'input': ['...'], 'output': ['...'], 'rule': 'shape-preserving-module/v1'}
    if base.startswith('Conv') and not base.startswith('ConvTranspose'):
        num_dims = int(base[-2]) if base[-2:].endswith('d') and base[-2].isdigit() else None
        channels = _arg(call, 1, 'out_channels')
        if num_dims is not None:
            axes = ['L'] if num_dims == 1 else ['H', 'W'] if num_dims == 2 else ['D', 'H', 'W']
            return {'input': ['N', _arg(call, 0, 'in_channels'), *axes],
                    'output': ['N', channels, *[f'{axis}_out' for axis in axes]],
                    'spatial_dimensions': 'dependent_on_kernel_stride_padding_dilation',
                    'rule': 'conv-channel-signature/v1'}
    if base == 'MultiheadAttention':
        return {'query': ['L', 'N', _arg(call, 0, 'embed_dim')],
                'key': ['S', 'N', _arg(call, 0, 'embed_dim')],
                'value': ['S', 'N', _arg(call, 0, 'embed_dim')],
                'output': ['L', 'N', _arg(call, 0, 'embed_dim')],
                'constraints': ['batch_first=True changes tensor axis order',
                                'kdim/vdim can override key/value feature dimensions'],
                'rule': 'mha-default-shape-template/v1'}
    return {'status': 'unavailable', 'reason': 'No versioned symbolic shape rule for this layer'}


def _imports(unit: Unit) -> dict[str, str]:
    imports: dict[str, str] = {}
    rebound: set[str] = set()
    for stmt in unit.tree.body:
        if isinstance(stmt, ast.Import):
            for a in stmt.names:
                imports[a.asname or a.name.split('.')[0]] = a.name if a.asname else a.name.split('.')[0]
        elif isinstance(stmt, ast.ImportFrom) and stmt.level == 0:
            for a in stmt.names:
                if a.name != '*':
                    imports[a.asname or a.name] = f'{stmt.module}.{a.name}'
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for target in targets:
                rebound.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
    return {k: v for k, v in imports.items() if k not in rebound}


def _qualified(node: ast.AST, imports: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return imports.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return f'{_qualified(node.value, imports)}.{node.attr}'
    return expression(node)


def analyze_pytorch(graph: Graph, units: list[Unit]) -> None:
    for unit in sorted(units, key=lambda u: u.path):
        ids = syntax_ids(unit)
        imports = _imports(unit)
        parents = parent_links(unit.tree)
        recognized: dict[ast.Call, str] = {}
        for node in ast.walk(unit.tree):
            if not isinstance(node, ast.Call):
                continue
            base_node = node.func
            while isinstance(base_node, ast.Attribute):
                base_node = base_node.value
            if isinstance(base_node, ast.Name) and alias_is_shadowed(node, base_node.id, parents):
                continue
            qual = _qualified(node.func, imports)
            if not (qual.startswith('torch.nn.') and len(qual.split('.')) == 3):
                continue
            base = qual.split('.')[-1]
            if base not in LAYER_NAMES:
                unresolved(graph, unit, node, 'unknown_torch_nn_construct',
                           f'Unrecognized torch.nn callable: {qual}')
                continue
            ident = stable_id(unit.path, 'pytorch-layer-spec', ids[node])
            args = {'positional': [value_descriptor(a) for a in node.args],
                    'keywords': [{'name': k.arg, 'value': value_descriptor(k.value)} for k in node.keywords]}
            graph.add_node({'id': ident, 'kind': 'pytorch_layer_spec',
                            'attributes': {'declared_type': qual, 'arguments': args, 'shape_template': signature(qual, node)},
                            'source_refs': [span(unit, node)], 'evidence_class': 'derived',
                            'resolution': 'conditional', 'analysis_rule': 'pytorch-constructor-syntax/v1'})
            call = stable_id(unit.path, 'semantic', 'call_site', ids[node], '')
            derived_edge(graph, unit, node, ident, call, 'DESCRIBES_PYTORCH_CALL',
                         'pytorch-constructor-syntax/v1')
            recognized[node] = ident
        for parent, parent_id in recognized.items():
            for nested in ast.walk(parent):
                if nested is parent:
                    continue
                if isinstance(nested, ast.Call) and nested in recognized:
                    derived_edge(graph, unit, nested, parent_id, recognized[nested],
                                 'CONTAINS_NESTED_CONSTRUCTOR', 'nested-pytorch-constructor/v1')
