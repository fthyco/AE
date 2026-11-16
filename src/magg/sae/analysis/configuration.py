"""Static configuration tree ingestion for JSON/TOML, without evaluation."""
from __future__ import annotations

import json
import math
import tomllib
from typing import Any

from ..inventory.walker import SourceFile
from ..ir.canonical import stable_id
from ..ir.models import Graph


def _file_span(file: SourceFile) -> dict:
    return {'path': file.path, 'start_byte': 0, 'end_byte': len(file.data),
            'start_line': 1, 'start_column': 0, 'end_line': file.data.count(b'\n') + 1,
            'end_column': 0, 'precision': 'file'}


def analyze_configuration(graph: Graph, files: list[SourceFile]) -> None:
    for file in files:
        if not file.path.lower().endswith(('.json', '.toml')):
            continue
        try:
            if file.path.lower().endswith('.json'):
                data = json.loads(file.data, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f'Non-finite JSON value: {x}')))
            else:
                data = tomllib.loads(file.data.decode('utf-8-sig'))
        except (ValueError, UnicodeError, tomllib.TOMLDecodeError) as exc:
            graph.issue(kind='config_parse_failure', path=file.path, message=str(exc))
            continue
        ref = _file_span(file)
        root = stable_id(file.path, 'source-file')
        def traverse(value: Any, pointer: str, parent: str, depth: int = 0) -> None:
            if depth > 100:
                graph.issue(kind='config_depth_limit', path=file.path,
                            message=f'Nested config depth limit at {pointer}')
                return
            node = stable_id(file.path, 'configuration', pointer)
            container = isinstance(value, (dict, list))
            attrs = {'pointer': pointer, 'value_kind': type(value).__name__}
            if not container:
                if isinstance(value, float) and not math.isfinite(value):
                    graph.issue(kind='nonfinite_config_number', path=file.path,
                                message=f'Non-finite numeric value at {pointer}')
                    attrs['value_kind'] = 'nonfinite_float'
                    attrs['value'] = str(value)
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    attrs['value'] = value
                else:
                    attrs['value'] = str(value)
            graph.add_node({'id': node, 'kind': 'configuration_entry', 'attributes': attrs,
                            'source_refs': [ref], 'evidence_class': 'derived',
                            'resolution': 'resolved', 'analysis_rule': 'json-toml-parse/v1'})
            graph.add_edge({'id': stable_id(file.path, 'config-edge', parent, node),
                            'source': parent, 'target': node, 'relation': 'HAS_CONFIGURATION_ENTRY',
                            'evidence_class': 'derived', 'analysis_rule': 'json-toml-parse/v1',
                            'source_refs': [ref]})
            if isinstance(value, dict):
                for key in sorted(value):
                    encoded = str(key).replace('~', '~0').replace('/', '~1')
                    traverse(value[key], pointer + '/' + encoded, node, depth + 1)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    traverse(child, pointer + '/' + str(index), node, depth + 1)
        traverse(data, '', root)
