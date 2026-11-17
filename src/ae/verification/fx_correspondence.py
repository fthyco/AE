"""Match static source sites to independently recorded FX events."""
from __future__ import annotations

from collections import defaultdict

OPERATIONS = frozenset({
    'ones', 'zeros', 'empty', 'rand', 'randn', 'cat', 'concat',
    'matmul', 'mm', 'bmm', 'reshape', 'view', 'flatten',
    'add', 'sub', 'mul', 'truediv',
})


def _static_operation(node: dict) -> str | None:
    attrs = node['attributes']
    if node['kind'] == 'forward_augassign':
        name = attrs.get('name', '')
        return 'add' if '+=' in name else None
    if node['kind'] == 'operation_site':
        opname = attrs.get('operator', '').lower()
        return opname if opname in OPERATIONS else None
    if node['kind'] != 'call_site':
        return None
    expression = attrs.get('callee_expression', '')
    if expression.startswith('self.'):
        return 'module'  # Only actual FX leaf calls can match.
    short = expression.split('.')[-1]
    return short if short in OPERATIONS else None


def fx_operation(event: dict) -> str | None:
    if event['op'] == 'call_module':
        return 'module'
    if event['op'] not in ('call_function', 'call_method'):
        return None
    name = event.get('target_name', '')
    return name if name in OPERATIONS else None


def _key(path: str, line: int, operation: str) -> tuple[str, int, str]:
    return path.replace('\\', '/'), line, operation


def compare_source_sites(ir: dict, events: list[dict]) -> dict:
    """Match uniquely evidenced source locations and record FX multiplicity."""
    static: dict[tuple, list[dict]] = defaultdict(list)
    for node in ir['nodes']:
        operation = _static_operation(node)
        if operation is None:
            continue
        ref = node['source_refs'][0]
        key = _key(ref['path'], ref['start_line'], operation)
        static[key].append({
            'id': node['id'], 'start_column': ref.get('start_column'),
            'end_column': ref.get('end_column'),
            'end_line': ref.get('end_line'),
        })

    observed: dict[tuple, list[dict]] = defaultdict(list)
    unknown_locations = 0
    for event in events:
        operation = fx_operation(event)
        if operation is None:
            continue
        source = event.get('source')
        if not source:
            unknown_locations += 1
            continue
        key = _key(source['path'], source['line'], operation)
        observed[(key, source.get('column'), source.get('end_column'))].append(event)

    matches, ambiguous, absent = [], [], []
    unique: dict[str, str] = {}
    for (key, column, end_column), group in sorted(observed.items(),
                                                  key=lambda pair:(pair[0][0],pair[0][1] or -1,pair[0][2] or -1)):
        candidates = static.get(key, [])
        if column is not None:
            matching = [item for item in candidates
                        if item['start_column'] == column and item['end_column'] == end_column]
            # A unique single-line source call remains matchable without a precise FX span.
            candidates = matching or (candidates if len(candidates) == 1 else [])
        ids = sorted({c['id'] for c in candidates})
        record = {'path': key[0], 'line': key[1], 'operation': key[2],
                  'column': column, 'end_column': end_column,
                  'fx_occurrences': len(group), 'ae_sites': ids,
                  'fx_names': [e['name'] for e in group]}
        if len(ids) == 1:
            matches.append(record)
            if len(group) == 1:
                unique[group[0]['name']] = ids[0]
        elif len(ids) > 1:
            ambiguous.append(record)
        else:
            absent.append(record)

    fx_edges = set()
    for event in events:
        target = unique.get(event['name'])
        if target is None:
            continue
        for source in event.get('input_nodes', []):
            if source in unique:
                fx_edges.add((unique[source], target))
    static_edges = {
        (edge['source'], edge['target'])
        for edge in ir['edges']
        if edge['relation'] == 'STATIC_VALUE_DEPENDENCY'
        and edge['source'] in unique.values() and edge['target'] in unique.values()
    }
    true_edges = static_edges & fx_edges
    false_edges = static_edges - fx_edges
    missed_edges = fx_edges - static_edges
    all_sites = len(observed)
    comparable = bool(static_edges or fx_edges)
    return {
        'scope': 'supported_fx_events_with_traceable_source_sites',
        'static_site_count': sum(len(v) for v in static.values()),
        'observed_fx_site_count': all_sites,
        'observed_fx_occurrences': sum(len(v) for v in observed.values()),
        'source_location_unknown_count': unknown_locations,
        'matched_sites': len(matches),
        'ambiguous_sites': len(ambiguous),
        'unmatched_fx_sites': len(absent),
        'observed_site_recall': len(matches) / all_sites if all_sites else None,
        'one_to_one_events': len(unique),
        'repeated_sites': [m for m in matches if m['fx_occurrences'] > 1],
        'missed_sites': absent,
        'ambiguous_site_details': ambiguous,
        'matched_site_details': matches,
        'comparable_edge_count': len(static_edges | fx_edges),
        'edge_true_positives': len(true_edges),
        'edge_false_positives': len(false_edges),
        'edge_false_negatives': len(missed_edges),
        'edge_precision': (len(true_edges)/len(static_edges)) if comparable and static_edges else None,
        'edge_recall': (len(true_edges)/len(fx_edges)) if comparable and fx_edges else None,
        'full_architecture_equivalence': 'not_verified',
        'notes': [
            'Site recall is limited to recognized FX operations with source locations.',
            'Repeated source sites are matched by location, not per-instance identity.',
            'Edge scores include only uniquely mapped one-to-one operation events.',
            'Static source operations on unexecuted paths are not false positives.',
        ],
    }
