"""View-scoped health; not a proof that an analyzed model executes correctly."""
from __future__ import annotations

from typing import Any


def issue(code: str, message: str, *, severity: str = 'warning',
          ids: list[str] | None = None, refs: list[dict] | None = None) -> dict:
    return {'code': code, 'severity': severity, 'message': message,
            'related_ir_ids': sorted(set(ids or [])),
            'source_refs': refs or [], 'affects_view': True}


def evaluate(bundle: Any | None, graph: dict | None, *, mode: str,
             level: str, input_error: str | None = None) -> dict:
    """Classify only flaws relevant to the selected projection, not global AST warnings."""
    issues: list[dict] = []
    if input_error is not None:
        issues.append(issue('INVALID_BUNDLE', input_error, severity='error'))
    elif bundle is not None and bundle.model_index.get('status') != 'indexed':
        issues.append(issue('ENTRYPOINT_UNRESOLVED',
                            f"Model index status: {bundle.model_index.get('status')}", severity='error'))
    elif bundle is not None and (not bundle.model_index.get('root_definition_id')
                                 or not bundle.model_index.get('modules')):
        issues.append(issue('ENTRYPOINT_UNRESOLVED', 'No indexed model root', severity='error'))
    elif graph is not None:
        issues.extend(graph.get('issues', []))
        if len(graph.get('nodes', [])) <= 1:
            issues.append(issue('NO_SUPPORTED_VIEW', 'Selected view has no usable model components or operations', severity='error'))
    issues.sort(key=lambda x: (x['code'], x['message'], x['related_ir_ids']))
    status = ('blocked' if any(i['severity'] == 'error' for i in issues)
              else 'partial' if issues else 'complete')
    return {'schema_version': '1.0', 'status': status, 'mode': mode, 'level': level,
            'entrypoint': bundle.ir.get('entrypoint') if bundle else None,
            'scope': 'Source-backed static module/call architecture; not verified runtime tensor flow',
            'issues': issues, 'render_allowed': status != 'blocked'}
