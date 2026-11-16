"""Read a validated SAE bundle as data; never import the analyzed model."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from magg.sae.api import validate_bundle
from magg.sae.indexing.model_index import validate_model_index


@dataclass(frozen=True)
class Bundle:
    directory: Path
    ir: dict[str, Any]
    model_index: dict[str, Any]
    manifest: dict[str, Any]
    coverage: dict[str, Any]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]
    nodes: dict[str, dict[str, Any]]
    edges: dict[str, dict[str, Any]]


def load_bundle(directory: str | Path) -> Bundle:
    """Raise ValueError if integrity, schema references, or source bytes fail verification."""
    path = Path(directory).resolve()
    if path.is_file():
        path = path.parent
    # Verify original artifacts before trusting the derived model index.
    validate_bundle(path)
    def read(name: str) -> dict[str, Any]:
        return json.loads((path / name).read_bytes())
    ir, index = read('architecture.ir.json'), read('model-index.json')
    validate_model_index(index, ir)
    return Bundle(path, ir, index, read('source-manifest.json'),
                  read('coverage.json'), read('diagnostics.json'),
                  read('extraction-meta.json'),
                  {n['id']: n for n in ir['nodes']},
                  {e['id']: e for e in ir['edges']})
