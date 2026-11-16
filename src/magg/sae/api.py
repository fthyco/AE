"""Public static extraction API. Does not load or run analyzed projects."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from magg import __version__
from .analysis.analyzer import Analyzer
from .inventory.walker import inventory
from .ir.canonical import sha256, canonical_bytes, stable_id
from .ir.models import Graph
from .ir.validator import validate_ir
from .output.bundle import write_bundle
from .parsing.python_parser import parse_python
from .settings import Settings


def _source_span(path: str, size: int) -> dict:
    return {"path": path, "start_byte": 0, "end_byte": size,
            "start_line": 1, "start_column": 0, "end_line": 1, "end_column": 0,
            "precision": "file"}


def extract(source_root: str | Path, output: str | Path, *, entrypoint: str | None = None,
            max_files: int = 2000, max_file_bytes: int = 2_000_000) -> dict[str, Any]:
    """Extract reproducible source-grounded facts into a self-contained IR bundle.

    Source directories are never imported or executed. Only local files are read.
    Returns the same metadata written to ``extraction-meta.json``.
    """
    settings = Settings(Path(source_root), Path(output), entrypoint, max_files, max_file_bytes)
    sources, manifest = inventory(settings)
    graph = Graph()
    source_index: dict[str, Any] = {"schema_version": "1.0", "files": []}
    parsed = 0
    python_total = 0
    for file in sources:
        source_id = stable_id(file.path, "source-file")
        graph.add_node({"id": source_id, "kind": "source_file",
                 "attributes": {"path": file.path, "sha256": file.sha256, "size": len(file.data)},
                 "source_refs": [_source_span(file.path, len(file.data))],
                 "evidence_class": "explicit", "resolution": "resolved"})
        entry: dict[str, Any] = {"path": file.path, "source_id": source_id, "sha256": file.sha256}
        if file.path.lower().endswith(".py"):
            python_total += 1
            source, tree, tokens, error = parse_python(file.path, file.data)
            entry["tokens"] = tokens
            entry["encoding"] = source.encoding if source else None
            entry["parsed"] = tree is not None
            if error is not None:
                graph.diagnostics.append(error)
            if source is not None and tree is not None:
                try:
                    Analyzer(graph, source, tree, source_id).run()
                    parsed += 1
                except (RecursionError, ValueError) as exc:
                    # A parser/analysis failure must not erase original source.
                    graph.issue(kind="analysis_failure", path=file.path, message=str(exc))
            source_index["files"].append(entry)
        else:
            # Config byte preservation is implemented; configuration semantics
            # are not evaluated by this version of SAE.
            entry["tokens"] = []
            entry["parsed"] = False
            entry["encoding"] = None
            source_index["files"].append(entry)
    graph.nodes.sort(key=lambda n: n["id"])
    graph.edges.sort(key=lambda e: e["id"])
    graph.unresolved.sort(key=lambda u: (u["path"], u.get("site_id", ""), u["kind"], u.get("expression", "")))
    graph.diagnostics.sort(key=lambda d: (d["path"], d["kind"], d["message"]))
    source_index["files"].sort(key=lambda f: f["path"])
    ir = {"schema_version": "1.0", "extractor_version": __version__,
          "entrypoint": entrypoint, "nodes": graph.nodes,
          "edges": graph.edges, "unresolved": graph.unresolved}
    check = validate_ir(ir, {f.path: len(f.data) for f in sources})
    coverage = {"schema_version": "1.0", "included_files": len(sources),
                "python_files": python_total, "parsed_python_files": parsed,
                "unparsed_python_files": python_total - parsed,
                "semantic_resolution": {"unresolved": len(graph.unresolved)},
                "notes": ["Runtime behavior is not analyzed", "No complete dynamic-path coverage is claimed",
                          "Configuration source bytes are preserved, not evaluated",
                          "Static recognition is limited to known PyTorch import patterns"]}
    artifacts = {"architecture.ir.json": ir,
                 "source-index.json": source_index,
                 "source-manifest.json": {"schema_version": "1.0", "files": manifest},
                 "coverage.json": coverage,
                 "diagnostics.json": {"schema_version": "1.0", "diagnostics": graph.diagnostics}}
    hashes = {k: sha256(canonical_bytes(v)) for k, v in artifacts.items()}
    metadata = {"schema_version": "1.0", "extractor_version": __version__,
                "entrypoint": entrypoint, "verification": check,
                "artifact_sha256": hashes}
    artifacts["extraction-meta.json"] = metadata
    write_bundle(settings.output, artifacts, sources)
    return metadata


def validate_bundle(path: str | Path) -> dict:
    folder = Path(path)
    if folder.is_file():
        folder = folder.parent
    ir = json.loads((folder / "architecture.ir.json").read_bytes())
    manifest = json.loads((folder / "source-manifest.json").read_bytes())
    meta = json.loads((folder / "extraction-meta.json").read_bytes())
    sources = {}
    issues = []
    for record in manifest["files"]:
        if record["status"] == "included":
            target = folder / "sources" / record["path"]
            if not target.is_file():
                issues.append(f"Missing source: {record['path']}")
                continue
            raw = target.read_bytes()
            sources[record["path"]] = len(raw)
            if sha256(raw) != record["sha256"]:
                issues.append(f"Source checksum mismatch: {record['path']}")
    if issues:
        raise ValueError("; ".join(issues))
    verification = validate_ir(ir, sources)
    for filename, expected in meta["artifact_sha256"].items():
        if sha256((folder / filename).read_bytes()) != expected:
            raise ValueError(f"Artifact checksum mismatch: {filename}")
    return {"valid": True, "verification": verification, "sources_checked": len(sources)}


def inspect_bundle(path: str | Path) -> dict:
    folder = Path(path)
    if folder.is_file():
        folder = folder.parent
    ir = json.loads((folder / "architecture.ir.json").read_bytes())
    coverage = json.loads((folder / "coverage.json").read_bytes())
    kind_counts: dict[str, int] = {}
    for node in ir["nodes"]:
        kind = node["kind"]
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    return {"entrypoint": ir.get("entrypoint"), "entities_by_kind": dict(sorted(kind_counts.items())),
            "relations": len(ir["edges"]), "unresolved": len(ir["unresolved"]), "coverage": coverage}
