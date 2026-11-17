"""Canonical artifact writing, with a byte-exact source snapshot."""
from __future__ import annotations

from pathlib import Path

from ..inventory.walker import SourceFile
from ..ir.canonical import canonical_bytes, sha256


def write_bundle(output: Path, artifacts: dict[str, object], sources: list[SourceFile]) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, str] = {}
    for filename in sorted(artifacts):
        content = canonical_bytes(artifacts[filename])
        (output / filename).write_bytes(content)
        result[filename] = sha256(content)
    for source in sources:
        target = output / "sources" / source.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.data)
    return result
