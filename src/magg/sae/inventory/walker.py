"""Filesystem inventory: no imports or execution of target files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from ..ir.canonical import sha256
from ..settings import Settings

# Default scope is source and common text configuration, not media/datasets.
IN_SCOPE = {".py", ".json", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".txt"}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache", ".tox", "build", "dist"}


@dataclass(frozen=True)
class SourceFile:
    path: str
    data: bytes
    sha256: str


def inventory(settings: Settings) -> tuple[list[SourceFile], list[dict]]:
    root = settings.source_root.resolve()
    if not root.is_dir():
        raise ValueError(f"Source root is not a directory: {root}")
    output = settings.output.resolve()
    sources: list[SourceFile] = []
    manifest: list[dict] = []
    seen = 0
    for current, dirs, files in os.walk(root, followlinks=False):
        base = Path(current)
        accepted: list[str] = []
        for dirname in sorted(dirs):
            path = base / dirname
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                manifest.append({"path": rel, "status": "excluded", "reason": "symlink_directory"})
            elif dirname in SKIP_DIRS or path == output or output in path.parents and path == output:
                manifest.append({"path": rel, "status": "excluded", "reason": "excluded_directory"})
            else:
                accepted.append(dirname)
        dirs[:] = accepted
        for filename in sorted(files):
            full = base / filename
            rel = full.relative_to(root).as_posix()
            if full.is_symlink():
                manifest.append({"path": rel, "status": "excluded", "reason": "symlink_file"})
                continue
            if output == full or output in full.parents:
                manifest.append({"path": rel, "status": "excluded", "reason": "output_directory"})
                continue
            if full.suffix.lower() not in IN_SCOPE:
                manifest.append({"path": rel, "status": "excluded", "reason": "unsupported_extension"})
                continue
            if seen >= settings.max_files:
                manifest.append({"path": rel, "status": "excluded", "reason": "file_count_limit"})
                continue
            try:
                size = full.stat().st_size
                if size > settings.max_file_bytes:
                    manifest.append({"path": rel, "status": "excluded", "reason": "file_size_limit", "size": size})
                    continue
                raw = full.read_bytes()
            except OSError as exc:
                manifest.append({"path": rel, "status": "unreadable", "reason": type(exc).__name__})
                continue
            if len(raw) > settings.max_file_bytes:
                manifest.append({"path": rel, "status": "excluded", "reason": "file_size_limit", "size": len(raw)})
                continue
            hash_value = sha256(raw)
            sources.append(SourceFile(rel, raw, hash_value))
            manifest.append({"path": rel, "status": "included", "sha256": hash_value, "size": len(raw), "kind": "python" if full.suffix.lower() == ".py" else "config_or_text"})
            seen += 1
    manifest.sort(key=lambda r: r["path"])
    sources.sort(key=lambda f: f.path)
    return sources, manifest
