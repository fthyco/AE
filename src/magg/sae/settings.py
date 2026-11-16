"""Deliberately narrow, static-only source inventory configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    source_root: Path
    output: Path
    entrypoint: str | None = None
    max_files: int = 2000
    max_file_bytes: int = 2_000_000

    def __post_init__(self) -> None:
        if self.max_files < 1 or self.max_file_bytes < 1:
            raise ValueError("File limits must be positive")
