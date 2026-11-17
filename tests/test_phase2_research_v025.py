"""Unit tests for independent non-executing research-source oracle."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmark_external_research import CASES, source_oracle


def test_source_oracle_preserves_original_bytes_and_methods(tmp_path):
    target = tmp_path / "maia3" / "models_cf_act.py"
    target.parent.mkdir(parents=True)
    code = (
        "class MAIA3ActModel:\n"
        "    def __init__(self, cfg):\n"
        "        self.cfg = cfg\n"
        "    def forward(self, x):\n"
        "        if x:\n"
        "            return x\n"
        "        return None\n"
    )
    target.write_text(code)
    spec = CASES["chessformer-act"]
    result = source_oracle(tmp_path, spec)
    assert result["target_class"] == "MAIA3ActModel"
    assert result["target_methods"] == ["__init__", "forward"]
    assert result["has_conditional_construction"]
    assert result["file_sha256"]
    assert target.read_text() == code


def test_source_oracle_fails_closed_on_missing_forward(tmp_path):
    target = tmp_path / "model.py"
    target.write_text("class AWN:\n    def __init__(self): pass\n")
    with pytest.raises(AssertionError, match="interface changed"):
        source_oracle(tmp_path, CASES["awn"])


def test_research_references_are_pinned():
    for name, data in CASES.items():
        assert len(data["commit"]) == 40
        assert all(c in "0123456789abcdef" for c in data["commit"])
        assert data["file"].endswith(".py")
        assert ":" in data["entrypoint"]
