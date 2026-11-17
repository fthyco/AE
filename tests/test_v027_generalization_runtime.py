"""v0.27: generic, evidence-only source hierarchy and optional runtime oracle."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ae.sae.api import extract
from ae.presenter.semantic import source_category
from ae.presenter.web_viewer import build_viewer
from ae.verification.runtime_reference import runtime_audit


SAMPLES = Path(__file__).parents[1] / "examples"


def test_generic_semantic_taxonomy_does_not_guess_custom_types():
    assert source_category("component", "self.proj: nn.Linear") == "linear"
    assert source_category("component", "self.norm: nn.LayerNorm") == "normalization"
    assert source_category("component", "self.myblock: MyMagicBlock") == "unknown"
    assert source_category("operation", "Add") == "elementwise"
    assert source_category("operation", "foo(x)") == "unknown"
    assert source_category("model", "nn.Linear") == "unknown"


def test_auto_fallback_is_evidence_backed_and_three_levels_differ(tmp_path):
    bundle = tmp_path / "bundle"
    extract(SAMPLES / "cnn_residual_mvp", bundle, entrypoint="model:CNN")
    result = build_viewer(bundle, tmp_path / "viewer", view="auto")
    assert result["graph_type"] == "structure"
    assert result["source_execution"] is False
    assert result["rendered_levels"] == ["overview", "block", "operation"]
    records = []
    for name in result["rendered_levels"]:
        data = json.loads((tmp_path / "viewer" / name /
                           "presentation-manifest.json").read_text())
        assert all(x["ir_node_ids"] and x["source_refs"]
                   for x in data["visual_elements"].values())
        records.append(len(data["visual_elements"]))
    assert records[0] < records[1] < records[2]
    category = [n.get("semantic_category")
                for n in data["visual_elements"].values()]
    assert "linear" in category
    html = (tmp_path / "viewer" / "index.html").read_text()
    assert 'id="search"' in html
    assert "function searchGraph()" in html
    assert "matching nodes" in html
    assert "data-operation-category" in html


def test_refuse_execution_without_explicit_opt_in(tmp_path):
    source = tmp_path / "sample"
    source.mkdir()
    (source / "trusted.py").write_text(
        "from pathlib import Path\nPath('EXECUTED').write_text('unsafe')\n")
    with pytest.raises(ValueError, match="--allow-execute"):
        runtime_audit(source_root=source, factory="trusted:build",
                      input_shapes=[(1, 3, 8, 8)],
                      output=tmp_path / "reference", allow_execute=False)
    assert not Path("EXECUTED").exists()
    assert not (tmp_path / "reference").exists()


def test_runtime_reference_on_trusted_residual_model(tmp_path):
    pytest.importorskip("torch")
    source = tmp_path / "trusted"
    source.mkdir()
    (source / "toy_ref_g027.py").write_text(
        "import torch\n"
        "from torch import nn\n"
        "class Model(nn.Module):\n"
        " def __init__(self):\n"
        "  super().__init__()\n"
        "  self.left=nn.Linear(4,4)\n"
        "  self.right=nn.Linear(4,2)\n"
        " def forward(self,x):\n"
        "  return self.right(self.left(x)+x)\n"
        "def build():\n"
        " return Model()\n"
    )
    result = runtime_audit(source_root=source, factory="toy_ref_g027:build",
                           input_shapes=[(2,4)], output=tmp_path / "reference",
                           allow_execute=True)
    assert result["status"] == "traced", result.get("reason")
    assert result["fx_nodes"] >= 5
    assert result["fx_operand_edges"] >= 4
    assert result["runtime_leaf_invocations"] == 2
    assert result["ae_static_equivalence"] == "not_verified"
    assert result["dynamic_control_flow_completeness"] == "not_verified"
    data = json.loads((tmp_path / "reference" / "fx-runtime-reference.json").read_text())
    assert data["output"]["shape"] == [2, 2]
    assert all("operand_ids" in n for n in data["nodes"])


def test_dynamic_tensor_branch_is_explicitly_unsupported(tmp_path):
    pytest.importorskip("torch")
    source=tmp_path / "dynamic"
    source.mkdir()
    (source / "trace_branch_g027.py").write_text(
        "from torch import nn\n"
        "class Model(nn.Module):\n"
        " def forward(self,x):\n"
        "  if x.sum() > 0:\n"
        "   return x * 2\n"
        "  return x\n"
        "def build(): return Model()\n"
    )
    result = runtime_audit(source_root=source, factory="trace_branch_g027:build",
                           input_shapes=[(1,4)], output=tmp_path / "out",
                           allow_execute=True)
    assert result["status"] == "unsupported"
    assert "TraceError" in result["reason"]
    assert not (tmp_path / "out" / "fx-runtime-reference.json").exists()
