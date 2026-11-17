"""Offline viewer acceptance: provenance, fail-closed exports and CLI."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from ae.sae.api import extract
from ae.presenter.web_viewer import build_viewer


EXAMPLES = Path(__file__).parents[1] / "examples"


def make_bundle(tmp_path):
    output = tmp_path / "bundle"
    extract(EXAMPLES / "cnn_residual_mvp", output, entrypoint="model:CNN")
    return output


def test_viewer_renders_three_evidence_backed_levels_offline(tmp_path):
    bundle = make_bundle(tmp_path)
    result = build_viewer(bundle, tmp_path / "viewer", view="structure", title="AE < Demo")
    assert result["source_execution"] is False
    assert result["rendered_levels"] == ["overview", "block", "operation"]
    html = (tmp_path / "viewer/index.html").read_text()
    assert "AE &lt; Demo" in html
    assert html.count("<svg ") == 3
    assert "https://" not in html and "http://" not in html.replace('http://www.w3.org/2000/svg', '')
    assert "application/json" in html and "Show operations" in html
    assert "data-ir-ids" in html
    for level in result["rendered_levels"]:
        assert (tmp_path / "viewer" / level / "architecture.svg").exists()
        details = json.loads((tmp_path / "viewer" / level / "presentation-manifest.json").read_text())
        assert details["visual_elements"]
        assert all(x["ir_node_ids"] for x in details["visual_elements"].values())


def test_unproven_instance_view_is_blocked_not_imagined(tmp_path):
    bundle = make_bundle(tmp_path)
    result = build_viewer(bundle, tmp_path / "viewer", view="instance")
    assert result["rendered_levels"] == []
    assert result["blocked_levels"] == ["overview", "block", "operation"]
    html = (tmp_path / "viewer/index.html").read_text()
    assert "<svg " not in html
    assert "unsupported by the available source evidence" in html
    assert not (tmp_path / "viewer/operation/architecture.svg").exists()


def test_corruption_removes_stale_outputs_and_denies_render(tmp_path):
    bundle = make_bundle(tmp_path)
    first = build_viewer(bundle, tmp_path / "viewer", view="structure")
    assert first["rendered_levels"]
    ir = bundle / "architecture.ir.json"
    ir.write_bytes(ir.read_bytes() + b" ")
    second = build_viewer(bundle, tmp_path / "viewer", view="structure")
    assert not second["rendered_levels"]
    assert not (tmp_path / "viewer/operation/architecture.svg").exists()


def test_installed_cli_viewer_entrypoint(tmp_path):
    bundle = make_bundle(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    run = subprocess.run(
        [sys.executable, "-m", "ae.cli", "viewer", str(bundle),
         "--view", "structure", "--output", str(tmp_path / "web")],
        capture_output=True, text=True, env=env,
    )
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout)
    assert len(report["rendered_levels"]) == 3
    assert (tmp_path / "web/index.html").is_file()
