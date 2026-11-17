"""README gallery stays source-generated, committed, and correctly linked."""
from __future__ import annotations

import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets/readme"
EXPECTED = {
    "vgg16-architecture.svg", "vgg16-block.svg",
    "vgg16-operation-focus.svg", "vgg16-operation-full.svg",
    "vgg16-viewer.png",
}


def test_readme_gallery_files_and_manifest() -> None:
    manifest = json.loads((ASSETS / "gallery-manifest.json").read_text())
    assert manifest["model"] == "torchvision.models.vgg:vgg16"
    assert manifest["torchvision"] == "0.25.0"
    assert manifest["input_shape"] == [1, 3, 224, 224]
    assert manifest["reference_status"] == "source_and_fx_verified"
    assert manifest["fx_operations"] == 42
    assert manifest["ordered_dependencies"] == 41
    assert set(manifest["gallery"]) == EXPECTED
    readme = (ROOT / "README.md").read_text()
    for name in EXPECTED:
        path = ASSETS / name
        assert path.is_file() and path.stat().st_size > 100
        assert "docs/assets/readme/" + name in readme


def test_reference_svgs_are_genuine_vgg16_vector_documents() -> None:
    for path in ASSETS.glob("*.svg"):
        root = ET.parse(path).getroot()
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        assert "viewBox" in root.attrib
        titles = [e.text or "" for e in root if e.tag.endswith("}title")]
        assert titles and "torchvision.models.vgg:vgg16" in titles[0]
        assert root.find(".//{http://www.w3.org/2000/svg}rect") is not None


def test_viewer_screenshot_is_valid_png() -> None:
    data = (ASSETS / "vgg16-viewer.png").read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (1440, 900)
