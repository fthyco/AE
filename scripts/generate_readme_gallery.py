"""Regenerate the README images directly from independently verified AE output.

Requires torch==2.10.0, torchvision==0.25.0, AE[dev,render],
playwright==1.56.0 and Chromium.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

from ae.presenter.api import present
from ae.presenter.web_viewer import build_viewer
from scripts.verify_full_vgg16_v023 import verify


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs/assets/readme"


def generate() -> None:
    with tempfile.TemporaryDirectory(prefix="ae-readme-gallery-") as tmp:
        result = verify(Path(tmp) / "vgg16")
        assert result["status"] == "source_and_fx_verified"
        assert all(c["matching"] for c in result["cases"])
        case = Path(tmp) / "vgg16/1x3x224x224"
        assets = {
            "vgg16-architecture.svg": case / "figures-overview/architecture.svg",
            "vgg16-block.svg": case / "figures-block/architecture.svg",
            "vgg16-operation-full.svg": case / "figures/architecture.svg",
        }
        focus_dir = Path(tmp) / "focus"
        focus = present(case / "bundle", focus_dir, view="instance",
                        level="operation", focus="stage.2", format="svg")
        assert focus["render_allowed"], focus
        assets["vgg16-operation-focus.svg"] = focus_dir / "architecture.svg"
        DEST.mkdir(parents=True, exist_ok=True)
        for name, path in assets.items():
            assert path.is_file(), path
            svg = path.read_text(encoding="utf-8")
            assert svg.startswith("<svg ") and "torchvision.models.vgg:vgg16" in svg
            shutil.copyfile(path, DEST / name)

        viewer_dir = Path(tmp) / "viewer"
        viewer = build_viewer(case / "bundle", viewer_dir,
                              view="instance", title="Architecture Extractor")
        assert viewer["rendered_levels"] == ["overview", "block", "operation"]
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900},
                                    device_scale_factor=1)
            errors, external = [], []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("request", lambda r: external.append(r.url)
                    if r.url.startswith(("http:", "https:")) else None)
            page.goto((viewer_dir / "index.html").as_uri())
            page.get_by_role("button", name="Block").click()
            page.locator("#zoom-fit").click()
            assert page.locator("#graph-block svg").is_visible()
            page.screenshot(path=str(DEST / "vgg16-viewer.png"))
            browser.close()
            assert not errors, errors
            assert not external, external

        report = {"model": "torchvision.models.vgg:vgg16",
                  "torchvision": "0.25.0", "input_shape": [1, 3, 224, 224],
                  "fx_operations": 42, "ordered_dependencies": 41,
                  "reference_status": result["status"],
                  "gallery": sorted([*assets, "vgg16-viewer.png"])}
        (DEST / "gallery-manifest.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    generate()
