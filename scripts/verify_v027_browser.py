"""Headless Chromium acceptance for the single-file offline Architecture Viewer."""
from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from ae.sae.api import extract
from ae.presenter.web_viewer import build_viewer


def verify(output: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    bundle = output / "bundle"
    destination = output / "viewer"
    extract(root / "examples/cnn_residual_mvp", bundle, entrypoint="model:CNN")
    report = build_viewer(bundle, destination, view="auto", title="AE Source Hierarchy")
    assert report["graph_type"] == "structure"
    assert report["rendered_levels"] == ["overview", "block", "operation"]
    manifest = json.loads((destination / "overview/presentation-manifest.json").read_text())
    model_id = next(vid for vid, n in manifest["visual_elements"].items()
                    if n["kind"] == "model")
    assert len(manifest["visual_elements"][model_id]["member_ir_node_ids"]) > 1

    errors = []
    external = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda r: external.append(r.url) if r.url.startswith(("http:", "https:")) else None)
        page.goto((destination / "index.html").resolve().as_uri())
        assert page.locator("#graph-overview svg").is_visible()
        assert page.get_by_role("button", name="Architecture").get_attribute("aria-pressed") == "true"

        page.locator("#search").fill("conv")
        search_count = page.locator("#search-results").inner_text()
        assert "matching nodes" in search_count
        page.locator("#search").press("Enter")
        assert "Label:" in page.locator("#details").inner_text()
        page.locator("#search").fill("")

        page.locator(f'rect[data-parent-id="{model_id}"]').click(position={"x":6,"y":6})
        assert page.locator("#drill").is_enabled()
        page.locator("#drill").click()
        assert page.get_by_role("button", name="Operation").get_attribute("aria-pressed") == "true"
        highlights = page.locator("#graph-operation svg rect.ae-selected").count()
        assert highlights >= 1, "Source hierarchy drill-down matched zero original operations"
        zoom_old = page.locator("#evidence-status").inner_text()
        page.locator("#zoom-in").click()
        zoom_new = page.locator("#evidence-status").inner_text()
        assert zoom_old != zoom_new
        assert "Evidence:" in zoom_new
        page.locator("#zoom-fit").click()
        page.screenshot(path=str(output / "offline-viewer-v027.png"), full_page=False)

        browser.close()
    assert not errors, errors
    assert not external, external
    summary = {
        "status": "passed", "view": report["graph_type"],
        "levels": report["rendered_levels"], "source_drill_highlights": highlights,
        "search": search_count, "browser_page_errors": errors,
        "external_network_requests": external, "screenshot": "offline-viewer-v027.png",
    }
    (output / "browser-report.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    return summary


if __name__ == "__main__":
    out = Path("benchmark-output/v027-browser")
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(verify(out), indent=2))
