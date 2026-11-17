"""Fully offline, evidence-preserving Architecture/Block/Operation viewer.

The viewer is presentation-only. It never executes the source model, derives
unverified tensor connections, or introduces additional graph operations.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

from .api import present
from .reader import load_bundle


HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:">
<title>__AE_TITLE__</title>
<style>
:root { color-scheme:light; --ink:#17191d;--muted:#60656d;--line:#d8dadc;--bg:#f6f7f7;--panel:#fff; }
*{box-sizing:border-box}body{margin:0;color:var(--ink);font:13px/1.5 Inter,system-ui,Arial,sans-serif;background:var(--bg)}
header{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:16px 22px;background:#fff;border-bottom:1px solid var(--line)}
h1{margin:0;font-size:17px;font-weight:650}header p{margin:2px 0 0;color:var(--muted);font-size:11px}
.controls{display:flex;align-items:center;gap:7px;flex-wrap:wrap}
#search{min-width:160px;max-width:270px;padding:7px 10px;border:1px solid var(--line);border-radius:6px;font:inherit}
#search:focus-visible{outline:2px solid #527996;outline-offset:2px}
button{font:inherit;border:1px solid var(--line);background:#fff;border-radius:6px;padding:7px 12px;color:var(--ink);cursor:pointer}
button[aria-pressed="true"]{background:#181a1d;color:#fff;border-color:#181a1d}
button:disabled{cursor:not-allowed;opacity:.35}button:focus-visible{outline:2px solid #527996;outline-offset:2px}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 300px;min-height:calc(100vh - 81px)}
main{min-width:0;position:relative}.toolbar{display:flex;align-items:center;gap:9px;padding:9px 16px;border-bottom:1px solid var(--line);background:#fff}
#evidence-status{margin-left:auto;font-size:11px;color:var(--muted)}
.graph{overflow:auto;height:calc(100vh - 136px);background:#fff;cursor:grab;touch-action:pan-x pan-y}
.graph.dragging{cursor:grabbing}.graph[hidden]{display:none}.shell{margin:24px;position:relative}
.shell svg{display:block;transform-origin:top left;max-width:none}
svg rect[id^="v-"],svg rect[data-parent-id]{cursor:pointer;transition:opacity .15s,stroke-width .15s}
svg rect.ae-dim,svg path.ae-dim{opacity:.12}
svg rect.ae-selected{stroke:#b26023!important;stroke-width:3!important;opacity:1!important}
aside{padding:18px 18px 28px;background:#fcfcfc;border-left:1px solid var(--line);overflow:auto;max-height:calc(100vh - 81px)}
aside h2{font-size:14px;margin:0 0 10px}#details{white-space:pre-wrap;overflow-wrap:anywhere;color:#33363b;font:12px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace}
.hint{font-size:11px;color:var(--muted);margin:12px 0 18px}
#drill{width:100%;margin:0 0 18px}footer{border-top:1px solid var(--line);margin-top:18px;padding-top:12px;font-size:11px;color:var(--muted)}
@media(max-width:900px){header{align-items:flex-start;flex-direction:column}.layout{grid-template-columns:1fr}.graph{height:65vh}aside{border-top:1px solid var(--line);border-left:0;max-height:none}}
</style></head>
<body>
<header><div><h1>__AE_TITLE__</h1><p>Architecture Extractor · Source-backed interactive inspection · Works offline</p></div>
<nav class="controls" aria-label="Graph resolution">
<button type="button" data-level="overview">Architecture</button>
<button type="button" data-level="block">Block</button>
<button type="button" data-level="operation">Operation</button>
</nav></header>
<div class="layout"><main>
<div class="toolbar"><input id="search" type="search" placeholder="Find module / operation" aria-label="Search visible graph nodes"><span id="search-results" role="status"></span><button type="button" id="zoom-out" aria-label="Zoom out">−</button><button type="button" id="zoom-in" aria-label="Zoom in">+</button>
<button type="button" id="zoom-fit">Fit width</button>
<span id="evidence-status" role="status"></span></div>
__AE_PANELS__
</main><aside><h2>Source provenance</h2><p class="hint">Select a component to inspect its original IR identity and source span. Click “Show operations” to highlight its source-proven operations. This does not infer hidden runtime edges.</p>
<button type="button" id="drill" disabled>Show operations</button>
<div id="details" aria-live="polite">Select an operation or block.</div>
<footer>Only graph evidence present in the extraction bundle is displayed. Partial and unsupported views remain labeled; color is a secondary visual cue, not proof of execution.</footer>
</aside></div>
<script id="ae-data" type="application/json">__AE_DATA__</script>
<script>
"use strict";
const data=JSON.parse(document.getElementById("ae-data").textContent);
const levels=["overview","block","operation"];
let active=levels.find(k=>data.levels[k].rendered)||"overview";
let scale=1, selected=null, lastDrag=false;
const panes=Object.fromEntries(levels.map(k=>[k,document.getElementById("graph-"+k)]));
function clearHighlight(){
  document.querySelectorAll("svg .ae-dim,svg .ae-selected").forEach(n=>n.classList.remove("ae-dim","ae-selected"));
}
function fit(){
  const svg=panes[active].querySelector("svg");if(!svg)return;
  const width=Number(svg.getAttribute("width"))||svg.viewBox.baseVal.width;
  const height=Number(svg.getAttribute("height"))||svg.viewBox.baseVal.height;
  scale=Math.max(.025,Math.min(2,(panes[active].clientWidth-56)/Math.max(width,1)));
  resize(width,height);
}
function resize(width,height){
  const svg=panes[active].querySelector("svg");if(!svg)return;
  width=width||Number(svg.getAttribute("width"));height=height||Number(svg.getAttribute("height"));
  svg.style.transform="scale("+scale+")";
  svg.parentElement.style.width=(width*scale)+"px";
  svg.parentElement.style.height=(height*scale)+"px";
  document.getElementById("evidence-status").textContent="Evidence: "+data.levels[active].status+" · Zoom: "+Math.round(scale*100)+"%";
}
function show(level){
  if(!data.levels[level].rendered)return;
  active=level;selected=null;clearHighlight();
  for(const k of levels)panes[k].hidden=k!==level;
  document.querySelectorAll("button[data-level]").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.level===level)));
  document.getElementById("drill").disabled=true;
  document.getElementById("details").textContent="View: "+level+"\nEvidence: "+data.levels[level].status+
    "\nVisible components: "+Object.keys(data.levels[level].elements).length+
    "\nVisible links: "+Object.keys(data.levels[level].links).length+
    "\n\nSelect a component to inspect exact evidence.";
  panes[level].scrollTop=0;panes[level].scrollLeft=0;fit();
  searchGraph();
}
function inspect(id){
  const info=data.levels[active].elements[id];
  if(!info)return;
  selected=info;clearHighlight();
  const rect=document.getElementById("v-"+id);if(rect)rect.classList.add("ae-selected");
  const refs=(info.source_refs||[]).map(r=>(r.path||"?")+" : "+(r.start_line??"?")+"-"+(r.end_line??"?"));
  const lines=[
    "Visual ID: "+id,
    "Kind: "+info.kind,
    "Label: "+(info.label||"unknown"),
    "Tensor shape: "+(info.shape||"not proven"),
    "Evidence: "+(info.evidence||data.levels[active].status),
    "Semantic category: "+(info.semantic_category||"unclassified"),
    "Original IR IDs ("+(info.ir_node_ids||[]).length+"):",
    ...(info.ir_node_ids||[]),
    "Source spans ("+refs.length+"):",
    ...refs,
  ];
  document.getElementById("details").textContent=lines.join("\n");
  document.getElementById("drill").disabled=active==="operation"||!data.levels.operation.rendered;
}
function drill(){
  if(!selected||!data.levels.operation.rendered)return;
  const ids=new Set(selected.member_ir_node_ids||selected.ir_node_ids||[]);
  show("operation");
  const operation=data.levels.operation;
  const matched=new Set(Object.entries(operation.elements)
    .filter(([id,entry])=>(entry.ir_node_ids||[]).some(x=>ids.has(x))).map(([id])=>id));
  if(!matched.size){
    document.getElementById("details").textContent="No source-proven operation correspondence for this group. No graph was fabricated.";
    return;
  }
  for(const el of panes.operation.querySelectorAll('svg rect[id^="v-"]')){
    el.classList.add(matched.has(el.id.slice(2))?"ae-selected":"ae-dim");
  }
  const graphEdges=data.instance_edges||[];
  const kept=new Set(graphEdges.filter(e=>matched.has(e.source)||matched.has(e.target)).map(e=>e.id));
  for(const el of panes.operation.querySelectorAll('svg path[id^="l-"]')){
    if(graphEdges.length&&!kept.has(el.id.slice(2)))el.classList.add("ae-dim");
  }
  const first=panes.operation.querySelector("svg rect.ae-selected");
  if(first){const y=Number(first.getAttribute("y"))||0;panes.operation.scrollTop=Math.max(0,(y*scale)-110);}
  document.getElementById("details").textContent="Highlighted "+matched.size+" original operations.\nNo new nodes or edges were generated.";
}
function searchGraph(){
  const term=document.getElementById("search").value.trim().toLowerCase();
  const meta=data.levels[active].elements;
  const matches=new Set(Object.entries(meta).filter(([id,n])=>[
    n.label||"",n.kind||"",n.semantic_category||"",
    ...(n.source_refs||[]).map(r=>r.path||"")
  ].some(x=>x.toLowerCase().includes(term))).map(([id])=>id));
  const rects=panes[active].querySelectorAll('svg rect[id^="v-"]');
  for(const rect of rects){
    rect.classList.remove("ae-dim");
    if(term&&!matches.has(rect.id.slice(2)))rect.classList.add("ae-dim");
  }
  const counter=document.getElementById("search-results");
  counter.textContent=term?(matches.size+" matching nodes"):"";
  return matches;
}
document.getElementById("search").addEventListener("input",searchGraph);
document.getElementById("search").addEventListener("keydown",event=>{
  if(event.key!=="Enter")return;
  const matches=searchGraph();
  const found=[...panes[active].querySelectorAll('svg rect[id^="v-"]')]
    .find(el=>matches.has(el.id.slice(2)));
  if(found){inspect(found.id.slice(2));
    panes[active].scrollTop=Math.max(0,Number(found.getAttribute("y"))*scale-100);
    panes[active].scrollLeft=Math.max(0,Number(found.getAttribute("x"))*scale-100);
  }
});
document.querySelectorAll("button[data-level]").forEach(btn=>{
  btn.disabled=!data.levels[btn.dataset.level].rendered;
  btn.addEventListener("click",()=>show(btn.dataset.level));
});
document.getElementById("drill").addEventListener("click",drill);
document.getElementById("zoom-fit").addEventListener("click",fit);
for(const [id,mul] of [["zoom-in",1.25],["zoom-out",.8]]){
  document.getElementById(id).addEventListener("click",()=>{scale=Math.min(4,Math.max(.025,scale*mul));resize();});
}
for(const pane of Object.values(panes)){
  let origin=null;
  pane.addEventListener("pointerdown",e=>{origin={x:e.clientX,y:e.clientY,left:pane.scrollLeft,top:pane.scrollTop};lastDrag=false;});
  pane.addEventListener("pointermove",e=>{
    if(!origin)return;
    const dx=e.clientX-origin.x,dy=e.clientY-origin.y;
    if(Math.abs(dx)+Math.abs(dy)>5){lastDrag=true;pane.classList.add("dragging");pane.scrollLeft=origin.left-dx;pane.scrollTop=origin.top-dy;}
  });
  pane.addEventListener("pointerup",()=>{origin=null;pane.classList.remove("dragging");});
  pane.addEventListener("pointerleave",()=>{origin=null;pane.classList.remove("dragging");});
  pane.addEventListener("click",e=>{if(lastDrag){lastDrag=false;return;}const rect=e.target.closest('rect[id^="v-"],rect[data-parent-id]');if(rect)inspect(rect.getAttribute("data-parent-id")||rect.id.slice(2));});
}
show(active);
</script></body></html>
"""


def _safe_json(data: dict) -> str:
    # A source label must never close the inert JSON script element.
    return json.dumps(data, sort_keys=True, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def build_viewer(bundle_path: str | Path, output: str | Path, *,
                 view: str = "instance", title: str = "Architecture Extractor") -> dict:
    """Produce offline HTML and source-proven SVGs; keep blocked views blocked."""
    if view not in ("auto", "instance", "nested", "flow", "structure"):
        raise ValueError("Unsupported viewer graph type: " + view)
    if view == "auto":
        source = load_bundle(bundle_path).directory
        candidate = source / "instance-forward.json"
        try:
            snapshot = json.loads(candidate.read_text())
            scopes = {
                "pinned_torchvision_0_25_0_efficientnet_b0",
                "pinned_torchvision_0_25_0_vgg16",
                "pinned_torchvision_0_25_0_mobilenet_v3_small",
                "pinned_timm_1_0_22_vit_tiny_patch16_224",
            }
            view = ("instance" if snapshot.get("status") == "expanded_static"
                    and snapshot.get("scope") in scopes else "structure")
        except (OSError, ValueError, TypeError):
            view = "structure"
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    levels = {}
    panels = []
    for level in ("overview", "block", "operation"):
        out = folder / level
        result = present(bundle_path, out, level=level, view=view, format="svg")
        manifest = json.loads((out / "presentation-manifest.json").read_text())
        health = json.loads((out / "presentation-health.json").read_text())
        if result["render_allowed"]:
            image = (out / "architecture.svg").read_text()
        else:
            image = '<p role="status">This view is unsupported by the available source evidence.</p>'
        levels[level] = {
            "rendered": bool(result["render_allowed"]),
            "status": health["status"],
            "elements": manifest["visual_elements"],
            "links": manifest["visual_links"],
            "collapsed_edge_ids": manifest.get("collapsed_instance_edge_ids", []),
            "omitted_node_ids": manifest.get("omitted_instance_node_ids", []),
            "omitted_edge_ids": manifest.get("omitted_instance_edge_ids", []),
            "issues": [x.get("code", "unknown") for x in health["issues"]],
        }
        panels.append('<div class="graph" id="graph-' + level + '" hidden><div class="shell">' + image + "</div></div>")
    if view == "instance":
        instance_file = Path(bundle_path) / "instance-forward.json"
        try:
            instance_graph = json.loads(instance_file.read_text())
            edges = [
                {"id": x["id"], "source": x["source"], "target": x["target"]}
                for x in instance_graph["edges"]
            ] if instance_graph.get("status") == "expanded_static" else []
        except (OSError, ValueError, KeyError, TypeError):
            edges = []
    else:
        edges = []
    payload = {"schema_version": "1.0", "graph_type": view,
               "levels": levels, "instance_edges": edges}
    page = (HTML.replace("__AE_TITLE__", html.escape(title, quote=True))
            .replace("__AE_PANELS__", "\n".join(panels))
            .replace("__AE_DATA__", _safe_json(payload)))
    (folder / "index.html").write_text(page, encoding="utf-8")
    status = {
        "schema_version": "1.0", "graph_type": view,
        "viewer": "index.html",
        "rendered_levels": [k for k in levels if levels[k]["rendered"]],
        "blocked_levels": [k for k in levels if not levels[k]["rendered"]],
        "level_status": {k: levels[k]["status"] for k in levels},
        "html_sha256": hashlib.sha256(page.encode("utf-8")).hexdigest(),
        "source_execution": False,
    }
    (folder / "viewer-manifest.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return status
