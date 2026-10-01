"""An interactive page: cliffs (LP) versus smooth (surrogate) while moving a joint on a rotten block.

    python examples/proto2d/cliffs_demo.py                      # -> out/proto2d/cliffs_demo.html
    python examples/proto2d/cliffs_demo.py --fields 3 --n1 41 --n2 29 --families tenon dovetail

Everything is precomputed here (the page cannot run the LP or the network):
for every family with a trained model and a few held-out fields (the same
fields for every family), two of the family's parameters are swept on a grid;
at every point the LP capacity and forces, the surrogate's capacity and its
gradient, the sound wood removed and the contact geometry are stored. The page
then lets you pick a family and a field, drag the two parameters, shows both
landscapes, the LP's slope (finite difference on the grid) next to the
surrogate's, a 1D slice with the staircase and the curve, and two walkers that
climb the objective using each slope from a start point you pick.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from repair.proto2d import dataset as ds, draw as dw, families as fm, model as sp
    from repair.proto2d.train import load_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from repair.proto2d import dataset as ds, draw as dw, families as fm, model as sp
    from repair.proto2d.train import load_model

X0 = 4.0          # the picture spans x in [X0, length]
SCALE = 60
SWEEP = {"tenon": ("a", "ell"), "tenon_flip": ("a", "ell"), "dovetail": ("a", "ell"), "dovetail_flip": ("a", "ell"),
         "hooked_scarf": ("a", "L"), "hooked_scarf_flip": ("a", "L")}


def i16(arr, scale):
    """Base64 of an int16 array: compact per-point geometry for the page."""
    return base64.b64encode(np.round(np.asarray(arr, float) * scale).astype(np.int16).tobytes()).decode("ascii")


def precompute(fam, model, ck, f, raster_t, g1, g2, n1, n2, fixed, device, grid, st, b):
    n = ck["normalization"]; lo, hi = np.array(n["lo"], np.float32), np.array(n["hi"], np.float32)
    names = n["param_names"]
    pts, geom = [], []
    LP = np.zeros((len(g2), len(g1))); R = np.zeros_like(LP)
    for i, v2 in enumerate(g2):
        for j, v1 in enumerate(g1):
            p = fam.feasible(dict(fixed, **{n1: float(v1), n2: float(v2)}), b)
            M, r, cs, Fc = fm.evaluate(fam, p, f, st, grid)
            LP[i, j], R[i, j] = M, r
            pts.append([p[k] for k in names])
            segs = [s for lay in fam.layers(p, b) for s in fm.interface_segments(lay, b)]
            geom.append({"s": [[round(float(v), 3) for v in (s[0][0], s[0][1], s[1][0], s[1][1])] for s in segs],
                         "c": i16([[c.point[0], c.point[1]] for c in cs], 1000), "l": "".join("1" if c.live else "0" for c in cs),
                         "f": i16(Fc, 10)})
    P = torch.tensor((np.array(pts, np.float32) - lo) / (hi - lo), device=device, requires_grad=True)
    with torch.no_grad():
        code = model.encode(raster_t.unsqueeze(0))
    out = model.from_code(code.expand(len(P), -1), P)
    out.sum().backward()
    NN = out.detach().cpu().numpy().reshape(LP.shape) * n["peak"]
    G = P.grad.cpu().numpy() * n["peak"] / (hi - lo)                       # d capacity / d physical parameter
    i1, i2 = names.index(n1), names.index(n2)
    return LP, np.clip(NN, 0, None), R, G[:, i1].reshape(LP.shape), G[:, i2].reshape(LP.shape), geom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/proto2d")
    ap.add_argument("--models", default="out/proto2d/models")
    ap.add_argument("--out", default="out/proto2d/cliffs_demo.html")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--fields", type=int, default=3)
    ap.add_argument("--n1", type=int, default=41)
    ap.add_argument("--n2", type=int, default=29)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keys = a.families or [k for k in SWEEP if (Path(a.models) / k / "model.pt").exists()]
    fields, fams = ds.load(a.data)
    b, grid, st = sp.Block(), sp.Grid(), sp.Statics()
    rng = np.random.default_rng(a.seed)
    test_fields = np.unique(fams[keys[0]]["field"][fams[keys[0]]["test"]])
    picked = [int(x) for x in rng.choice(test_fields, size=a.fields, replace=False)]
    raster_all = torch.tensor(fields["raster"].astype(np.float32), device=device).unsqueeze(1)
    out_fields = []
    for fi in picked:
        f = ds.field_of(fields, fi, b)
        out_fields.append({"id": fi, "knobs": {k: round(float(v), 2) for k, v in zip(fields["knob_names"], fields["knobs"][fi])},
                           "png": dw.png_base64(dw.draw(fm.FAMILIES["butt"], {"a": b.length - 1e-3}, f, None, b, SCALE, X0))})
    out_fams = {}
    for k in keys:
        fam = fm.FAMILIES[k]; n1, n2 = SWEEP[k]
        model, ck = load_model(Path(a.models) / k / "model.pt", device)
        q1 = next(q for q in fam.params if q.name == n1); q2 = next(q for q in fam.params if q.name == n2)
        g1 = np.linspace(q1.lo, q1.hi, a.n1); g2 = np.linspace(q2.lo, q2.hi, a.n2)
        fixed = {kk: v for kk, v in fam.defaults().items() if kk not in (n1, n2)}
        per = []
        for fi in picked:
            f = ds.field_of(fields, fi, b)
            LP, NN, R, G1, G2, geom = precompute(fam, model, ck, f, raster_all[fi], g1, g2, n1, n2, fixed, device, grid, st, b)
            per.append({"lp": np.round(LP, 1).tolist(), "nn": np.round(NN, 1).tolist(), "R": np.round(R, 2).tolist(),
                        "g1": np.round(G1, 1).tolist(), "g2": np.round(G2, 1).tolist(), "geom": geom})
            print(f"{k:18s} field {fi}: LP peak {LP.max():6.1f}, surrogate peak {NN.max():6.1f}, rmse {np.sqrt(((LP - NN) ** 2).mean()):5.1f}")
        out_fams[k] = {"title": fam.title, "source": fam.source, "p1": {"name": n1, "doc": q1.doc}, "p2": {"name": n2, "doc": q2.doc},
                       "g1": g1.round(4).tolist(), "g2": g2.round(4).tolist(), "fixed": {kk: round(v, 2) for kk, v in fixed.items()},
                       "rmse_test": ck["val"]["rmse_capacity"], "r2_test": ck["val"]["r2"], "fields": per}
    data = {"x0": X0, "scale": SCALE, "length": b.length, "height": b.height, "fields": out_fields, "families": out_fams}
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(html)
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Capacity, cliffs, and a smooth stand-in</title>
<style>
body{font-family:sans-serif;max-width:1100px;margin:auto;padding:16px;color:#222;line-height:1.35}
h1{font-size:22px;margin:0 0 6px} h2{font-size:16px;margin:22px 0 6px} p{margin:6px 0}
.row{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start} @media(max-width:760px){.row{grid-template-columns:1fr}}
.top{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;align-items:start} @media(max-width:900px){.top{grid-template-columns:1fr 1fr}} @media(max-width:560px){.top{grid-template-columns:1fr}}
body.nonn .nnonly{display:none} body.nonn .read{grid-template-columns:auto 1fr} .lponly{display:none} body.nonn .lponly{display:inline}
.pic{position:relative;width:100%} .pic img{width:100%;display:block} .pic svg{position:absolute;left:0;top:0;width:100%;height:100%}
label{display:block;font-size:13px;margin-top:6px} input[type=range]{width:100%} input#lam{width:220px;display:inline-block;vertical-align:middle;margin-left:8px}
.read{font-size:14px;margin:8px 0;display:grid;grid-template-columns:auto 1fr 1fr;gap:2px 12px} .read b{font-weight:600}
.lp{color:#111} .nn{color:#e8701a} .muted{color:#777;font-size:12px}
canvas{width:100%;border:1px solid #ccc;display:block;cursor:crosshair} .hm{font-size:12px}
button{margin:4px 6px 4px 0;padding:5px 10px;font-size:13px} .note{background:#f6f3ee;border-left:3px solid #e8701a;padding:8px 10px;font-size:14px}
</style></head><body>
<h1>Repairing a rotten block: capacity, cliffs, and a smooth stand-in</h1>
<label>joint family <select id="family"></select> <span class="muted" id="famdoc"></span></label>
<label>damage field <select id="field"></select> <span class="muted" id="knobs"></span> &nbsp; <input type="checkbox" id="shownn" checked> <span class="nn">show the surrogate</span> <span class="muted">(off: just the exact solver)</span></label>

<h2>1. Capacity: how much bending a repair can take</h2>
<div class="row">
 <div>
  <div class="pic"><img id="img"><svg id="ov" viewBox="0 0 480 120" preserveAspectRatio="none"></svg></div>
  <div class="muted">retained wood coloured by rot; the new wood is pale; green = live contact, red = dead</div>
  <label><span id="p1name"></span> <span id="aval"></span><input type="range" id="a" min="0" step="1"></label>
  <label><span id="p2name"></span> <span id="lval"></span><input type="range" id="l" min="0" step="1"></label>
 </div>
 <div>
  <p>The rotten end of the block is cut away and a new piece is fitted with a joint. The two pieces touch along the black interface, sampled at the dots.</p>
  <p><b>Capacity is a moment</b>: the largest twist the joint can balance before it slips or comes apart. An exact solver (a linear program, LP) finds it by placing pushes on the contact dots (orange arrows): a dot can push but not pull, friction limits sideways force, and a dot backed by rot (red) carries nothing. The pushes add up to zero net force, but they act at different places, so their lever arms give a net moment. The largest twist they can balance, the curved arrow on the new piece, is the capacity. A plain cut has capacity zero: all its pushes point the same way, so they cannot balance any twist.</p>
  <div class="read">
   <span>capacity (exact solver)</span><span class="lp" id="lpM"></span><span class="nn nnonly"><span class="muted">surrogate says</span> <span id="nnM"></span></span>
   <span>sound wood removed</span><span id="R"></span><span class="nnonly"></span>
   <span>the drawn pushes</span><span id="chk"></span><span class="nnonly"></span>
  </div>
  <label><input type="checkbox" id="showf" checked> show the pushes and the twist they balance</label>
  <p class="muted">Drag the sliders. Watch dots turn red as faces cross into rot, the pushes move to whatever is left, and the capacity drop.</p>
 </div>
</div>

<h2>2. Cliffs: capacity changes in steps, not slopes</h2>
<p>A face carries load until the moment it crosses into rot, then nothing. So as the cut moves, capacity stays flat and then drops in a step. Searching for the best joint means nudging the design and asking "better or worse?", and on a flat step the answer is always "no change". The best joint sits right at a step edge, as close to the rot as it dares.<span class="nnonly"> The trained surrogate is a smooth hill through the same staircase, so a nudge always points somewhere.</span></p>
<div class="top">
 <div><div class="hm"><b class="lp">LP capacity</b> over (<span class="pn"></span>). Click to set the walkers' start.</div><canvas id="hmLP" width="366" height="246"></canvas></div>
 <div class="nnonly"><div class="hm"><b class="nn">surrogate capacity</b> over (<span class="pn"></span>)</div><canvas id="hmNN" width="366" height="246"></canvas></div>
 <div>
  <div class="read">
   <span></span><b class="lp">exact solver (LP)</b><b class="nn nnonly">surrogate</b>
   <span>slope along <span class="p1n"></span></span><span class="lp" id="lpGa"></span><span class="nn nnonly" id="nnGa"></span>
   <span>slope along <span class="p2n"></span></span><span class="lp" id="lpGl"></span><span class="nn nnonly" id="nnGl"></span>
  </div>
  <div class="muted">LP slope = finite difference across one grid step. On a plateau it is exactly 0: no direction to improve.<span class="nnonly"> The surrogate's slope is its true gradient.</span></div>
 </div>
</div>
<svg id="slice" viewBox="0 0 1000 220" width="100%"></svg>
<div class="muted">A slice along <span class="p1n"></span> at the current <span class="p2n"></span>.</div>
<h2>3. Let <span class="nnonly">two walkers</span><span class="lponly">a walker</span> climb</h2>
<p>Objective = capacity / peak &minus; &lambda; &middot; sound wood removed / max. <span class="nnonly">Each walker starts at the clicked point and takes small uphill steps, one using the LP's finite-difference slope, one using the surrogate's gradient. Both end points are then scored by the exact solver.</span><span class="lponly">The walker starts at the clicked point and takes small uphill steps using the LP's finite-difference slope: on a plateau it feels no slope in capacity at all.</span></p>
<label>&lambda; (weight on sound wood removed) <span id="lamval"></span><input type="range" id="lam" min="0" max="4" step="0.1" value="1.5"></label>
<label>landscapes show <select id="show"><option value="cap">capacity</option><option value="obj">objective (changes with &lambda;)</option></select></label>
<button id="walk"><span class="nnonly">walk both</span><span class="lponly">walk</span></button><button id="reset">reset</button> <span id="jumps" style="display:none"><button id="jumpLP">go to the LP walker's end</button><button id="jumpNN" class="nnonly">go to the surrogate walker's end</button></span>
<div class="read">
 <span></span><b class="lp">LP-slope walker</b><b class="nn nnonly">surrogate-slope walker</b>
 <span>steps taken</span><span id="wsLP"></span><span class="nnonly" id="wsNN"></span>
 <span>ends at (<span class="pn"></span>)</span><span id="weLP"></span><span class="nnonly" id="weNN"></span>
 <span>objective there, scored by the LP</span><span id="woLP"></span><span class="nnonly" id="woNN"></span>
 <span>best on the grid, scored by the LP</span><span id="wbest"></span><span class="nnonly"></span>
</div>
<div class="note" id="verdict"></div>
<p class="muted"><span class="nnonly">Surrogate test error on held-out fields: RMSE <span id="rmse"></span> capacity units, R&sup2; <span id="r2"></span>. </span>Fixed parameters: <span id="fixed"></span>. Everything on this page was precomputed on a <span id="gridsz"></span> grid; the page only interpolates.</p>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const KEYS = Object.keys(D.families);
let FAM = D.families[KEYS[0]], A, L, NA, NL, F, FD, ia, il, start = null, paths = {lp: null, nn: null};
const b64i16 = s => { const bin = atob(s), out = new Int16Array(bin.length / 2); for (let i = 0; i < out.length; i++) out[i] = (bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8)) << 16 >> 16; return out; };
const sel = $("field"); D.fields.forEach((f, i) => { const o = document.createElement("option"); o.value = i; o.textContent = "held-out field " + f.id; sel.appendChild(o); });
const fsel = $("family"); KEYS.forEach(k => { const o = document.createElement("option"); o.value = k; o.textContent = k; fsel.appendChild(o); });
function setFamily(key, fieldIndex) {
  FAM = D.families[key]; A = FAM.g1; L = FAM.g2; NA = A.length; NL = L.length;
  FD = D.fields[fieldIndex]; F = FAM.fields[fieldIndex];
  ia = Math.round(NA * 0.6); il = Math.round(NL * 0.2); start = null; paths = {lp: null, nn: null};
  $("a").max = NA - 1; $("l").max = NL - 1; $("a").value = ia; $("l").value = il;
  $("p1name").textContent = FAM.p1.name + ": " + FAM.p1.doc; $("p2name").textContent = FAM.p2.name + ": " + FAM.p2.doc;
  document.querySelectorAll(".pn").forEach(e => e.textContent = FAM.p1.name + ", " + FAM.p2.name);
  document.querySelectorAll(".p1n").forEach(e => e.textContent = FAM.p1.name); document.querySelectorAll(".p2n").forEach(e => e.textContent = FAM.p2.name);
  $("famdoc").textContent = FAM.title + (FAM.source ? "  [MiGumi " + FAM.source + "]" : "");
  $("rmse").textContent = FAM.rmse_test.toFixed(1); $("r2").textContent = FAM.r2_test.toFixed(3); $("gridsz").textContent = NA + " x " + NL;
  $("fixed").textContent = Object.entries(FAM.fixed).map(([k, v]) => k + " = " + v).join(", ");
  ["wsLP", "wsNN", "weLP", "weNN", "woLP", "woNN", "wbest"].forEach(id => $(id).textContent = ""); $("verdict").textContent = "";
}
setFamily(KEYS[0], 0);
const peak = f => Math.max(...f.lp.flat()), Rmax = f => Math.max(...f.R.flat());
function obj(f, i, j, lam) { return f.lp[i][j] / peak(f) - lam * f.R[i][j] / Rmax(f); }
function objNN(f, i, j, lam) { return f.nn[i][j] / peak(f) - lam * f.R[i][j] / Rmax(f); }
function fd(M, i, j, axis) {   // finite difference on the grid, central where possible
  if (axis === 0) { const j0 = Math.max(j - 1, 0), j1 = Math.min(j + 1, NA - 1); return (M[i][j1] - M[i][j0]) / (A[j1] - A[j0]); }
  const i0 = Math.max(i - 1, 0), i1 = Math.min(i + 1, NL - 1); return (M[i1][j] - M[i0][j]) / (L[i1] - L[i0]);
}
function color(v, vmax) { const t = vmax > 0 ? v / vmax : 0; const r = Math.round(247 - 200 * t), g = Math.round(251 - 170 * t), b = Math.round(255 - 110 * t); return `rgb(${r},${g},${b})`; }
function landscape(M) {   // capacity, or the objective at the current lambda
  if ($("show").value === "cap") return M;
  const lam = +$("lam").value, pk = peak(F), rm = Rmax(F);
  return M.map((row, i) => row.map((v, j) => v / pk - lam * F.R[i][j] / rm));
}
function heat(cv, M0, label) {
  const M = landscape(M0);
  const ctx = cv.getContext("2d"), W = cv.width, H = cv.height, m = {l: 36, r: 6, t: 6, b: 24};
  const flat = M.flat(), vmin = Math.min(...flat), vmax = Math.max(...flat);
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
  const cw = (W - m.l - m.r) / NA, ch = (H - m.t - m.b) / NL;
  for (let i = 0; i < NL; i++) for (let j = 0; j < NA; j++) { ctx.fillStyle = color(M[i][j] - vmin, vmax - vmin); ctx.fillRect(m.l + j * cw, H - m.b - (i + 1) * ch, cw + 0.5, ch + 0.5); }
  ctx.fillStyle = "#333"; ctx.font = "11px sans-serif"; ctx.textAlign = "center";
  for (const j of [0, Math.floor(NA / 2), NA - 1]) ctx.fillText(A[j].toFixed(1), m.l + (j + 0.5) * cw, H - 8);
  ctx.textAlign = "right"; for (const i of [0, Math.floor(NL / 2), NL - 1]) ctx.fillText(L[i].toFixed(1), m.l - 4, H - m.b - (i + 0.5) * ch + 4);
  ctx.textAlign = "left"; ctx.fillText(FAM.p1.name + " →", W - 34, H - 8); ctx.save(); ctx.translate(10, 44); ctx.rotate(-Math.PI / 2); ctx.fillText(FAM.p2.name + " →", 0, 0); ctx.restore();
  const px = (i, j) => [m.l + (j + 0.5) * cw, H - m.b - (i + 0.5) * ch];
  for (const [key, col] of [["lp", "#111"], ["nn", "#e8701a"]]) { const p = paths[key]; if (!p) continue; ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); p.forEach(([i, j], k) => { const [x, y] = px(i, j); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); const [x, y] = px(...p[p.length - 1]); ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, 4, 0, 7); ctx.fill(); }
  if (start) { const [x, y] = px(...start); ctx.strokeStyle = "#444"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, y, 5, 0, 7); ctx.stroke(); }
  const [x, y] = px(il, ia); ctx.fillStyle = "#1ab07a"; ctx.beginPath(); ctx.arc(x, y, 5, 0, 7); ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.stroke();
  cv.onclick = e => { const r = cv.getBoundingClientRect(); const x = (e.clientX - r.left) * W / r.width, y = (e.clientY - r.top) * H / r.height; const j = Math.min(NA - 1, Math.max(0, Math.floor((x - m.l) / cw))), i = Math.min(NL - 1, Math.max(0, Math.floor((H - m.b - y) / ch))); start = [i, j]; paths = {lp: null, nn: null}; ia = j; il = i; $("a").value = j; $("l").value = i; render(); };
}
function overlay() {
  const g0 = F.geom[il * NA + ia], s = D.scale, X0 = D.x0, Hh = D.height;
  const cxy = b64i16(g0.c), fxy = b64i16(g0.f), nC = cxy.length / 2;
  const g = {s: g0.s, c: [], f: []};
  for (let n = 0; n < nC; n++) { g.c.push([cxy[2 * n] / 1000, cxy[2 * n + 1] / 1000, g0.l[n] === "1" ? 1 : 0]); g.f.push([fxy[2 * n] / 10, fxy[2 * n + 1] / 10]); }
  const P = (x, y) => [((x - X0) * s).toFixed(1), ((Hh - y) * s).toFixed(1)];
  let svg = "";
  for (const [x0, y0, x1, y1] of g.s) { const [a0, b0] = P(x0, y0), [a1, b1] = P(x1, y1); svg += `<line x1="${a0}" y1="${b0}" x2="${a1}" y2="${b1}" stroke="#000" stroke-width="2"/>`; }
  for (const [x, y, live] of g.c) { const [cx, cy] = P(x, y); svg += `<circle cx="${cx}" cy="${cy}" r="3.2" fill="${live ? '#14a014' : '#d23232'}"/>`; }
  let sfx = 0, sfy = 0, mom = 0, fmax = 1e-9;
  g.f.forEach(([fx, fy]) => { fmax = Math.max(fmax, Math.hypot(fx, fy)); });
  if ($("showf").checked) {
    svg = `<defs><marker id="ah" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#e8701a"/></marker></defs>` + svg;
    const k = 0.9 / fmax;                                   // longest arrow 0.9 block units
    g.c.forEach(([x, y], n) => { const [fx, fy] = g.f[n]; if (Math.hypot(fx, fy) < 1e-6) return;
      sfx += fx; sfy += fy; mom += x * fy - y * fx;
      const [x0, y0] = P(x, y), [x1, y1] = P(x + k * fx, y + k * fy);
      svg += `<line x1="${x0}" y1="${y0}" x2="${x1}" y2="${y1}" stroke="#e8701a" stroke-width="2" marker-end="url(#ah)"/>`; });
    const M = F.lp[il][ia];
    if (M > 1e-6) {                                           // the twist being balanced: a counter-clockwise arc on the new piece
      const cx = Math.min(A[ia] + 1.3, D.length - 0.6), cy = D.height / 2, r = 0.45;
      const arc = (t) => P(cx + r * Math.cos(t), cy + r * Math.sin(t));
      const [ax, ay] = arc(-0.6 * Math.PI), [bx, by] = arc(0.6 * Math.PI);
      svg += `<path d="M${ax},${ay} A${r * s},${r * s} 0 1 0 ${bx},${by}" fill="none" stroke="#333" stroke-width="2.5" marker-end="url(#ah2)"/>`
          + `<defs><marker id="ah2" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="4" markerHeight="4" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#333"/></marker></defs>`
          + `<text x="${P(cx, cy)[0]}" y="${P(cx, cy)[1] + 4}" font-size="11" text-anchor="middle" fill="#333">M</text>`;
    }
  }
  $("ov").innerHTML = svg;
  $("chk").textContent = $("showf").checked ? `net force (${sfx.toFixed(1)}, ${sfy.toFixed(1)}); moment ${(-mom).toFixed(1)} = capacity` : "";
}
function slice() {
  const W = 1000, H = 220, m = {l: 48, r: 10, t: 10, b: 30}, lp = landscape(F.lp)[il], nn = landscape(F.nn)[il];
  const lo0 = Math.min(0, ...lp, ...($("shownn").checked ? nn : [])), top0 = Math.max(1e-6, ...lp, ...($("shownn").checked ? nn : []));
  const top = top0 + 0.08 * (top0 - lo0), bot = lo0 - 0.08 * (top0 - lo0);
  const sx = j => m.l + j / (NA - 1) * (W - m.l - m.r), sy = v => m.t + (1 - (v - bot) / (top - bot)) * (H - m.t - m.b);
  let steps = "", curve = "";
  for (let j = 0; j < NA - 1; j++) steps += `${sx(j)},${sy(lp[j])} ${sx(j + 1)},${sy(lp[j])} `;
  const showNN = $("shownn").checked;
  for (let j = 0; j < NA; j++) curve += `${sx(j)},${sy(nn[j])} `;
  let ticks = ""; for (const j of [0, Math.floor(NA / 4), Math.floor(NA / 2), Math.floor(3 * NA / 4), NA - 1]) ticks += `<text x="${sx(j)}" y="${H - 10}" font-size="12" text-anchor="middle">${A[j].toFixed(1)}</text>`;
  let yt = ""; const isObj = $("show").value === "obj"; for (let k = 0; k < 4; k++) { const v = lo0 + k * (top0 - lo0) / 3; yt += `<text x="${m.l - 6}" y="${sy(v) + 4}" font-size="12" text-anchor="end">${isObj ? v.toFixed(2) : v.toFixed(0)}</text>`; }
  $("slice").innerHTML = `<rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="none" stroke="#ccc"/>
   <polyline points="${steps}" fill="none" stroke="#111" stroke-width="2"/>${showNN ? `<polyline points="${curve}" fill="none" stroke="#e8701a" stroke-width="2.5"/>` : ""}
   <line x1="${sx(ia)}" y1="${m.t}" x2="${sx(ia)}" y2="${H - m.b}" stroke="#1ab07a" stroke-width="2" stroke-dasharray="4 3"/>${ticks}${yt}
   <text x="${W - m.r}" y="${m.t + 14}" font-size="13" text-anchor="end"><tspan fill="#111">LP (staircase)</tspan>${showNN ? `   <tspan fill="#e8701a">surrogate (smooth)</tspan>` : ""}   <tspan fill="#777">${isObj ? "objective" : "capacity"}</tspan></text>
   <text x="${(m.l + W) / 2}" y="${H - 0}" font-size="12" text-anchor="middle">${FAM.p1.name}  (${FAM.p2.name} = ${L[il].toFixed(2)})</text>`;
}
function render() {
  $("aval").textContent = A[ia].toFixed(2); $("lval").textContent = L[il].toFixed(2);
  $("lpM").textContent = F.lp[il][ia].toFixed(1); $("nnM").textContent = F.nn[il][ia].toFixed(1); $("R").textContent = F.R[il][ia].toFixed(2);
  $("lpGa").textContent = fd(F.lp, il, ia, 0).toFixed(1); $("lpGl").textContent = fd(F.lp, il, ia, 1).toFixed(1);
  $("nnGa").textContent = F.g1[il][ia].toFixed(1); $("nnGl").textContent = F.g2[il][ia].toFixed(1);
  $("knobs").textContent = Object.entries(FD.knobs).map(([k, v]) => k + " " + v).join("  ");
  $("img").src = "data:image/png;base64," + FD.png; overlay(); slice(); heat($("hmLP"), F.lp, "LP"); heat($("hmNN"), F.nn, "surrogate");
  $("lamval").textContent = (+$("lam").value).toFixed(1);
}
function walk(useNN) {   // gradient ascent on the objective, in grid units, with a fixed step
  const lam = +$("lam").value, pk = peak(F), rm = Rmax(F), path = [[il, ia]];
  let i = il, j = ia, fi = i, fj = j;
  const da = A[1] - A[0], dl = L[1] - L[0], step = 0.35;   // step in grid cells along the normalized gradient
  for (let k = 0; k < 400; k++) {
    const I = Math.round(fi), J = Math.round(fj);
    let ga, gl;
    if (useNN) { ga = F.g1[I][J] / pk - lam * fd(F.R, I, J, 0) / rm; gl = F.g2[I][J] / pk - lam * fd(F.R, I, J, 1) / rm; }
    else { ga = fd(F.lp, I, J, 0) / pk - lam * fd(F.R, I, J, 0) / rm; gl = fd(F.lp, I, J, 1) / pk - lam * fd(F.R, I, J, 1) / rm; }
    const ca = ga * da, cl = gl * dl, n = Math.hypot(ca, cl);      // gradient per grid cell
    if (n < 1e-4) break;
    const sc = step * Math.min(1, n / 0.02) / n;                    // shrink steps where the slope is tiny
    fj = Math.min(NA - 1, Math.max(0, fj + sc * ca)); fi = Math.min(NL - 1, Math.max(0, fi + sc * cl));
    const Ni = Math.round(fi), Nj = Math.round(fj);
    if (Ni !== path[path.length - 1][0] || Nj !== path[path.length - 1][1]) path.push([Ni, Nj]);
    if (!useNN && path.length > 3 && obj(F, Ni, Nj, lam) < obj(F, I, J, lam) - 1e-9) { path.pop(); break; }  // stepped off a cliff: stop at the edge
  }
  return path;
}
$("walk").onclick = () => {
  if (!start) start = [il, ia];
  il = start[0]; ia = start[1];
  const showNN = $("shownn").checked;
  paths.lp = walk(false); paths.nn = showNN ? walk(true) : null;
  const lam = +$("lam").value; let best = -1e9, bi = 0, bj = 0;
  for (let i = 0; i < NL; i++) for (let j = 0; j < NA; j++) { const o = obj(F, i, j, lam); if (o > best) { best = o; bi = i; bj = j; } }
  const eLP = paths.lp[paths.lp.length - 1], eNN = showNN ? paths.nn[paths.nn.length - 1] : eLP;
  $("wsLP").textContent = paths.lp.length - 1; $("wsNN").textContent = showNN ? paths.nn.length - 1 : "";
  $("weLP").textContent = `(${A[eLP[1]].toFixed(2)}, ${L[eLP[0]].toFixed(2)})`; $("weNN").textContent = `(${A[eNN[1]].toFixed(2)}, ${L[eNN[0]].toFixed(2)})`;
  const oLP = obj(F, eLP[0], eLP[1], lam), oNN = obj(F, eNN[0], eNN[1], lam);
  $("woLP").textContent = oLP.toFixed(3); $("woNN").textContent = oNN.toFixed(3); $("wbest").textContent = `${best.toFixed(3)} at (${A[bj].toFixed(2)}, ${L[bi].toFixed(2)})`;
  $("jumps").style.display = "inline";
  const lpStalled = paths.lp.length - 1 < 3;
  if (!showNN) { $("verdict").textContent = lpStalled
    ? `The walker stopped after ${paths.lp.length - 1} step(s). On a plateau every finite difference of capacity is zero, so the only slope it feels is "remove less wood"; it drifts toward the rot until the first cliff, where it stops. Its objective is ${oLP.toFixed(3)}; the best cell on the grid is ${best.toFixed(3)}.`
    : `The walker took ${paths.lp.length - 1} steps to an objective of ${oLP.toFixed(3)}; the best cell on the grid is ${best.toFixed(3)}.`; render(); return; }
  $("verdict").textContent = lpStalled
    ? `The LP-slope walker stopped after ${paths.lp.length - 1} step(s): on a plateau every finite difference of capacity is zero, so the only slope it feels is "remove less wood", and the first cliff stops it. The surrogate-slope walker took ${paths.nn.length - 1} steps and reached an objective of ${oNN.toFixed(3)} (exact solver), against ${oLP.toFixed(3)}; the best cell on the grid is ${best.toFixed(3)}.`
    : `LP-slope walker: ${paths.lp.length - 1} steps to objective ${oLP.toFixed(3)}. Surrogate-slope walker: ${paths.nn.length - 1} steps to ${oNN.toFixed(3)}. Best cell on the grid: ${best.toFixed(3)}. Both end points are scored by the exact solver.`;
  render();
};
$("reset").onclick = () => { start = null; paths = {lp: null, nn: null}; $("jumps").style.display = "none"; ["wsLP", "wsNN", "weLP", "weNN", "woLP", "woNN", "wbest"].forEach(id => $(id).textContent = ""); $("verdict").textContent = ""; render(); };
$("a").oninput = () => { ia = +$("a").value; render(); }; $("l").oninput = () => { il = +$("l").value; render(); };
$("lam").oninput = render; $("show").onchange = render;
$("jumpLP").onclick = () => { if (!paths.lp) return; const e = paths.lp[paths.lp.length - 1]; il = e[0]; ia = e[1]; $("a").value = ia; $("l").value = il; render(); };
$("jumpNN").onclick = () => { if (!paths.nn) return; const e = paths.nn[paths.nn.length - 1]; il = e[0]; ia = e[1]; $("a").value = ia; $("l").value = il; render(); };
$("showf").onchange = render;
$("shownn").onchange = () => { document.body.classList.toggle("nonn", !$("shownn").checked); $("reset").onclick(); };
sel.onchange = () => { setFamily(fsel.value, +sel.value); render(); };
fsel.onchange = () => { setFamily(fsel.value, +sel.value); render(); };
render();
</script></body></html>"""


if __name__ == "__main__":
    main()
