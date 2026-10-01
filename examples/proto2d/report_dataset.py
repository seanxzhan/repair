"""Render a sample of dataset rows to one self-contained HTML file.

    python examples/proto2d/report_dataset.py                          # out/proto2d -> out/proto2d/report.html
    python examples/proto2d/report_dataset.py --n 6 --families tenon dovetail_flip
    python examples/proto2d/report_dataset.py --data out/proto2d_small --out /tmp/report.html --seed 3

Each example is rebuilt from its stored field (seed + knobs) and clipped
parameters, re-solved, and drawn: retained wood coloured by severity, the new
wood showing the damage it replaces faded, the crit contour in black, the interface in black, contact samples as
green (live) or red (dead) dots, with the capacity, sound wood removed and the
parameters underneath. One section per family; each row is one damage field\nwith several of that family's sampled interfaces on it, side by side. PNGs embedded\nas base64.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import sys
from pathlib import Path

import numpy as np
import shapely
from PIL import Image, ImageDraw

try:
    from repair.proto2d import dataset as ds, families as fm, model as sp
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from repair.proto2d import dataset as ds, families as fm, model as sp

S = 60                                   # pixels per block unit
X0 = 4.0                                 # draw x in [X0, length]
WOOD, ROT, NEW = np.array([236, 220, 190]), np.array([120, 20, 20]), np.array([250, 248, 244])


def render(fam, p, f, b: sp.Block, cs, M, R):
    W, H = int((b.length - X0) * S), int(b.height * S)
    xs = X0 + (np.arange(W) + 0.5) / S; ys = b.height - (np.arange(H) + 0.5) / S
    X, Y = np.meshgrid(xs, ys)
    sev = np.clip(f.severity(X, Y), 0, 1)[..., None]
    rgb = (WOOD * (1 - sev) + ROT * sev).astype(np.uint8)
    dead = sev[..., 0] >= f.crit
    edge = np.zeros_like(dead); edge[:, 1:] |= dead[:, 1:] != dead[:, :-1]; edge[1:, :] |= dead[1:, :] != dead[:-1, :]
    # the new wood: the damage it replaces stays visible, faded toward the new-wood tint
    for lay in fam.layers(p, b):
        removed = shapely.contains_xy(lay.removed, X.ravel(), Y.ravel()).reshape(X.shape)
        rgb[removed] = (0.35 * rgb[removed] + 0.65 * NEW).astype(np.uint8)
    rgb[edge] = (0, 0, 0)
    img = Image.fromarray(rgb); dr = ImageDraw.Draw(img)
    to_px = lambda x, y: ((x - X0) * S, (b.height - y) * S)
    for lay in fam.layers(p, b):
        for p0, p1, _ in fm.interface_segments(lay, b):
            dr.line([to_px(*p0), to_px(*p1)], fill=(0, 0, 0), width=2)
    for c in cs:
        x, y = to_px(*c.point); r = 3
        dr.ellipse([x - r, y - r, x + r, y + r], fill=(20, 160, 20) if c.live else (210, 50, 50))
    dr.rectangle([0, 0, W - 1, H - 1], outline=(0, 0, 0))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/proto2d")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=4, help="fields per family; each field shows --per-row of its interfaces")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-only", action="store_true", help="sample only held-out rows")
    ap.add_argument("--per-row", type=int, default=4, help="interfaces shown per damage field")
    a = ap.parse_args()
    out = Path(a.out or Path(a.data) / "report.html")
    fields, fams = ds.load(a.data)
    keys = a.families or list(fams)
    rng = np.random.default_rng(a.seed)
    b, grid, st = sp.Block(), sp.Grid(), sp.Statics()
    nav = "".join(f'<a href="#{k}">{k}</a>' for k in keys)
    parts = [f'<nav>{nav}</nav>',
             f"<p>Each row: one damage field, {a.per_row} of its sampled interfaces from the same family.</p>"]
    for k in keys:
        fam, r = fm.FAMILIES[k], fams[k]
        fields_ok = np.unique(r["field"][r["test"]]) if a.test_only else np.unique(r["field"])
        pick = rng.choice(fields_ok, size=min(a.n, len(fields_ok)), replace=False)
        parts.append(f'<h2 id="{k}">{k}</h2>')
        for fi in pick:
            fi = int(fi); f = ds.field_of(fields, fi, b)
            rows = np.nonzero(r["field"] == fi)[0]
            two = rng.choice(rows, size=min(a.per_row, len(rows)), replace=False)
            cells = []
            for i in two:
                p = fam.from_vector(r["params"][i])
                M, R, cs, _ = fm.evaluate(fam, p, f, st, grid)
                png = render(fam, p, f, b, cs, M, R)
                ps = "  ".join(f"{n} {v:.2f}" for n, v in zip(r["param_names"], r["params"][i]))
                cells.append(f'<div class="ex"><img src="data:image/png;base64,{png}">'
                             f"<div>M {M:.1f}  &middot;  sound removed {R:.2f}  &middot;  live {sum(c.live for c in cs)}/{len(cs)}<br>{ps}</div></div>")
            parts.append(f'<div class="pair">{"".join(cells)}</div>')
    css = ("body{font-family:sans-serif;max-width:1800px;margin:auto;padding:16px;color:#333} img{width:100%} "
           f".pair{{display:grid;grid-template-columns:repeat({a.per_row},1fr);gap:12px;margin:0 0 20px}} .ex{{font-size:11px}} h2{{margin-top:28px}} "
           "nav{position:sticky;top:0;z-index:1;background:#fff;border-bottom:1px solid #ddd;padding:8px 0;margin:-16px -16px 12px;padding-left:16px} "
           "nav a{display:inline-block;margin:2px 10px 2px 0;padding:3px 8px;border:1px solid #bbb;border-radius:4px;color:#333;text-decoration:none;font-size:13px} "
           "nav a:hover{background:#eee} h2{scroll-margin-top:60px} "
           "@media(max-width:900px){.pair{grid-template-columns:1fr 1fr}} @media(max-width:500px){.pair{grid-template-columns:1fr}}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
                   f"<title>proto2d dataset report</title><style>{css}</style></head><body>{''.join(parts)}</body></html>")
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
