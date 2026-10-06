"""A gallery page: example damage fields, and separately the 2D joint families.

    python examples/proto2d/gallery.py                    # -> out/proto2d/gallery.html
    python examples/proto2d/gallery.py --fields 36 --variations 4 --seed 1

Section 1 draws random damage fields on the whole block, with no joint: pale
is sound wood, red is rot, the black line is the threshold the solver sees.
Section 2 draws every joint family in sound wood, so the shape reads cleanly:
its defaults first, then a few random feasible variations. Old wood is tan,
the new piece pale, the interface black. A checkbox in the top bar shows or
hides the parameters under every picture. One self-contained HTML file, images embedded as base64.
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import numpy as np

try:
    from repair.proto2d import damage as dm, dataset as ds, draw as dw, families as fm, model as sp
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from repair.proto2d import damage as dm, dataset as ds, draw as dw, families as fm, model as sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/proto2d/gallery.html")
    ap.add_argument("--fields", type=int, default=24, help="damage fields to show")
    ap.add_argument("--variations", type=int, default=3, help="random variations per family, after the defaults")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    b = sp.Block(); rng = np.random.default_rng(a.seed)

    cells = []
    for i in range(a.fields):
        k = dm.sample_knobs(rng); f = dm.random_field(int(rng.integers(0, 2**31 - 1)), k, b)
        cap = f"reach {k.reach:.1f} &middot; end {k.end_frac:.2f} &middot; aniso {k.aniso:.2f} &middot; " + (f"pocket {k.patch:.2f}" if k.patch > 0 else "no pocket")
        cells.append(f'<div class="c"><img src="data:image/png;base64,{dw.png_base64(dw.draw_field(f, b))}"><div class="cap">{cap}</div></div>')
    damage = f'<div class="grid g3">{"".join(cells)}</div>'

    sound = dm.FieldDamage(np.zeros((40, 240)), b)            # no rot: the joint shapes only
    fam_nav, fam_html = [], []
    for key, fam in fm.FAMILIES.items():
        fam_nav.append(f'<a href="#{key}">{key}</a>')
        ps = [fam.defaults()] + [ds.sample_params(fam, rng, b) for _ in range(a.variations)]
        cells = []
        for j, p in enumerate(ps):
            p = fam.feasible(p, b)
            cap = ("<b>defaults</b> &middot; " if j == 0 else "") + " &middot; ".join(f"{n} {v:.2f}" for n, v in p.items())
            cells.append(f'<div class="c"><img src="data:image/png;base64,{dw.png_base64(dw.draw(fam, p, sound, None, b))}"><div class="cap">{cap}</div></div>')
        src = f" &middot; from MiGumi {fam.source}" if fam.source else ""
        plist = "".join(f"<li><b>{q.name}</b>: {html.escape(q.doc)} [{q.lo:g}, {q.hi:g}]</li>" for q in fam.params)
        fam_html.append(f'<h3 id="{key}">{key}</h3><p>{html.escape(fam.title)}{src}</p><ul class="cap">{plist}</ul>'
                        f'<div class="grid g2">{"".join(cells)}</div>')

    css = ("body{font-family:sans-serif;max-width:1400px;margin:auto;padding:16px;color:#333} img{width:100%;display:block} "
           "nav{position:sticky;top:0;z-index:1;background:#fff;border-bottom:1px solid #ddd;margin:-16px -16px 12px;padding:8px 16px} "
           "nav a{display:inline-block;margin:2px 8px 2px 0;padding:3px 8px;border:1px solid #bbb;border-radius:4px;color:#333;text-decoration:none;font-size:13px} "
           "h2,h3{scroll-margin-top:60px} h2{margin-top:30px} ul{margin:4px 0 10px;font-size:13px} "
           ".grid{display:grid;gap:14px} .g3{grid-template-columns:repeat(3,1fr)} .g2{grid-template-columns:repeat(2,1fr)} .c{font-size:12px} "
           "body.nocap .cap{display:none} nav label{font-size:13px;margin-left:8px} "
           "@media(max-width:800px){.g3,.g2{grid-template-columns:1fr}}")
    page = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            f"<title>damage fields and joint families</title><style>{css}</style></head><body>"
            f'<nav><a href="#damage">damage fields</a><a href="#families">joint families</a>{"".join(fam_nav)}'
            '<label><input type="checkbox" id="showcap" checked onchange="document.body.classList.toggle(&quot;nocap&quot;, !this.checked)"> show parameters</label></nav>'
            f'<h2 id="damage">Damage fields</h2><p>{a.fields} random fields on the whole block. Rot enters from the right end and, in most fields, '
            f"from a pocket on one face. Pale = sound, red = rot, black line = the threshold the solver sees.</p>{damage}"
            f'<h2 id="families">Joint families</h2><p>Each family in sound wood: tan = old wood, pale = the new piece, black = the interface. '
            f"Defaults first, then {a.variations} random variations. Shown from x = 4 to the end of the block.</p>"
            f'{"".join(fam_html)}</body></html>')
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(page)
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
