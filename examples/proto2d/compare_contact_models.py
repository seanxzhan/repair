"""Side by side: the cap model (model.py) against the pressure-bounded model
(pressure.py), swept on held-out damage fields.

    python examples/proto2d/compare_contact_models.py              # -> out/proto2d/contact_models.html
    python examples/proto2d/compare_contact_models.py --fields 3 --families tenon dovetail_flip

For each family and field: a picture of the field with the base interface,
then one parameter swept with three curves, the cap model (black staircase),
the pressure model at 7 samples (orange) and at 21 samples (dashed blue, to
show the sample count no longer matters). Each curve is normalized by its
own peak so the shapes compare. The caption counts the steps larger than 3%
of peak in each curve, and a zoom test on the largest step that tells a jump from a steep slope.
The sweep covers only the range where the shoulder moves without feasibility
clipping altering another parameter.
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import numpy as np

try:
    from repair.proto2d import dataset as ds, draw as dw, families as fm, model as sp, pressure as pr
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from repair.proto2d import dataset as ds, draw as dw, families as fm, model as sp, pressure as pr


def svg(x, series, name, width=560, height=210):
    """series: list of (values, colour, dash, label), each normalized to its own peak."""
    lo, hi = float(x[0]), float(x[-1]); ml, mr, mt, mb = 40, 10, 14, 30
    sx = lambda v: ml + (v - lo) / (hi - lo) * (width - ml - mr)
    sy = lambda v: mt + (1 - v / 1.05) * (height - mt - mb)
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px">'
           f'<rect x="{ml}" y="{mt}" width="{width - ml - mr}" height="{height - mt - mb}" fill="none" stroke="#ccc"/>']
    for y, col, dash, _ in series:
        pts = " ".join(f"{sx(x[i]):.1f},{sy(y[i]):.1f}" for i in range(len(x)))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" stroke-dasharray="{dash}"/>')
    out.append("".join(f'<text x="{sx(v):.1f}" y="{height - 10}" font-size="10" text-anchor="middle">{v:.1f}</text>' for v in np.linspace(lo, hi, 5)))
    out.append("".join(f'<text x="{ml - 4}" y="{sy(v) + 3:.1f}" font-size="10" text-anchor="end">{v:.1f}</text>' for v in (0, 0.5, 1)))
    out.append(f'<text x="{(ml + width - mr) / 2:.0f}" y="{height - 20}" font-size="11" text-anchor="middle">{html.escape(name)}</text>')
    legend = "  ".join(f'<tspan fill="{col}">{lab}</tspan>' for _, col, _, lab in series)
    out.append(f'<text x="{width - mr}" y="{mt + 11}" font-size="11" text-anchor="end">{legend}</text></svg>')
    return "".join(out)


def biggest_step(y):
    """Largest single step between neighbouring sweep points, as a fraction of the peak."""
    y = np.asarray(y); return float(np.abs(np.diff(y)).max() / max(y.max(), 1e-9))


def jump_ratio(xs, y, g, k=8):
    """Zoom into the largest step: evaluate g at k points inside it and return the
    largest sub-step over the whole step. A continuous slope splits into pieces
    (about 1/k when linear, below 1 in any case); a true jump keeps one sub-step
    equal to the whole step (about 1)."""
    y = np.asarray(y); j = int(np.argmax(np.abs(np.diff(y)))); full = abs(y[j + 1] - y[j])
    if full < 1e-9:
        return 0.0
    inner = np.linspace(xs[j], xs[j + 1], k + 2)
    z = np.array([y[j]] + [g(float(x)) for x in inner[1:-1]] + [y[j + 1]])
    return float(np.abs(np.diff(z)).max() / full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/proto2d")
    ap.add_argument("--out", default="out/proto2d/contact_models.html")
    ap.add_argument("--families", nargs="*", default=["tenon", "tenon_flip", "dovetail", "hooked_scarf"])
    ap.add_argument("--fields", type=int, default=3)
    ap.add_argument("--points", type=int, default=121)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    fields, fams = ds.load(a.data)
    b, grid = sp.Block(), sp.Grid(); st_cap, st_pr = sp.Statics(), pr.PressureStatics()
    rng = np.random.default_rng(a.seed)
    sections, totals = [], {"cap": 0.0, "pr7": 0.0, "pr21": 0.0, "rcap": 0.0, "rpr": 0.0, "n": 0}
    for key in a.families:
        fam, r = fm.FAMILIES[key], fams[key]
        q = next(q for q in fam.params if q.name == "a")
        test_fields = np.unique(r["field"][r["test"]])
        sections.append(f'<h2 id="{key}">{key}: {html.escape(fam.title)}</h2>')
        for fi in rng.choice(test_fields, size=min(a.fields, len(test_fields)), replace=False):
            fi = int(fi); f = ds.field_of(fields, fi, b)
            base = fam.from_vector(r["params"][int(rng.choice(np.nonzero(r["field"] == fi)[0]))])
            # sweep only where the shoulder moves freely: points where feasibility
            # clipping would alter another parameter (a face merging into the block's
            # edge, a tongue clipped against the end) are parametrization-bound
            # effects, not contact-model ones, and are left out
            free = [x for x in np.linspace(q.lo, q.hi, a.points)
                    if all(abs(fam.feasible(dict(base, a=float(x)), b)[k] - v) < 1e-9 for k, v in dict(base, a=float(x)).items())]
            xs = np.array(free)
            pts = [dict(base, a=float(x)) for x in xs]
            cap = np.array([fm.evaluate(fam, p, f, st_cap, grid)[0] for p in pts])
            p7 = np.array([pr.evaluate(fam, p, f, st_pr, grid, 7)[0] for p in pts])
            p21 = np.array([pr.evaluate(fam, p, f, st_pr, grid, 21)[0] for p in pts])
            norm = lambda y: y / max(y.max(), 1e-9)
            s_cap, s7, s21 = biggest_step(cap), biggest_step(p7), biggest_step(p21)
            g_cap = lambda x: fm.evaluate(fam, dict(base, a=x), f, st_cap, grid)[0]
            g_pr = lambda x: pr.evaluate(fam, dict(base, a=x), f, st_pr, grid, 7)[0]
            r_cap, r_pr = jump_ratio(xs, cap, g_cap), jump_ratio(xs, p7, g_pr)
            totals["cap"] += s_cap; totals["pr7"] += s7; totals["pr21"] += s21; totals["rcap"] += r_cap; totals["rpr"] += r_pr; totals["n"] += 1
            png = dw.png_base64(dw.draw(fam, base, f, None, b))
            held = " ".join(f"{n} {v:.2f}" for n, v in base.items() if n != "a")
            sections.append(
                f'<div class="ex"><div><img src="data:image/png;base64,{png}"><div class="cap">held-out field {fi}; {held}. '
                f"Largest single step, % of peak: cap model {100 * s_cap:.0f}%, pressure model {100 * s7:.0f}% (7 samples), {100 * s21:.0f}% (21 samples). "
                f"Zoom test on the largest step (1 = a true jump, small = a steep slope): cap {r_cap:.2f}, pressure {r_pr:.2f}. "
                f"Peaks: {cap.max():.1f} / {p7.max():.1f} / {p21.max():.1f}; max |7 − 21| = {np.abs(p7 - p21).max():.2f}.</div></div>"
                + svg(xs, [(norm(cap), "#111", "", "cap model"), (norm(p7), "#e8701a", "", "pressure, 7"), (norm(p21), "#2b6cb0", "6,4", "pressure, 21")],
                      "shoulder position a; each curve normalized by its own peak") + "</div>")
    nav = "".join(f'<a href="#{k}">{k}</a>' for k in a.families)
    css = ("body{font-family:sans-serif;max-width:1200px;margin:auto;padding:16px;color:#333} img{width:100%} "
           "nav{position:sticky;top:0;z-index:1;background:#fff;border-bottom:1px solid #ddd;margin:-16px -16px 12px;padding:8px 16px} "
           "nav a{display:inline-block;margin:2px 10px 2px 0;padding:3px 8px;border:1px solid #bbb;border-radius:4px;color:#333;text-decoration:none;font-size:13px} "
           "h2{scroll-margin-top:60px} .ex{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:center;margin:0 0 22px} .cap{font-size:12px;margin-top:4px} "
           "@media(max-width:800px){.ex{grid-template-columns:1fr}}")
    n = totals["n"]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
                   f"<title>contact models</title><style>{css}</style></head><body><nav>{nav}</nav>"
                   f"<p><b>Cap model</b> (black): a global cap on the sum of normal forces, each sample live or dead at one probe point. "
                   f"<b>Pressure model</b> (orange, dashed blue): each sample's normal force bounded by a bearing pressure times the sound length of its own segment, no cap, no flag. "
                   f"Over {n} sweeps, the largest single step as a share of the peak averages {100 * totals['cap'] / n:.0f}% for the cap model against "
                   f"{100 * totals['pr7'] / n:.0f}% (7 samples) and {100 * totals['pr21'] / n:.0f}% (21 samples) for the pressure model. "
                   f"Zooming into each sweep's largest step with 8 more evaluations: a true jump survives as one sub-step equal to the whole (ratio 1), a steep slope splits into smaller pieces. "
                   f"The ratio averages {totals['rcap'] / n:.2f} for the cap model and {totals['rpr'] / n:.2f} for the pressure model.</p>"
                   f"{''.join(sections)}</body></html>")
    print(out)


if __name__ == "__main__":
    main()
