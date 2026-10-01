"""Evaluate trained surrogates on held-out fields, with the staircase figure.

    python -m repair.proto2d.evaluate                      # every family with a model -> models/report.html
    python -m repair.proto2d.evaluate families=[tenon] sweep.n_fields=8

For each family: test metrics (RMSE, R^2, dead/live accuracy), then for a few
held-out fields a 1D sweep of one parameter with the LP's capacity (the
staircase) and the surrogate's prediction, drawn as inline SVG next to a
picture of the field with the base interface. Config: configs/evaluate.yaml.
"""
from __future__ import annotations

import html
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from . import dataset as ds
from . import draw as dw
from . import families as fm
from . import model as sp
from .train import FamilyData, evaluate as metrics_on, load_model


def predict(model, ck, raster_t, params_dict_list, device):
    """Surrogate capacity, in capacity units, for parameter dicts on one raster."""
    n = ck["normalization"]
    lo, hi = np.array(n["lo"], np.float32), np.array(n["hi"], np.float32)
    P = np.array([[p[k] for k in n["param_names"]] for p in params_dict_list], np.float32)
    P = torch.tensor((P - lo) / (hi - lo), device=device)
    with torch.no_grad():
        code = model.encode(raster_t.unsqueeze(0))
        out = model.from_code(code.expand(len(P), -1), P)
    return out.cpu().numpy() * n["peak"]


def svg_sweep(x, lp, nn_, name, width=520, height=200):
    """Inline SVG: the LP staircase (black) and the surrogate (orange)."""
    lo, hi = float(x[0]), float(x[-1]); top = max(float(lp.max()), float(nn_.max()), 1e-6) * 1.08
    ml, mr, mt, mb = 44, 10, 10, 28
    sx = lambda v: ml + (v - lo) / (hi - lo) * (width - ml - mr)
    sy = lambda v: mt + (1 - v / top) * (height - mt - mb)
    steps = " ".join(f"{sx(x[i]):.1f},{sy(lp[i]):.1f} {sx(x[i + 1]):.1f},{sy(lp[i]):.1f}" for i in range(len(x) - 1))
    curve = " ".join(f"{sx(x[i]):.1f},{sy(nn_[i]):.1f}" for i in range(len(x)))
    ticks = "".join(f'<text x="{sx(v):.1f}" y="{height - 8}" font-size="10" text-anchor="middle">{v:.1f}</text>'
                    for v in np.linspace(lo, hi, 5))
    yt = "".join(f'<text x="{ml - 4}" y="{sy(v) + 3:.1f}" font-size="10" text-anchor="end">{v:.0f}</text>'
                 for v in np.linspace(0, top / 1.08, 4))
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px">'
            f'<rect x="{ml}" y="{mt}" width="{width - ml - mr}" height="{height - mt - mb}" fill="none" stroke="#ccc"/>'
            f'<polyline points="{steps}" fill="none" stroke="#111" stroke-width="1.5"/>'
            f'<polyline points="{curve}" fill="none" stroke="#e8701a" stroke-width="2"/>'
            f'{ticks}{yt}<text x="{(ml + width - mr) / 2:.0f}" y="{height - 18}" font-size="11" text-anchor="middle">{html.escape(name)}</text>'
            f'<text x="{width - mr}" y="{mt + 12}" font-size="11" text-anchor="end"><tspan fill="#111">LP</tspan>  <tspan fill="#e8701a">surrogate</tspan></text></svg>')


@hydra.main(version_base=None, config_path="../../../configs", config_name="evaluate")
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    models_dir = Path(cfg.models_dir)
    keys = list(cfg.families) if cfg.families else sorted(p.parent.name for p in models_dir.glob("*/model.pt"))
    fields, fams = ds.load(cfg.data_dir)
    rng = np.random.default_rng(cfg.seed)
    b, grid, st = sp.Block(), sp.Grid(), sp.Statics()
    nav = "".join(f'<a href="#{k}">{k}</a>' for k in keys)
    rows, sections = [], []
    for k in keys:
        model, ck = load_model(models_dir / k / "model.pt", device)
        data = FamilyData(cfg.data_dir, k, device)
        m = metrics_on(model, data, data.test_idx)
        rows.append(f"<tr><td>{k}</td><td>{len(data.train_idx)}</td><td>{len(data.test_idx)}</td><td>{m['rmse_capacity']:.2f}</td>"
                    f"<td>{100 * m['rmse']:.1f}%</td><td>{m['r2']:.3f}</td><td>{100 * m['zero_acc']:.1f}%</td><td>{ck['val']['epoch']}</td></tr>")
        fam = fm.FAMILIES[k]; r = fams[k]
        sweep_name = cfg.sweep.param or "a"
        q = next(q for q in fam.params if q.name == sweep_name)
        test_fields = np.unique(r["field"][r["test"]])
        sections.append(f'<h2 id="{k}">{k}: {html.escape(fam.title)}</h2>')
        for fi in rng.choice(test_fields, size=min(cfg.sweep.n_fields, len(test_fields)), replace=False):
            fi = int(fi); f = ds.field_of(fields, fi, b)
            raster_t = data.raster[fi]
            base_row = int(rng.choice(np.nonzero(r["field"] == fi)[0]))
            base = fam.from_vector(r["params"][base_row])
            xs = np.linspace(q.lo, q.hi, cfg.sweep.n_points)
            pts = [fam.feasible(dict(base, **{q.name: float(x)}), b) for x in xs]
            lp = np.array([fm.evaluate(fam, p, f, st, grid)[0] for p in pts])
            nn_ = np.clip(predict(model, ck, raster_t, pts, device), 0, None)
            M0, R0, cs, _ = fm.evaluate(fam, base, f, st, grid)
            png = dw.png_base64(dw.draw(fam, base, f, cs, b))
            held = " ".join(f"{n} {v:.2f}" for n, v in base.items() if n != q.name)
            err = np.sqrt(np.mean((lp - nn_) ** 2))
            sections.append(f'<div class="ex"><div><img src="data:image/png;base64,{png}">'
                            f"<div class=cap>held-out field {fi}; base interface drawn ({q.name} {base[q.name]:.2f}: M {M0:.1f}); "
                            f"swept {q.name} over [{q.lo}, {q.hi}] with {held}; sweep RMSE {err:.1f}</div></div>"
                            f"{svg_sweep(xs, lp, nn_, f'{q.name}  ({q.doc})')}</div>")
    table = ("<table><tr><th>family</th><th>train rows</th><th>test rows</th><th>RMSE (capacity)</th><th>RMSE (% of peak)</th>"
             "<th>R²</th><th>dead/live acc</th><th>best epoch</th></tr>" + "".join(rows) + "</table>")
    css = ("body{font-family:sans-serif;max-width:1200px;margin:auto;padding:16px;color:#333} "
           "nav{position:sticky;top:0;z-index:1;background:#fff;border-bottom:1px solid #ddd;padding:8px 0;margin:-16px -16px 12px;padding-left:16px} "
           "nav a{display:inline-block;margin:2px 10px 2px 0;padding:3px 8px;border:1px solid #bbb;border-radius:4px;color:#333;text-decoration:none;font-size:13px} "
           "h2{scroll-margin-top:60px} table{border-collapse:collapse;margin:12px 0} td,th{border:1px solid #999;padding:3px 8px;text-align:right} td:first-child{text-align:left} "
           ".ex{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:center;margin:0 0 22px} img{width:100%} .cap{font-size:12px;margin-top:4px} "
           "@media(max-width:800px){.ex{grid-template-columns:1fr}}")
    out = Path(cfg.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
                   f"<title>surrogate evaluation</title><style>{css}</style></head><body><nav>{nav}</nav>"
                   f"<p>Surrogates on held-out fields. Metrics on all test rows; below, per family, one parameter swept on a few held-out fields: "
                   f"the LP's capacity (black staircase) against the surrogate (orange).</p>{table}{''.join(sections)}</body></html>")
    print(out)


if __name__ == "__main__":
    main()
