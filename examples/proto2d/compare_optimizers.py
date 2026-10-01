"""Step 4a: the comparison table. Four routes to the optimum on held-out fields.

    python examples/proto2d/compare_optimizers.py
    python examples/proto2d/compare_optimizers.py n_fields=10 families=[tenon,dovetail] starts=4

For every (held-out field, family) a reference optimum is found on the hard LP
(random search + Nelder-Mead). Then from the same random starts, three
gradient routes climb the objective: through the surrogate, on the LP with
softened contact flags, and with finite differences on the hard LP. Every end
point is re-scored by the hard LP. Writes compare.json and compare.html to
out/proto2d/optimize/. Config: configs/optimize.yaml.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

try:
    from repair.proto2d import dataset as ds, families as fm, model as sp, optimize as opt
    from repair.proto2d.train import load_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from repair.proto2d import dataset as ds, families as fm, model as sp, optimize as opt
    from repair.proto2d.train import load_model

_CACHE = {}


def job(args):
    """One (field, family): reference plus every route from every start. Runs in a worker on CPU."""
    cfg, fi, key = args
    torch.set_num_threads(1)
    if "fields" not in _CACHE:
        _CACHE["fields"], _ = ds.load(cfg["data_dir"])
    if key not in _CACHE:
        _CACHE[key] = load_model(Path(cfg["models_dir"]) / key / "model.pt", "cpu")
    fields = _CACHE["fields"]; model, ck = _CACHE[key]
    b = sp.Block(); f = ds.field_of(fields, fi, b)
    raster = torch.tensor(fields["raster"][fi].astype(np.float32)).unsqueeze(0)
    obj = opt.Objective(lam=cfg["objective"]["lam"], m_req=cfg["objective"]["m_req"], pen=cfg["objective"]["pen"],
                        m_ref=ck["normalization"]["peak"], r_ref=cfg["objective"]["r_ref"])
    prob = opt.Problem(fm.FAMILIES[key], f, obj, model, ck, raster, "cpu", tau=cfg["tau"])
    rng = np.random.default_rng([cfg["seed"], fi, abs(hash(key)) % 10**6])
    ref = opt.reference(prob, cfg["reference"]["n_random"], cfg["reference"]["n_refine"], seed=int(rng.integers(1 << 30)))
    rows = []
    for s in range(cfg["starts"]):
        # the start: the best of a few random LP probes, so no route begins on the
        # dead plateau; the probes are charged to every route's LP count equally
        cands = [prob.project(rng.random(len(prob.names))) for _ in range(max(1, cfg["start_probes"]))]
        x0 = max(cands, key=lambda x: obj(*prob.hard(x)))
        for route in cfg["routes"]:
            r = opt.ascend(prob, route, x0, cfg["ascent"]["steps"], cfg["ascent"]["lr"], cfg["ascent"]["h"])
            rows.append(dict(field=fi, family=key, start=s, route=route, J_true=r.J_true, J_est=r.J_est, M=r.M_true, R=r.R_true,
                             lp_calls=r.lp_calls + max(1, cfg["start_probes"]), steps=r.steps, on_bound=r.on_bound, x0=r.x0.tolist(), x=r.x.tolist(),
                             J_ref=ref.J_true, J_start=float(obj(*prob.hard(x0)))))
    rows.append(dict(field=fi, family=key, start=-1, route="reference", J_true=ref.J_true, J_est=ref.J_true, M=ref.M_true,
                     R=ref.R_true, lp_calls=ref.lp_calls, steps=0, on_bound=ref.on_bound, x0=ref.x.tolist(), x=ref.x.tolist(),
                     J_ref=ref.J_true, J_start=ref.J_true))
    return rows


def table(rows, routes):
    """Per route: mean true objective, gap to the reference, how often it ends dead, LP calls."""
    out = []
    ref_calls = np.mean([r["lp_calls"] for r in rows if r["route"] == "reference"])
    for route in routes + ["reference"]:
        rr = [r for r in rows if r["route"] == route]
        if not rr:
            continue
        J = np.array([r["J_true"] for r in rr]); gap = np.array([r["J_ref"] - r["J_true"] for r in rr])
        gain = np.array([r["J_true"] - r["J_start"] for r in rr])
        out.append(dict(route=route, runs=len(rr), J_mean=float(J.mean()), gap_mean=float(gap.mean()), gap_median=float(np.median(gap)),
                        within_005=float((gap <= 0.05).mean()), gain_mean=float(gain.mean()),
                        dead_end=float(np.mean([r["M"] <= 1e-6 for r in rr])),
                        est_err=float(np.mean([abs(r["J_est"] - r["J_true"]) for r in rr])),
                        lp_calls=float(np.mean([r["lp_calls"] for r in rr])), steps=float(np.mean([r["steps"] for r in rr])),
                        on_bound=float(np.mean([len(r["on_bound"]) > 0 for r in rr]))))
    return out


def html_report(cfg, rows, routes, keys, picked, seconds):
    def tbl(stats, title):
        h = (f"<h2>{title}</h2><table><tr><th>route</th><th>runs</th><th>true objective, mean</th><th>gap to reference, mean</th>"
             "<th>gap, median</th><th>within 0.05 of reference</th><th>gain over start</th><th>ends with zero capacity</th>"
             "<th>|own estimate − true|</th><th>LP calls per run</th><th>steps</th><th>any parameter on a bound</th></tr>")
        for s in stats:
            h += (f"<tr><td>{s['route']}</td><td>{s['runs']}</td><td>{s['J_mean']:.3f}</td><td>{s['gap_mean']:.3f}</td><td>{s['gap_median']:.3f}</td>"
                  f"<td>{100 * s['within_005']:.0f}%</td><td>{s['gain_mean']:+.3f}</td><td>{100 * s['dead_end']:.0f}%</td><td>{s['est_err']:.3f}</td>"
                  f"<td>{s['lp_calls']:.0f}</td><td>{s['steps']:.0f}</td><td>{100 * s['on_bound']:.0f}%</td></tr>")
        return h + "</table>"
    parts = [f"<h1>Routes to the optimum, held-out fields</h1>",
             f"<p>{len(picked)} held-out fields ({', '.join(map(str, picked))}), {len(keys)} families, {cfg.starts} starts each; "
             f"starts are the best of {cfg.start_probes} random LP probes; objective = M / M_ref &minus; {cfg.objective.lam} &middot; R / {cfg.objective.r_ref}"
             + (f" &minus; {cfg.objective.pen} &middot; shortfall below {cfg.objective.m_req}" if cfg.objective.m_req > 0 else "")
             + f"; ascent {cfg.ascent.steps} steps, lr {cfg.ascent.lr}, h {cfg.ascent.h}; tau {cfg.tau}; {seconds:.0f} s.</p>"
             "<p>Every route starts from the same random points and every end point is scored by the hard LP. "
             "The reference is random search plus Nelder-Mead on the hard LP, the best this budget finds. "
             "<b>Gap</b> is reference minus the route's true objective (smaller is better). "
             "<b>|own estimate − true|</b> is how far the route's own view of its end point is from the LP's.</p>",
             tbl(table(rows, routes), "All families")]
    for k in keys:
        parts.append(tbl(table([r for r in rows if r["family"] == k], routes), k))
    css = ("body{font-family:sans-serif;max-width:1300px;margin:auto;padding:16px;color:#333} table{border-collapse:collapse;margin:8px 0;font-size:13px} "
           "td,th{border:1px solid #999;padding:3px 8px;text-align:right} td:first-child{text-align:left} h2{margin-top:24px;font-size:16px}")
    return f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>optimizer comparison</title><style>{css}</style></head><body>{''.join(parts)}</body></html>"


@hydra.main(version_base=None, config_path="../../configs", config_name="optimize")
def main(cfg: DictConfig):
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(cfg.families) if cfg.families else sorted(p.parent.name for p in Path(cfg.models_dir).glob("*/model.pt"))
    fields, fams = ds.load(cfg.data_dir)
    rng = np.random.default_rng(cfg.seed)
    test_fields = np.unique(fams[keys[0]]["field"][fams[keys[0]]["test"]])
    picked = [int(x) for x in rng.choice(test_fields, size=cfg.n_fields, replace=False)]
    plain = OmegaConf.to_container(cfg, resolve=True)
    jobs = [(plain, fi, k) for fi in picked for k in keys]
    routes = list(cfg.routes)
    print(f"{len(jobs)} (field, family) jobs x {cfg.starts} starts x {len(routes)} routes + reference")
    t0 = time.time(); rows = []
    workers = cfg.workers or max(1, mp.cpu_count() - 1)
    with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
        for n, rr in enumerate(pool.imap_unordered(job, jobs), 1):
            rows += rr
            print(f"  {n}/{len(jobs)} done ({time.time() - t0:.0f} s)")
    seconds = time.time() - t0
    (out_dir / "compare.json").write_text(json.dumps(dict(config=plain, fields=picked, rows=rows), indent=1))
    (out_dir / "compare.html").write_text(html_report(cfg, rows, routes, keys, picked, seconds))
    for s in table(rows, routes):
        print(f"  {s['route']:10s} J {s['J_mean']:.3f}  gap {s['gap_mean']:.3f} (median {s['gap_median']:.3f})  within .05 {100 * s['within_005']:3.0f}%  "
              f"dead {100 * s['dead_end']:3.0f}%  |est-true| {s['est_err']:.3f}  LP calls {s['lp_calls']:6.0f}  steps {s['steps']:3.0f}")
    print(out_dir / "compare.html")


if __name__ == "__main__":
    main()
