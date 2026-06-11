"""09 - End-to-end inference: prior picks the template, optimizer fits it.

On a fresh damage instance, compare the prior+fit repair against the (slow) oracle that
tries every template. When they agree, the prior matched the oracle while fitting only
one template.

    python examples/09_inference.py [--show]
"""
import sys

import numpy as np

from repair import dataset as ds, viz
from repair.charts import tile
from repair.config import ensure_out
from repair.damage import sample_damage
from repair.inference import plan_repair
from repair.optimizer import oracle
from repair.prior import TorchCNNPrior
from repair.rasterize import rasterize
from repair.templates import TEMPLATES

SHOW = "--show" in sys.argv
out = ensure_out()


def main():
    if not (out / "prior.pt").exists():
        print("  run examples/08_train_prior.py first (need out/prior.pt)")
        return
    prior = TorchCNNPrior.load(out / "prior.pt")

    key = "RJ_KA"   # a joint; works whether or not it was in training
    m = ds.load_member(key)
    mate = ds.load_mate_interface(ds.load_jwood(key), m)
    dmg = sample_damage(m, np.random.default_rng(21), kind="central_rot")

    raster = rasterize(m, dmg, mate)
    proba = prior.predict_proba(raster)
    pred = prior.predict_template(raster)
    print("  prior proba:", {n: round(float(p), 2)
                             for n, p in zip(prior.classes, proba)})

    fit = plan_repair(m, dmg, mate, prior, raster=raster, rng=np.random.default_rng(0))
    cut_pred = TEMPLATES[fit.template].apply(m, dmg, fit.theta)

    oname, ofit, _ = oracle(m, dmg, mate, rng=np.random.default_rng(0))
    cut_oracle = TEMPLATES[oname].apply(m, dmg, ofit.theta)
    print(f"  prior -> {fit.template} (E={fit.total:.3f}) | "
          f"oracle -> {oname} (E={ofit.total:.3f}) | "
          f"{'MATCH' if fit.template == oname else 'differ'}")

    p1 = viz.render_repair(m, dmg, cut_pred, mate, out / "09_prior.png",
                           title=f"prior: {fit.template}")
    p2 = viz.render_repair(m, dmg, cut_oracle, mate, out / "09_oracle.png",
                           title=f"oracle: {oname}")
    tile([p1, p2], [f"prior+fit: {fit.template} (E={fit.total:.2f})",
                    f"oracle: {oname} (E={ofit.total:.2f})"],
         out / "09_inference.png", cols=2,
         suptitle=f"End-to-end repair on {key} (regret={fit.total - ofit.total:.3f})")
    print("  -> out/09_inference.png")

    if SHOW:
        viz.render_repair(m, dmg, cut_pred, mate, out / "09_prior.png",
                          title=f"prior: {fit.template}")
        viz.show()


if __name__ == "__main__":
    main()
