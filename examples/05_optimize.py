"""05 - Fit one template's parameters by minimizing the proxy energy.

Shows the classical per-template optimizer: default theta (before) vs the energy-minimizing
theta (after), and the energy drop.

    python examples/05_optimize.py
"""
import numpy as np

from repair import dataset as ds, viz
from repair.charts import bar_energy_terms, tile
from repair.config import ensure_out
from repair.damage import sample_damage
from repair.energy import energy
from repair.optimizer import fit_template
from repair.templates import TEMPLATES

out = ensure_out()


def main():
    key = "CJ_DT"
    m = ds.load_member(key)
    mate = ds.load_mate_interface(ds.load_jwood(key), m)
    dmg = sample_damage(m, np.random.default_rng(11), kind="ground_contact")
    t = TEMPLATES["ledge"]

    theta0 = t.default_theta(m, dmg)
    cut0 = t.apply(m, dmg, theta0)
    e0 = energy(m, dmg, cut0, mate)

    fit = fit_template(m, dmg, t, mate, rng=np.random.default_rng(0))
    cut1 = t.apply(m, dmg, fit.theta)
    e1 = fit.energy

    print(f"  before: theta={np.round(theta0, 2).tolist()} total={e0.total:.3f}")
    print(f"  after : theta={np.round(fit.theta, 2).tolist()} total={e1.total:.3f}")

    p0 = viz.render_repair(m, dmg, cut0, mate, out / "05_before.png", title="default theta")
    p1 = viz.render_repair(m, dmg, cut1, mate, out / "05_after.png", title="optimized theta")
    tile([p0, p1], [f"before (E={e0.total:.2f})", f"after (E={e1.total:.2f})"],
         out / "05_optimize.png", cols=2, suptitle="Per-template parameter fit (ledge)")
    bar_energy_terms([e0, e1], ["before", "after"], out / "05_energy.png",
                     title="Energy terms before vs after optimization")
    print("  -> out/05_optimize.png, out/05_energy.png")


if __name__ == "__main__":
    main()
