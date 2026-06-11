"""06 - The oracle: fit every template, pick the lowest-energy one (= the label).

This is the key correctness gate: the chosen repair should look sensible (removes the
rot, leaves a clean interface, mate untouched). Also renders the energy landscape of one
template over two parameters, with the feasibility boundary and the optimizer's endpoints.

    python examples/06_oracle.py
"""
import numpy as np

from repair import dataset as ds, viz
from repair.charts import bar_totals, heatmap_landscape, tile
from repair.config import ensure_out
from repair.damage import sample_damage
from repair.energy import BIG, energy
from repair.optimizer import fit_template, oracle
from repair.templates import TEMPLATES

out = ensure_out()


def main():
    key = "CJ_DT"
    m = ds.load_member(key)
    mate = ds.load_mate_interface(ds.load_jwood(key), m)
    dmg = sample_damage(m, np.random.default_rng(6), kind="end_rot")

    best_name, best_fit, fits = oracle(m, dmg, mate, rng=np.random.default_rng(0))
    print(f"  oracle winner: {best_name} (E={best_fit.total:.3f})")

    paths, titles = [], []
    for name, f in fits.items():
        cut = TEMPLATES[name].apply(m, dmg, f.theta)
        p = viz.render_repair(m, dmg, cut, mate, out / f"06_{name}.png", title=name)
        paths.append(p)
        mark = "  <-- WINNER" if name == best_name else ""
        titles.append(f"{name}  (E={f.total:.2f}){mark}")
        print(f"    {name:9s} E={f.total:7.3f} sound={f.energy.sound_removed:.3f}")
    tile(paths, titles, out / "06_oracle_CJ_DT.png", cols=2,
         suptitle=f"Oracle fits all templates -> winner: {best_name}")
    bar_totals(list(fits), [f.total for f in fits.values()],
               out / "06_totals.png", winner=best_name)

    _landscape(m, dmg, mate)
    print("  -> out/06_oracle_CJ_DT.png, out/06_totals.png, out/06_landscape.png")


def _landscape(m, dmg, mate):
    """Total energy over scarf (length x slope), with feasibility + optimizer endpoints."""
    t = TEMPLATES["scarf"]
    lengths = np.linspace(0.12, 1.0, 40)
    slopes = np.linspace(-0.7, 0.7, 40)
    Z = np.zeros((len(slopes), len(lengths)))
    feas = np.zeros_like(Z, dtype=bool)
    for j, s in enumerate(slopes):
        for i, L in enumerate(lengths):
            e = energy(m, dmg, t.apply(m, dmg, np.array([L, s])), mate)
            Z[j, i] = min(e.total, 5.0)            # clip the BIG penalty plateau for color
            feas[j, i] = e.feasible and e.total < BIG
    theta0 = t.default_theta(m, dmg)
    fit = fit_template(m, dmg, t, mate, rng=np.random.default_rng(0))
    path = [theta0, fit.theta]
    heatmap_landscape(lengths, slopes, Z, out / "06_landscape.png",
                      xlabel="scarf length", ylabel="scarf slope",
                      path=path, feasible=feas,
                      title="Scarf energy landscape (white dashed = feasibility)")


if __name__ == "__main__":
    main()
