"""04 - The proxy energy and its central tension.

One template (scarf) is evaluated at three settings -- a too-short interface, a balanced
one, and a too-long one -- so you can see ``sound_removed`` trade against ``structural``
and ``fabrication``. This tension is what makes the repair choice non-trivial.

    python examples/04_energy.py
"""
import numpy as np

from repair import dataset as ds, viz
from repair.charts import bar_energy_terms, tile
from repair.config import ensure_out
from repair.damage import sample_damage
from repair.energy import energy
from repair.templates import TEMPLATES

out = ensure_out()


def main():
    key = "CJ_DT"
    m = ds.load_member(key)
    mate = ds.load_mate_interface(ds.load_jwood(key), m)
    dmg = sample_damage(m, np.random.default_rng(4), kind="end_rot")
    t = TEMPLATES["scarf"]

    # scarf theta = [length, slope]; vary length within the feasible range so the
    # trade is visible: removing more sound wood buys a longer (stronger) interface.
    settings = {"tight (length=0.35)": [0.35, 0.35],
                "balanced (length=0.55)": [0.55, 0.35],
                "long (length=0.80)": [0.80, 0.35]}
    terms, labels, paths, titles = [], [], [], []
    for label, theta in settings.items():
        cut = t.apply(m, dmg, np.array(theta))
        e = energy(m, dmg, cut, mate)
        terms.append(e)
        labels.append(label)
        paths.append(viz.render_repair(m, dmg, cut, mate,
                                       out / f"04_{theta[0]:.2f}.png", title=label))
        titles.append(f"{label}\nsound={e.sound_removed:.2f} struct={e.structural:.2f} "
                      f"total={e.total:.2f}")
        print(f"  {label:26s} sound={e.sound_removed:.3f} struct={e.structural:.2f} "
              f"fab={e.fabrication:.2f} total={e.total:.2f}")

    bar_energy_terms(terms, labels, out / "04_energy_terms.png")
    tile(paths, titles, out / "04_geometry.png", cols=3,
         suptitle="Same template, three interface lengths")
    print("  -> out/04_energy_terms.png, out/04_geometry.png")


if __name__ == "__main__":
    main()
