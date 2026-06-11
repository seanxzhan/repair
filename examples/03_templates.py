"""03 - The repair vocabulary: 4 parametric templates at their default parameters.

Each template splits the member into kept (green) + insert (orange). They differ in
interface shape -- which is exactly what the energy will score in later steps.

    python examples/03_templates.py [--show]
"""
import sys

import numpy as np

from repair import dataset as ds, viz
from repair.charts import tile
from repair.config import ensure_out
from repair.damage import Damage
from repair.optimizer import fit_template
from repair.templates import TEMPLATES
from shapely.geometry import box

SHOW = "--show" in sys.argv
out = ensure_out()


def main():
    # Clean rectangular canvas so each template's shape is unambiguous (the storyboard's
    # other scripts use real MiGumi joints; this one is the vocabulary diagram).
    m = ds.rectangular_member()
    mate = ds.left_end_mate(m)
    x0, y0, x1, y1 = m.bbox
    dmg = Damage(must_replace=box(x1 - 0.22, y0, x1, y1), features=[], kind="end_rot")

    paths, titles = [], []
    for name, t in TEMPLATES.items():
        fit = fit_template(m, dmg, t, mate, rng=np.random.default_rng(0))  # optimized
        cut = t.apply(m, dmg, fit.theta)
        # exploded view lifts the insert out so the interface shape is visible
        p = viz.render_repair(m, dmg, cut, mate, out / f"03_{name}.png",
                              title=name, explode=0.4)
        paths.append(p)
        tag = "" if cut.feasible else "  (infeasible for end rot)"
        titles.append(f"{name}{tag}")
        print(f"  {name:9s} insert_area={cut.insert.area:.3f} feasible={cut.feasible}")
    tile(paths, titles, out / "03_templates.png", cols=2,
         suptitle="Repair templates EXPLODED (ledge=butt, scarf=slope, "
                  "dovetail=interlock, dutchman=interior-only)")
    print("  -> out/03_templates.png")

    if SHOW:
        t = TEMPLATES["dovetail"]
        fit = fit_template(m, dmg, t, mate, rng=np.random.default_rng(0))
        viz.render_repair(m, dmg, t.apply(m, dmg, fit.theta), mate,
                          out / "03_dovetail.png", title="dovetail", explode=0.4)
        viz.show()


if __name__ == "__main__":
    main()
