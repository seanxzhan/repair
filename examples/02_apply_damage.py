"""02 - Procedurally damage a member (rot) plus natural features (knots, checks).

Rot follows realistic modes: central rot, end rot, ground-contact (bottom) rot -- not
uniform noise. The red region is ``must_replace`` (the hard coverage constraint).

    python examples/02_apply_damage.py [--show]
"""
import sys

import numpy as np

from repair import dataset as ds, viz
from repair.charts import tile
from repair.config import ensure_out
from repair.damage import sample_damage

SHOW = "--show" in sys.argv
out = ensure_out()


def main():
    key = "CJ_DT"
    m = ds.load_member(key)
    mate = ds.load_mate_interface(ds.load_jwood(key), m)
    rng = np.random.default_rng(7)

    paths, titles = [], []
    for kind in ["central_rot", "end_rot", "ground_contact"]:
        dmg = sample_damage(m, rng, kind=kind)
        p = viz.render_damage(m, dmg, mate, out / f"02_{kind}.png")
        paths.append(p)
        titles.append(f"{kind}  (rot area={dmg.must_replace.area:.2f}, "
                      f"{len(dmg.features)} features)")
        print(f"  {kind}: rot_area={dmg.must_replace.area:.3f} features={len(dmg.features)}")
    tile(paths, titles, out / "02_damage_CJ_DT.png", cols=3,
         suptitle="Procedural damage modes (red = must-replace rot, brown = knots/checks)")
    print("  -> out/02_damage_CJ_DT.png")

    if SHOW:
        viz.render_damage(m, sample_damage(m, rng, kind="central_rot"), mate,
                          out / "02_central_rot.png")
        viz.show()


if __name__ == "__main__":
    main()
