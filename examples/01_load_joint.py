"""01 - Load a MiGumi joint and extract its 2D longitudinal member profile.

Shows the linchpin data step: a joint's richest LHF polygon -> a normalized 2D member,
extruded to a timber prism, with the frozen mate (coupling) face highlighted.

    python examples/01_load_joint.py [--show]
"""
import sys

from repair import dataset as ds, viz
from repair.charts import tile
from repair.config import ensure_out

SHOW = "--show" in sys.argv
out = ensure_out()


def main():
    keys = ds.loadable_joints()
    print(f"{len(keys)} loadable joints")

    key = "CJ_DT"
    m = ds.load_member(key)
    jw = ds.load_jwood(key)
    mate = ds.load_mate_interface(jw, m)
    print(f"{key}: part={m.part_name} '{m.meta.name}' ({m.meta.type}) "
          f"verts={len(m.polygon)} area={m.poly_shapely.area:.3f}")
    viz.render_member(m, out / "01_member_CJ_DT.png", mate=mate)
    print("  -> out/01_member_CJ_DT.png")

    # gallery of a few joints
    gallery = ["CJ_DT", "CJ_NT", "CJ_SAT_3", "RJ_KA", "CJ_KKT", "RJM_SDT"]
    paths, titles = [], []
    for k in gallery:
        mk = ds.load_member(k)
        mate_k = ds.load_mate_interface(ds.load_jwood(k), mk)
        p = viz.render_member(mk, out / f"01_g_{k}.png", mate=mate_k)
        paths.append(p)
        titles.append(f"{k}  ({mk.meta.name})")
    tile(paths, titles, out / "01_gallery.png", cols=3,
         suptitle="MiGumi joints -> 2D member profiles (blue = frozen mate face)")
    print("  -> out/01_gallery.png")

    if SHOW:
        viz.render_member(m, out / "01_member_CJ_DT.png", mate=mate)
        viz.show()


if __name__ == "__main__":
    main()
