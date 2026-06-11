"""Diagnostic / test for the repair algorithm.

On a clean rectangular member (so the template shapes aren't masked by a noisy joint
profile) this:
  1. renders the 4 templates EXPLODED -- the insert lifted out of its cavity -- so their
     distinct interfaces (dovetail flare, scarf slope, ledge step, dutchman patch) are visible,
  2. prints a metrics table proving the interfaces really differ,
  3. asserts the core algorithm invariants (coverage, mate preserved, label diversity).

Run:  python examples/test_templates.py [--show]
Exits non-zero if any check fails.
"""
import sys

import numpy as np
from shapely.geometry import LineString, Point, box

from repair import dataset as ds, viz
from repair.charts import tile
from repair.config import ensure_out
from repair.damage import Damage, sample_damage
from repair.energy import (_corner_count, _grain_penalty, _interface_length,
                           _segments, energy)
from repair.optimizer import fit_template, oracle
from repair.templates import TEMPLATES

SHOW = "--show" in sys.argv
out = ensure_out()
CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")


# --------------------------------------------------------------- fixtures

def make_rect_member(w=1.2, h=0.5):
    poly = box(-w / 2, -h / 2, w / 2, h / 2)
    ring = np.array(poly.exterior.coords)[:-1]
    return ds.Member(key="RECT", part_name="test", polygon=ring, poly_shapely=poly,
                     dropped_axis=2, plane_normal=np.array([0., 0., 1.]),
                     meta=ds.JointMeta("RECT", "Rectangle", "Test", 1))


def rect_mate(m):
    x0, y0, x1, y1 = m.bbox
    return LineString([(x0, y0), (x0, y1)])


def fixed_damage(m, where):
    if where == "central":
        rot = Point(0.12, 0.0).buffer(0.13)
    elif where == "end":
        x0, y0, x1, y1 = m.bbox
        rot = box(x1 - 0.22, y0 - 1, x1 + 1, y1 + 1)
    else:
        raise ValueError(where)
    return Damage(must_replace=rot.intersection(m.poly_shapely), features=[], kind=where)


# --------------------------------------------------------------- metrics

def interface_metrics(member, damage, cut):
    segs = _segments(cut.interface)
    return dict(
        insert_area=cut.insert.area,
        iface_len=_interface_length(segs),
        grain=_grain_penalty(segs),
        corners=_corner_count(cut.insert),
        feasible=cut.feasible,
    )


def coverage_ok(damage, cut, member):
    uncovered = damage.must_replace.difference(cut.insert).area
    return uncovered <= 0.02 * member.poly_shapely.area


# --------------------------------------------------------------- the test

def test_on(member, damage, mate, tag):
    print(f"\n--- {tag} damage (rot_area={damage.must_replace.area:.3f}) ---")
    print(f"  {'template':10s} {'area':>6s} {'iface':>6s} {'grain':>6s} "
          f"{'corners':>7s} {'total':>7s} {'feas':>5s} {'covers':>6s}")
    rows, paths, titles = {}, [], []
    for name, t in TEMPLATES.items():
        fit = fit_template(member, damage, t, mate, rng=np.random.default_rng(0))
        cut = t.apply(member, damage, fit.theta)
        mm = interface_metrics(member, damage, cut)
        cov = coverage_ok(damage, cut, member)
        rows[name] = dict(fit=fit, cut=cut, m=mm, cov=cov)
        print(f"  {name:10s} {mm['insert_area']:6.3f} {mm['iface_len']:6.3f} "
              f"{mm['grain']:6.2f} {mm['corners']:7d} {fit.total:7.3f} "
              f"{str(mm['feasible']):>5s} {str(cov):>6s}")
        p = viz.render_repair(member, damage, cut, mate,
                              out / f"test_{tag}_{name}.png", title=name, explode=0.4)
        paths.append(p)
        titles.append(f"{name}  (E={fit.total:.2f})")
    tile(paths, titles, out / f"test_templates_{tag}.png", cols=2,
         suptitle=f"Templates EXPLODED on {tag} damage (insert lifted out of cavity)")
    print(f"  -> out/test_templates_{tag}.png")
    return rows


def main():
    m = make_rect_member()
    mate = rect_mate(m)

    # ---- exploded renders + metrics on central and end damage
    central = test_on(m, fixed_damage(m, "central"), mate, "central")
    end = test_on(m, fixed_damage(m, "end"), mate, "end")

    print("\n--- invariant checks ---")
    # 1. every template yields a feasible, covering repair on at least one damage type
    for name in TEMPLATES:
        feas = central[name]["cut"].feasible or end[name]["cut"].feasible
        check(f"{name} produces a feasible cut", feas)

    # 2. coverage holds for the oracle winner on both damage types
    for tag, rows, dmg in [("central", central, fixed_damage(m, "central")),
                           ("end", end, fixed_damage(m, "end"))]:
        win = min(rows, key=lambda n: rows[n]["fit"].total)
        check(f"oracle({tag}) winner '{win}' covers all rot", rows[win]["cov"])

    # 3. interfaces genuinely DIFFER (the user's question): the four templates must not
    #    collapse to the same shape -- compare grain angle and corner count signatures.
    grains = [round(end[n]["m"]["grain"], 2) for n in TEMPLATES]
    corners = [end[n]["m"]["corners"] for n in TEMPLATES]
    check("interface grain angles differ across templates",
          len(set(grains)) >= 3, f"grains={grains}")
    check("interface corner counts differ across templates",
          len(set(corners)) >= 2, f"corners={corners}")

    # 4. mate is preserved by the oracle winner (no mate violation)
    for tag, rows in [("central", central), ("end", end)]:
        win = min(rows, key=lambda n: rows[n]["fit"].total)
        check(f"oracle({tag}) winner preserves mate",
              not rows[win]["fit"].energy.violations.get("mate", False))

    # 5. label diversity: dutchman should dominate CENTRAL rot; an end-splice template
    #    (ledge/scarf/dovetail) should win most END rot. Aggregate over random members.
    cwin = _winrate(m, mate, "central_rot")
    ewin = _winrate(m, mate, "end_rot")
    check("dutchman wins majority of central rot",
          cwin.get("dutchman", 0) > 0.5, f"central winrates={cwin}")
    check("an end-splice template wins majority of end rot",
          sum(ewin.get(k, 0) for k in ("ledge", "scarf", "dovetail")) > 0.5,
          f"end winrates={ewin}")

    n_pass = sum(CHECKS)
    print(f"\n{n_pass}/{len(CHECKS)} checks passed")
    if SHOW:
        d = fixed_damage(m, "central")
        t = TEMPLATES["dovetail"]
        fit = fit_template(m, d, t, mate, rng=np.random.default_rng(0))
        viz.render_repair(m, d, t.apply(m, d, fit.theta), mate,
                          out / "test_show.png", explode=0.4)
        viz.show()
    if n_pass != len(CHECKS):
        sys.exit(1)


def _winrate(member, mate, kind, n=20):
    from collections import Counter
    rng = np.random.default_rng(1)
    c = Counter()
    for _ in range(n):
        dmg = sample_damage(member, rng, kind=kind)
        name, _, _ = oracle(member, dmg, mate, rng=rng)
        c[name] += 1
    return {k: v / n for k, v in c.items()}


if __name__ == "__main__":
    main()
