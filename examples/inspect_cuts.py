"""Polyscope viewer for the CUTS that carve a part out of its stock.

Each part in the dataset is `Difference(stock, Union(cuts...))`, where every cut
is an LHF: a 2D sketch on a plane, swept `amount` along that plane's normal.
This viewer takes those cuts apart -- one at a time, in place, with the sketch
and the plane frame drawn where they actually sit.

    python examples/inspect_cuts.py              # CJ_DT
    python examples/inspect_cuts.py CJ_AKT       # 3-part keyed joint (holes)
    python examples/inspect_cuts.py --headless   # run the verification only

Three tabs:

  1 CUTS    The point of the tool. Pick a part, then step through its cuts. The
            selected cut is solid, the others are ghosted, and its sketch rings
            (base + at depth), plane origin and (u, v, n) frame are drawn on it.
  2 ORIENT  Why a cut is not fully determined by what the file stores. All 8
            in-plane orientations of the selected cut, grouped by which ones are
            actually the same solid.
  3 VERIFY  A check, not a workshop: construct cuts from scratch and confirm
            they survive a JSON round trip through the unmodified reader, and
            that the reader recovers them at ori=0 with no STL to lean on. This
            is what licenses the canonical exports written by `export_lhf.py`.
"""
from __future__ import annotations

import sys

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from repair.authoring import (SKETCHES, buried_fraction, make_lhf, orientation_classes,
                              roundtrip_error, tool_radius_violation,
                              verify_reader_recovers)
from repair.config import DATASET_ROOT
from repair.jwood import lhf_to_mesh, orientation_frame
from repair.relational import reconstruct_parts

VARIANTS = ["base", "mill", "odf", "ours"]
WOOD = (0.78, 0.72, 0.60)
STOCK = (0.55, 0.55, 0.55)
CUT_COLORS = [(0.85, 0.20, 0.15), (0.92, 0.62, 0.15), (0.20, 0.45, 0.85),
              (0.30, 0.68, 0.38), (0.62, 0.38, 0.78), (0.30, 0.70, 0.72)]
CLASS_COLORS = CUT_COLORS + [(0.85, 0.78, 0.25), (0.50, 0.50, 0.50)]
NORMALS = [("+X", (1, 0, 0)), ("+Y", (0, 1, 0)), ("+Z", (0, 0, 1)),
           ("+X+Y 45", (0.7071, 0.7071, 0))]


def _show(name, mesh, color, offset=(0.0, 0.0, 0.0), transparency=1.0):
    if mesh is None or mesh.is_empty or len(mesh.faces) == 0:
        return
    V = np.asarray(mesh.vertices, np.float64) + np.asarray(offset, np.float64)
    m = ps.register_surface_mesh(name, V, np.asarray(mesh.faces, np.int64),
                                 smooth_shade=False, transparency=transparency)
    m.set_color(color)


def _show_ring(name, pts3d, color, radius=0.006):
    pts = np.asarray(pts3d, np.float64)
    edges = np.column_stack([np.arange(len(pts) - 1), np.arange(1, len(pts))])
    ps.register_curve_network(name, pts, edges, radius=radius).set_color(color)


def _show_frame(name, origin, frame, scale=0.4):
    """(u, v, n): red and green span the sketch plane, blue is the sweep axis."""
    o = np.asarray(origin, float)
    for i, col in enumerate([(0.9, 0.2, 0.2), (0.2, 0.9, 0.2), (0.2, 0.4, 0.95)]):
        c = ps.register_curve_network(f"{name}/{'uvn'[i]}",
                                      np.array([o, o + scale * frame[:, i]]),
                                      np.array([[0, 1]]), radius=0.008)
        c.set_color(col)


def ring_to_3d(ring2d, lhf, ori=0, height=0.0):
    frame = orientation_frame(lhf.normal, ori)
    return lhf.origin + np.asarray(ring2d) @ frame[:, :2].T + height * frame[:, 2]


def _vec(v) -> str:
    return "[" + ",".join(f"{x:+.2f}" for x in v) + "]"


def _lhf_text(lhf, ori) -> str:
    rings = (f"{len(lhf.rings)} rings {lhf.signs}" if len(lhf.rings) > 1
             else f"{len(lhf.rings[0])}-gon")
    return (f"{lhf.name}: n={_vec(lhf.normal)} o={_vec(lhf.origin)} "
            f"amount={lhf.amount:+.3f} ori={ori} ({rings})")


# --------------------------------------------------------------------------- #
# Verification: constructed cuts must survive the reader untouched.
# --------------------------------------------------------------------------- #
def verification_rows(part) -> list[dict]:
    """One row per constructed cut: round-trip exactness and reader recovery."""
    rows = []
    for sk in SKETCHES:
        for nlabel, normal in NORMALS[:2]:
            cut = make_lhf(f"chk_{sk.label}", *sk.build(*sk.defaults), normal=normal,
                           origin=part.solid.bounds[0], amount=0.6)
            sym, vol = roundtrip_error(cut)
            _, co, ag, ok = verify_reader_recovers(part.spec.stock, [cut], n_samples=4000)
            rows.append({"case": f"{sk.label} @ {nlabel}", "sym": sym, "vol": vol,
                         "cut_ori": co[0], "agreement": ag, "recovered": ok,
                         "tool": tool_radius_violation(cut, 0.05),
                         "buried": buried_fraction(cut, part.solid), "lhf": cut})
    return rows


def rows_text(rows) -> str:
    head = (f"{'constructed cut':22s} {'roundtrip':>11s} {'ori':>4s} {'agree':>7s} "
            f"{'ok':>4s}\n")
    body = "".join(
        f"{r['case']:22s} {r['sym']:11.2e} {r['cut_ori']:4d} "
        f"{r['agreement'] * 100:6.2f}% {'PASS' if r['recovered'] and r['sym'] < 1e-9 else 'FAIL':>4s}\n"
        for r in rows)
    n_ok = sum(1 for r in rows if r["recovered"] and r["sym"] < 1e-9)
    return head + body + f"\n{n_ok}/{len(rows)} pass"


class Viewer:
    def __init__(self, key: str):
        self.joints = sorted(p.name for p in DATASET_ROOT.iterdir()
                             if (p / "info.json").exists())
        self.key = key if key in self.joints else self.joints[0]
        self.variant_idx = 0
        self.part_idx = 0
        self.cut_idx = 0          # -1 => show every cut at once
        self.tab = 0
        self.show_stock = False
        self.show_solid = True
        self.show_sketch = True
        self.rows = None          # verification results, computed on demand
        self.msg = ""
        self.load_joint()

    def load_joint(self):
        try:
            self.recons = reconstruct_parts(self.key, VARIANTS[self.variant_idx])
        except Exception as e:
            self.recons = []
            self.msg = f"cannot load {self.key}: {e!r}"
            return
        self.part_idx = min(self.part_idx, len(self.recons) - 1)
        self.cut_idx = 0
        self.rows = None
        self.refresh()

    @property
    def part(self):
        return self.recons[self.part_idx]

    @property
    def cuts(self):
        return list(zip(self.part.spec.cuts, self.part.cut_oris))

    def refresh(self):
        ps.remove_all_structures()
        if not self.recons:
            return
        [self.draw_cuts, self.draw_orient, self.draw_verify][self.tab]()

    # ------------------------------------------------------------- 1 CUTS
    def draw_cuts(self):
        r = self.part
        if self.show_solid:
            _show(f"{r.name} (solid)", r.solid, WOOD, transparency=0.30)
        if self.show_stock:
            _show(f"{r.name} (stock)", r.stock_mesh, STOCK, transparency=0.12)

        for i, (cut, ori) in enumerate(self.cuts):
            sel = (self.cut_idx < 0 or i == self.cut_idx)
            _show(f"cut {i}: {cut.name}", lhf_to_mesh(cut, ori),
                  CUT_COLORS[i % len(CUT_COLORS)],
                  transparency=0.85 if sel else 0.10)

        if 0 <= self.cut_idx < len(self.cuts) and self.show_sketch:
            cut, ori = self.cuts[self.cut_idx]
            _show_frame("plane", cut.origin, orientation_frame(cut.normal, ori))
            for j, (ring, sgn) in enumerate(zip(cut.rings, cut.signs)):
                col = (0.05, 0.05, 0.05) if sgn > 0 else (0.95, 0.85, 0.10)
                _show_ring(f"sketch/ring{j} (sign {sgn:+d})",
                           ring_to_3d(ring, cut, ori), col)
                _show_ring(f"sketch/ring{j} @ depth",
                           ring_to_3d(ring, cut, ori, abs(cut.amount)), col, radius=0.004)

        lines = [f"{r.name}: {r.spec.expression}",
                 f"  solid {r.solid.volume:.4f}  stock {r.stock_mesh.volume:.4f}  "
                 f"removed {r.stock_mesh.volume - r.solid.volume:.4f}",
                 f"  stock  {_lhf_text(r.spec.stock, r.stock_ori)}", ""]
        for i, (cut, ori) in enumerate(self.cuts):
            mark = ">" if i == self.cut_idx else " "
            lines.append(f" {mark}cut {i}  {_lhf_text(cut, ori)}")
            lines.append(f"     volume {lhf_to_mesh(cut, ori).volume:.4f}")
        if 0 <= self.cut_idx < len(self.cuts):
            lines += ["", "red/green = the sketch plane's (u, v) axes; blue = the sweep",
                      "axis. Black rings are boundaries, yellow rings are holes.",
                      "The two rings per boundary are the sketch at depth 0 and at",
                      "`amount` -- the cut is exactly the volume between them."]
        self.msg = "\n".join(lines)

    # ----------------------------------------------------------- 2 ORIENT
    def draw_orient(self):
        if not self.cuts:
            self.msg = f"{self.part.name} has no cuts"
            return
        i = max(self.cut_idx, 0)
        cut, ori = self.cuts[min(i, len(self.cuts) - 1)]
        cls = orientation_classes(cut)
        span = float(np.ptp(lhf_to_mesh(cut, 0).bounds[:, 0])) + 0.6
        for o in range(8):
            tag = " <-- RESOLVED" if o == ori else ""
            _show(f"ori {o} [class {cls[o]}]{tag}", lhf_to_mesh(cut, o),
                  CLASS_COLORS[cls[o] % len(CLASS_COLORS)],
                  offset=(o * span * 1.2 - 3.5 * span, 0, 0),
                  transparency=1.0 if o == ori else 0.55)
        groups = {c: [o for o in range(8) if cls[o] == c] for c in sorted(set(cls))}
        self.msg = (
            f"{cut.name}: 8 candidate orientations, {len(set(cls))} distinct solids.\n"
            "  same colour = same solid: "
            + "; ".join("{" + ",".join(map(str, v)) + "}" for v in groups.values())
            + f"\n  resolved to ori={ori}\n\n"
            "ori % 4 rotates the sketch 0/90/180/270 deg in its plane; ori >= 4 also\n"
            "reflects it. The file records the plane but not which of these the author\n"
            "meant, so the importer tries all 8 and keeps whichever matches the STL.\n"
            "Where several are the same solid, the recovered index is arbitrary.\n\n"
            "Across all 61 cached parts the resolver never picks 4-7, even though 40\n"
            "LHFs are chiral and a reflection would give a different solid -- so the\n"
            "authoring convention uses rotations only.")

    # ----------------------------------------------------------- 3 VERIFY
    def draw_verify(self):
        if self.rows is None:
            self.msg = ("VERIFY -- press 'run verification' below.\n\n"
                        "Constructs cuts from scratch, encodes each to the dataset's own\n"
                        "JSON form, decodes it with the unmodified reader, and checks the\n"
                        "solid is unchanged and recovered at ori=0 with no STL involved.")
            return
        _show(f"{self.part.name}", self.part.solid, WOOD, transparency=0.35)
        for i, r in enumerate(self.rows[:6]):
            _show(f"check/{r['case']}", lhf_to_mesh(r["lhf"], 0),
                  CUT_COLORS[i % len(CUT_COLORS)], transparency=0.6)
        self.msg = (
            rows_text(self.rows) + "\n\n"
            "roundtrip = symmetric-difference volume after encode/decode (0 = exact).\n"
            "ori/agree = what the reader recovered when handed the constructed solid\n"
            "as if it were ground truth. A cut authored in seed_basis(normal) comes\n"
            "back at ori=0 without any orientation search, which is what makes the\n"
            "canonical exports in out/lhf_base/ readable on their own.")

    # -------------------------------------------------------------------- GUI
    def gui(self):
        changed_any = False
        changed, idx = psim.Combo("joint", self.joints.index(self.key), self.joints)
        if changed:
            self.key = self.joints[idx]
            self.load_joint()
        changed, self.variant_idx = psim.Combo("variant", self.variant_idx, VARIANTS)
        if changed:
            self.load_joint()
        if self.recons:
            changed, self.part_idx = psim.Combo("part", self.part_idx,
                                                [r.name for r in self.recons])
            if changed:
                self.cut_idx, self.rows = 0, None
                changed_any = True
        psim.Separator()

        for i, t in enumerate(["1 CUTS", "2 ORIENT", "3 VERIFY"]):
            if psim.RadioButton(t, self.tab == i):
                self.tab, changed_any = i, True
            if i < 2:
                psim.SameLine()
        psim.Separator()

        if self.tab in (0, 1) and self.recons:
            labels = ["all cuts"] + [f"cut {i}: {c.name}"
                                     for i, (c, _) in enumerate(self.cuts)]
            changed, sel = psim.Combo("cut", self.cut_idx + 1, labels)
            if changed:
                self.cut_idx, changed_any = sel - 1, True
        if self.tab == 0:
            changed, self.show_solid = psim.Checkbox("show solid", self.show_solid)
            changed_any |= changed
            psim.SameLine()
            changed, self.show_stock = psim.Checkbox("show stock", self.show_stock)
            changed_any |= changed
            psim.SameLine()
            changed, self.show_sketch = psim.Checkbox("show sketch", self.show_sketch)
            changed_any |= changed
        if self.tab == 2:
            if psim.Button("run verification"):
                self.rows = verification_rows(self.part)
                print(rows_text(self.rows))
                changed_any = True

        if changed_any:
            self.refresh()
        psim.Separator()
        psim.TextUnformatted(self.msg)


def headless(key: str):
    recons = reconstruct_parts(key, "base")
    part = recons[0]
    print(f"=== cuts of {key}/{part.name} ===")
    print(f"  {part.spec.expression}")
    print(f"  stock  {_lhf_text(part.spec.stock, part.stock_ori)}")
    for i, (cut, ori) in enumerate(zip(part.spec.cuts, part.cut_oris)):
        cls = orientation_classes(cut)
        print(f"  cut {i}  {_lhf_text(cut, ori)}  volume="
              f"{lhf_to_mesh(cut, ori).volume:.4f}  distinct_of_8={len(set(cls))}")
    print(f"\n=== verification on {part.name} ===")
    print(rows_text(verification_rows(part)))


def main():
    args = sys.argv[1:]
    if "--headless" in args:
        args.remove("--headless")
        headless(args[0] if args else "CJ_DT")
        return
    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_up_dir("z_up")
    ps.set_navigation_style("turntable")
    viewer = Viewer(args[0] if args else "CJ_DT")
    ps.set_user_callback(viewer.gui)
    ps.show()


if __name__ == "__main__":
    main()
