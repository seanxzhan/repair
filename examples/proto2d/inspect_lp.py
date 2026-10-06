"""Polyscope playground for the capacity LP itself: contacts, friction cones, forces.

    python examples/proto2d/inspect_lp.py              # open the viewer
    python examples/proto2d/inspect_lp.py --headless   # print the readouts for every preset

Pick a contact configuration: two textbook cases (one face; two parallel
faces) or a family at its defaults, in sound wood or under the Figure 1
damage front. Then move the friction coefficient and the force cap, and kill
faces one at a time. At every change the LP is re-solved and the scene shows:

    grey wedge at each live contact   its friction cone (half-angle atan mu)
    orange arrow                      the force the LP put there
    orange wedge                      a contact whose force sits ON its cone edge
                                      (friction maxed out: it is about to slip)
    readout                           M, the force budget used, each contact's
                                      normal and friction force, how much of its
                                      cone it uses, and its moment contribution;
                                      and M against mu, to see the friction angle
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    from repair.proto2d import families as fm, model as sp
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from repair.proto2d import families as fm, model as sp

from inspect_splice import BAD, GOOD, INK, INK2, NEW_WOOD, ORANGE, RENDER_NX, RENDER_NY, _quad_grid

PRESETS = ["one face", "two parallel faces", "tenon", "dovetail", "hooked_scarf", "tenon_flip"]
DAMAGE = ["sound wood", "Figure 1 front"]


def textbook(kind: str, n: int = 7):
    """The hand-checkable cases from model.sanity_checks, placed on the block."""
    x0, x1, y0, y1 = 8.0, 10.0, 0.5, 1.5
    up = [sp.Contact(0, p, np.array([0.0, 1.0]), True) for p in sp._span((x0, y0), (x1, y0), n)]
    if kind == "one face":
        return up
    dn = [sp.Contact(1, p, np.array([0.0, -1.0]), True) for p in sp._span((x0, y1), (x1, y1), n)]
    return up + dn


class Viewer:
    def __init__(self, headless=False):
        self.headless = headless
        self.b, self.grid = sp.Block(), sp.Grid()
        self.preset_idx, self.damage_idx = PRESETS.index("tenon"), 1
        self.mu, self.cap, self.n_samp = 0.5, 100.0, 7
        self.killed: set[int] = set()
        self.show_cones, self.cone_len = True, 0.35
        self.msg = ""
        self.refresh()

    # ---------------------------------------------------------------- model
    @property
    def preset(self):
        return PRESETS[self.preset_idx]

    @property
    def family(self):
        return fm.FAMILIES[self.preset] if self.preset in fm.FAMILIES else None

    @property
    def damage(self):
        return sp.Damage(x0=30.0) if self.damage_idx == 0 else sp.Damage()

    def contacts(self):
        if self.family is None:
            cs = textbook(self.preset, self.n_samp)
        else:
            cs = fm.contact_set(self.family, self.family.defaults(), self.damage, self.b, self.n_samp)
        for c in cs:
            if c.face in self.killed:
                c.live = False
        return cs

    def solve(self, cs, mu=None):
        st = sp.Statics(mu=self.mu if mu is None else mu, cap=self.cap)
        return sp.solve_moment(cs, st)

    # -------------------------------------------------------------- drawing
    def refresh(self):
        cs = self.contacts()
        M, F = self.solve(cs)
        if not self.headless:
            import polyscope as ps
            ps.remove_all_structures()
            self.draw(cs, F)
        self.msg = self.readout(cs, M, F)

    def draw(self, cs, F):
        import polyscope as ps
        b = self.b
        outline = np.array([[0, 0, 0.01], [b.length, 0, 0.01], [b.length, b.height, 0.01], [0, b.height, 0.01]])
        ps.register_curve_network("block outline", outline, "loop", radius=0.004).set_color(INK)
        if self.family is not None:
            import shapely
            p, d = self.family.defaults(), self.damage
            V, Fq, cx, cy = _quad_grid(0, b.length, 0, b.height, RENDER_NX, RENDER_NY)
            removed = np.zeros(len(cx), bool)
            layers = self.family.layers(p, b)
            for lay in layers:
                removed |= shapely.contains_xy(lay.removed, cx, cy)
            kept = ps.register_surface_mesh("retained wood", V, Fq[~removed], smooth_shade=False)
            kept.add_scalar_quantity("damage severity", d.severity(V[:, 0], V[:, 1], b), cmap="reds", vminmax=(0.0, 1.0), enabled=True)
            kept.set_edge_width(0.0)
            if removed.any():
                new = ps.register_surface_mesh("new wood", V, Fq[removed], smooth_shade=False); new.set_color(NEW_WOOD); new.set_edge_width(0.0)
            segs = [s for lay in layers for s in fm.interface_segments(lay, b)]
            P = np.array([[s[0][0], s[0][1], 0.012] for s in segs] + [[s[1][0], s[1][1], 0.012] for s in segs])
            E = np.array([[k, k + len(segs)] for k in range(len(segs))])
            ps.register_curve_network("interface", P, E, radius=0.005).set_color(INK)
        else:
            # the textbook faces, as lines
            for fi in sorted({c.face for c in cs}):
                pts = np.array([c.point for c in cs if c.face == fi])
                P = np.array([[pts[0, 0] - 0.15, pts[0, 1], 0.012], [pts[-1, 0] + 0.15, pts[-1, 1], 0.012]])
                ps.register_curve_network(f"face {fi}", P, "line", radius=0.005).set_color(INK)
        # contacts, forces
        P = np.array([[c.point[0], c.point[1], 0.02] for c in cs])
        pc = ps.register_point_cloud("contacts", P); pc.set_radius(0.035, relative=False)
        pc.add_color_quantity("live / dead", np.array([GOOD if c.live else BAD for c in cs]), enabled=True)
        if np.any(F):
            pc.add_vector_quantity("LP force", np.column_stack([F, np.zeros(len(F))]), enabled=True, color=ORANGE, length=0.12, radius=0.004)
        # friction cones: a wedge of half-angle atan(mu) about the normal, orange where the force is on its edge
        if self.show_cones:
            half = np.arctan(self.mu)
            Vc, Fc, col = [], [], []
            for k, c in enumerate(cs):
                if not c.live:
                    continue
                n = c.normal; t = np.array([-n[1], n[0]]); L = self.cone_len
                e1 = n * np.cos(half) + t * np.sin(half); e2 = n * np.cos(half) - t * np.sin(half)
                i0 = len(Vc)
                Vc += [[*c.point, 0.015], [*(c.point + L * e1), 0.015], [*(c.point + L * e2), 0.015]]
                Fc.append([i0, i0 + 1, i0 + 2])
                fn, ft = float(F[k] @ n), float(F[k] @ t)
                on_edge = fn > 1e-6 and abs(ft) >= self.mu * fn - 1e-6 and self.mu > 0
                col.append(ORANGE if on_edge else (0.55, 0.55, 0.55))
            if Fc:
                m = ps.register_surface_mesh("friction cones", np.array(Vc), np.array(Fc))
                m.add_color_quantity("on cone edge", np.array(col), defined_on="faces", enabled=True)
                m.set_transparency(0.45); m.set_edge_width(1.0)

    def readout(self, cs, M, F):
        lines = [f"{self.preset}   {DAMAGE[self.damage_idx] if self.family else ''}   mu = {self.mu:.2f}   cap = {self.cap:.0f}",
                 f"capacity M = {M:.2f}      {sum(c.live for c in cs)}/{len(cs)} live" + (f"   killed faces: {sorted(self.killed)}" if self.killed else "")]
        fn_all = sum(float(F[k] @ c.normal) for k, c in enumerate(cs))
        net = F.sum(axis=0); mom = sum(float(c.point[0] * F[k][1] - c.point[1] * F[k][0]) for k, c in enumerate(cs))
        lines.append(f"budget: sum f_n = {fn_all:.1f} of {self.cap:.0f}      net force = ({net[0]:+.2f}, {net[1]:+.2f})      "
                     f"sum of moments = {mom:+.2f} = -M")
        lines.append("")
        lines.append(f"{'contact':>8} {'face':>4} {'point':>14} {'f_n':>7} {'f_t':>7} {'cone used':>10} {'moment':>8}")
        for k, c in enumerate(cs):
            if not c.live:
                continue
            fn, ft = float(F[k] @ c.normal), float(F[k] @ np.array([-c.normal[1], c.normal[0]]))
            if fn < 1e-6 and abs(ft) < 1e-6:
                continue
            used = abs(ft) / (self.mu * fn) if self.mu * fn > 1e-9 else 0.0
            mk = c.point[0] * F[k][1] - c.point[1] * F[k][0]
            flag = "  <- on the cone edge" if self.mu > 0 and used > 0.999 else ""
            lines.append(f"{k:>8} {c.face:>4} ({c.point[0]:5.2f},{c.point[1]:4.2f}) {fn:7.2f} {ft:+7.2f} {100 * used:9.0f}% {mk:+8.2f}{flag}")
        lines.append("  (only contacts carrying force are listed; the LP leaves most at zero)")
        lines.append("")
        row = []
        for mu in (0.0, 0.25, 0.5, 0.75, 1.0):
            row.append(f"mu {mu:.2f}: M {self.solve(cs, mu)[0]:6.1f}")
        lines.append("capacity against friction:   " + "   ".join(row))
        return "\n".join(lines)

    # ------------------------------------------------------------------ GUI
    def gui(self):
        import polyscope.imgui as psim
        changed = False
        psim.PushItemWidth(220)
        ch, idx = psim.Combo("contacts", self.preset_idx, PRESETS)
        if ch:
            self.preset_idx = idx; self.killed = set(); changed = True
        if self.family is not None:
            ch, self.damage_idx = psim.Combo("damage", self.damage_idx, DAMAGE); changed |= ch
        ch, self.n_samp = psim.SliderInt("samples per face", self.n_samp, 2, 15); changed |= ch
        if psim.CollapsingHeader("statics", True):
            ch, self.mu = psim.SliderFloat("friction mu", self.mu, 0.0, 1.5); changed |= ch
            ch, self.cap = psim.SliderFloat("cap on sum(f_n)", self.cap, 5.0, 400.0); changed |= ch
        if psim.CollapsingHeader("faces (uncheck to kill a face)", True):
            cs = self.contacts()
            for fi in sorted({c.face for c in cs}):
                cc = [c for c in cs if c.face == fi]; n = cc[0].normal
                ch, alive = psim.Checkbox(f"face {fi}  n=({n[0]:+.2f},{n[1]:+.2f})  {len(cc)} samples##face{fi}", fi not in self.killed)
                if ch:
                    (self.killed.discard if alive else self.killed.add)(fi); changed = True
        if psim.CollapsingHeader("cones", True):
            ch, self.show_cones = psim.Checkbox("show friction cones", self.show_cones); changed |= ch
            ch, self.cone_len = psim.SliderFloat("cone length (drawing only)", self.cone_len, 0.1, 1.0); changed |= ch
        psim.PopItemWidth()
        if changed:
            self.refresh()
        psim.Separator(); psim.TextUnformatted(self.msg)


def main():
    if "--headless" in sys.argv[1:]:
        v = Viewer(headless=True)
        for i in range(len(PRESETS)):
            v.preset_idx = i; v.refresh(); print(v.msg); print()
        return
    import polyscope as ps
    ps.init(); ps.set_ground_plane_mode("none"); ps.set_up_dir("z_up"); ps.set_navigation_style("turntable"); ps.set_view_projection_mode("perspective")
    viewer = Viewer()
    ps.set_user_callback(viewer.gui)
    ps.look_at((8.0, -5.0, 5.0), (8.0, 1.0, 0.0))
    ps.show()


if __name__ == "__main__":
    main()
