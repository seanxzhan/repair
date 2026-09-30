"""Polyscope playground for the 2D interface families in repair.proto2d.families.

Pick a family (plain cut, mortise-and-tenon, the CJ_AT dovetail and the CJ_DT
hooked scarf, the last three each in both orientations: `*_flip` is the same
joint cut the other way round); every parameter of the family is a slider, as
are the damage front and the statics. The block is drawn with its retained wood
coloured by damage severity, its new wood plain, and its contact samples live
(green) or dead (red) with the LP's forces on them.

    python examples/proto2d/inspect_families.py                # open the viewer
    python examples/proto2d/inspect_families.py --headless     # every family at its defaults, as text

"compute landscape" sweeps any two of the family's parameters with the rest
held at the sliders, and draws capacity, the objective or the sound wood
removed below the strips with the active-set cliffs in black, a marker where
you are and one at the optimum; "go to landscape optimum" moves the two swept
sliders there.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from repair.proto2d import families as fm, model as sp
except ImportError:                       # not installed: fall back to the source tree
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from repair.proto2d import families as fm, model as sp

from inspect_splice import (AQUA, BAD, FRAME, GOOD, INK, INK2, NEW_WOOD, ORANGE,
                            RENDER_NX, RENDER_NY, LAND_MODES, _quad_grid)

KEYS = list(fm.FAMILIES)
STRIP_GAP = 0.6                            # vertical space between layer strips
LAND_ORIGIN_Y = -3.6                       # the landscape sits below the last strip
LAND_SIZE = (8.0, 2.4)


class Viewer:
    def __init__(self, headless=False):
        self.headless = headless
        self.block, self.damage, self.statics = sp.Block(), sp.Damage(), sp.Statics()
        self.grid = sp.Grid(self.block)
        self.fam_idx = KEYS.index("tenon")
        self.params = {k: fm.FAMILIES[k].defaults() for k in KEYS}
        self.n_samp = 7
        self.lam = 1.5
        self.show_forces = True
        self.land = None                   # (i, j, gi, gj, M, R, signature)
        self.land_mode = 0
        self.land_res = 41
        self.land_height = 1.5
        self.sweep = [0, 1]                # which two parameters the landscape sweeps
        self.msg = ""
        self.refresh()

    # ---------------------------------------------------------------- model
    @property
    def fam(self) -> fm.Family:
        return fm.FAMILIES[KEYS[self.fam_idx]]

    @property
    def p(self) -> dict:
        return self.params[self.fam.key]

    def signature(self):
        """What the landscape depends on besides the two swept parameters."""
        fixed = tuple(v for k, v in self.p.items() if k not in self.swept_names())
        return (self.fam.key, tuple(self.sweep), self.damage, self.statics, self.n_samp, self.land_res, fixed)

    def swept_names(self):
        return [self.fam.params[i].name for i in self.sweep]

    def evaluate(self):
        return fm.evaluate(self.fam, self.p, self.damage, self.statics, self.grid, self.n_samp)

    def compute_landscape(self):
        i, j = self.sweep
        qi, qj = self.fam.params[i], self.fam.params[j]
        gi = np.linspace(qi.lo, qi.hi, self.land_res)
        gj = np.linspace(qj.lo, qj.hi, max(9, int(round(self.land_res * 0.62))))
        M = np.zeros((len(gj), len(gi))); R = np.zeros_like(M)
        for r, vj in enumerate(gj):
            for c, vi in enumerate(gi):
                q = dict(self.p, **{qi.name: float(vi), qj.name: float(vj)})
                M[r, c], R[r, c] = fm.evaluate(self.fam, q, self.damage, self.statics, self.grid, self.n_samp)[:2]
        self.land = (i, j, gi, gj, M, R, self.signature())

    def land_values(self):
        _, _, _, _, M, R, _ = self.land
        if self.land_mode == 0:
            return M, "capacity M_max"
        if self.land_mode == 1:
            return M / max(M.max(), 1e-9) - self.lam * R / max(R.max(), 1e-9), "objective"
        return R, "sound wood removed"

    def land_optimum(self):
        """(value_i, value_j, row, col) of the best cell of the shown landscape."""
        i, j, gi, gj, *_ = self.land
        vals, _ = self.land_values()
        r, c = np.unravel_index(np.argmax(vals if self.land_mode != 2 else -vals), vals.shape)
        return float(gi[c]), float(gj[r]), int(r), int(c)

    # -------------------------------------------------------------- drawing
    def refresh(self):
        M, R, cs, F = self.evaluate()
        if not self.headless:
            import polyscope as ps
            ps.remove_all_structures()
            self.draw_layers(cs, F)
            if self.land is not None:
                self.draw_landscape()
        self.msg = self.readout(M, R, cs)

    def strip_y0(self, li: int) -> float:
        return -li * (self.block.height + STRIP_GAP)

    def draw_layers(self, cs, F):
        import polyscope as ps
        b, d = self.block, self.damage
        layers = self.fam.layers(self.p, b)
        V0, Fq, cx, cy = _quad_grid(0, b.length, 0, b.height, RENDER_NX, RENDER_NY)
        sev = d.severity(V0[:, 0], V0[:, 1], b)
        import shapely
        for li, lay in enumerate(layers):
            y0 = self.strip_y0(li)
            V = V0.copy(); V[:, 1] += y0
            removed = shapely.contains_xy(lay.removed, cx, cy)
            kept = ps.register_surface_mesh(f"L{li} retained wood", V, Fq[~removed], smooth_shade=False)
            kept.add_scalar_quantity("damage severity", sev, cmap="reds", vminmax=(0.0, 1.0), enabled=True)
            kept.set_edge_width(0.0)
            if removed.any():
                new = ps.register_surface_mesh(f"L{li} new wood", V, Fq[removed], smooth_shade=False)
                new.set_color(NEW_WOOD); new.set_edge_width(0.0)
            ys = np.linspace(0, b.height, 40)
            front = np.column_stack([d.front_x(ys, b), ys + y0, np.full(ys.size, 0.01)])
            ps.register_curve_network(f"L{li} decay front", front, "line", radius=0.006).set_color(INK2)
            outline = np.array([[0, y0, 0.01], [b.length, y0, 0.01], [b.length, y0 + b.height, 0.01], [0, y0 + b.height, 0.01]])
            ps.register_curve_network(f"L{li} block outline", outline, "loop", radius=0.008).set_color(INK)
            segs = fm.interface_segments(lay, b)
            if segs:
                P = np.array([[s[0][0], s[0][1] + y0, 0.012] for s in segs] + [[s[1][0], s[1][1] + y0, 0.012] for s in segs])
                E = np.array([[k, k + len(segs)] for k in range(len(segs))])
                ps.register_curve_network(f"L{li} interface", P, E, radius=0.010).set_color(INK)
        if cs:
            P = np.array([[c.point[0], c.point[1] + self.strip_y0(c.layer), 0.02] for c in cs])
            col = np.array([GOOD if c.live else BAD for c in cs])
            pc = ps.register_point_cloud("contacts", P)
            pc.set_radius(0.035, relative=False)
            pc.add_color_quantity("live / dead", col, enabled=True)
            if F is not None and np.any(F):
                vec = np.column_stack([F, np.zeros(len(F))])
                pc.add_vector_quantity("LP contact force", vec, enabled=self.show_forces,
                                       color=ORANGE, length=0.12, radius=0.004)

    def land_origin(self):
        n_layers = len(self.fam.layers(self.p, self.block))
        return (2.0, self.strip_y0(n_layers - 1) + LAND_ORIGIN_Y)

    def to_scene(self, vi, vj, heights=None):
        i, j, gi, gj, *_ = self.land
        ox, oy = self.land_origin()
        x = ox + LAND_SIZE[0] * (vi - gi[0]) / max(gi[-1] - gi[0], 1e-9)
        y = oy + LAND_SIZE[1] * (vj - gj[0]) / max(gj[-1] - gj[0], 1e-9)
        z = 0.0
        if heights is not None:
            r = int(np.argmin(np.abs(gj - vj))); c = int(np.argmin(np.abs(gi - vi)))
            z = float(heights[r, c])
        return x, y, z

    def draw_landscape(self):
        import polyscope as ps
        i, j, gi, gj, M, _, _ = self.land
        vals, label = self.land_values()
        lo, hi = float(vals.min()), float(vals.max())
        H = self.land_height * ((vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals))
        x0, y0 = self.land_origin()
        x1, y1 = x0 + LAND_SIZE[0], y0 + LAND_SIZE[1]
        xs = np.linspace(x0, x1, len(gi)); ys = np.linspace(y0, y1, len(gj))
        X, Y = np.meshgrid(xs, ys)
        V = np.column_stack([X.ravel(), Y.ravel(), H.ravel()])
        na, nl = len(gi), len(gj)
        r, c = np.meshgrid(np.arange(nl - 1), np.arange(na - 1), indexing="ij")
        v00 = r * na + c
        Fq = np.column_stack([v00.ravel(), (v00 + 1).ravel(), (v00 + na + 1).ravel(), (v00 + na).ravel()])
        land = ps.register_surface_mesh("landscape", V, Fq, smooth_shade=True)
        land.add_scalar_quantity(label, vals.ravel(), cmap="blues", enabled=True)
        land.set_edge_width(0.0)
        cliffs = sp.cliff_mask(M, 0.03 * max(M.max(), 1e-9), robust=True).ravel()
        fc = cliffs[Fq].any(axis=1)
        if fc.any():
            Vc = V.copy(); Vc[:, 2] += 0.01 + 0.004 * self.land_height
            cm = ps.register_surface_mesh("landscape cliffs", Vc, Fq[fc], smooth_shade=True)
            cm.set_color(INK); cm.set_transparency(0.55); cm.set_edge_width(0.0)
        base = np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]])
        ps.register_curve_network("landscape frame", base, "loop", radius=0.008).set_color(FRAME)
        ni, nj = self.swept_names()
        here = self.to_scene(self.p[ni], self.p[nj], H)
        pc = ps.register_point_cloud("you are here", np.array([[here[0], here[1], here[2] + 0.04]]))
        pc.set_radius(0.09, relative=False); pc.set_color(AQUA)
        vi, vj, _, _ = self.land_optimum()
        opt = self.to_scene(vi, vj, H)
        po = ps.register_point_cloud("landscape optimum", np.array([[opt[0], opt[1], opt[2] + 0.04]]))
        po.set_radius(0.07, relative=False); po.set_color(ORANGE)

    # -------------------------------------------------------------- readout
    def readout(self, M, R, cs):
        d, st = self.damage, self.statics
        lines = [fm.summary(self.fam, self.p, d, st, self.grid, self.n_samp),
                 f"damage     front x0 = {d.x0:.2f}  slope = {d.slope:.2f}  width = {d.width:.2f}  crit = {d.crit:.2f}",
                 f"statics    mu = {st.mu:.2f}  cap sum(f_n) <= {st.cap:.0f}  ({self.n_samp} samples/face)"]
        if self.land is not None:
            i, j, gi, gj, Ml, Rl, sig = self.land
            stale = sig != self.signature()
            vals, label = self.land_values()
            vi, vj, _, _ = self.land_optimum()
            obj = M / max(Ml.max(), 1e-9) - self.lam * R / max(Rl.max(), 1e-9)
            cliffs = sp.cliff_mask(Ml, 0.03 * max(Ml.max(), 1e-9), robust=True)
            ci, cj = np.nonzero(cliffs)
            ni, nj = self.swept_names()
            r = int(np.argmin(np.abs(gj - self.p[nj]))); c = int(np.argmin(np.abs(gi - self.p[ni])))
            dist = np.min(np.hypot(ci - r, cj - c)) if ci.size else float("nan")
            lines += ["",
                      f"landscape  {ni} x {nj}, {len(gi)} x {len(gj)}" + ("   ** STALE: recompute **" if stale else ""),
                      f"  showing {label}; its optimum (orange) at {ni} = {vi:.2f}, {nj} = {vj:.2f}",
                      f"  objective here = {obj:.3f} (lambda = {self.lam:.2f}); "
                      f"nearest cliff {dist:.0f} cells away; cliffs = {100 * cliffs.mean():.0f}% of cells"]
        else:
            lines += ["", "no landscape yet: press 'compute landscape'"]
        return "\n".join(lines)

    # ------------------------------------------------------------------ GUI
    def gui(self):
        import polyscope.imgui as psim
        changed_any = False
        psim.PushItemWidth(220)

        ch, idx = psim.Combo("family", self.fam_idx, [f"{k}  ({fm.FAMILIES[k].title})" for k in KEYS])
        if ch and idx != self.fam_idx:
            self.fam_idx = idx; self.land = None
            self.sweep = [0, min(1, len(self.fam.params) - 1)]
            changed_any = True
        if self.fam.source:
            psim.TextUnformatted(f"cross-section of MiGumi {self.fam.source}")

        if psim.CollapsingHeader("interface (the output)", True):
            p = dict(self.p)
            for q in self.fam.params:
                ch, v = psim.SliderFloat(f"{q.name}: {q.doc}", p[q.name], q.lo, q.hi); changed_any |= ch
                p[q.name] = v
            self.params[self.fam.key] = self.fam.feasible(p, self.block)
            ch, self.n_samp = psim.SliderInt("samples per face", self.n_samp, 3, 15); changed_any |= ch

        if psim.CollapsingHeader("damage (the input)", True):
            d, b = self.damage, self.block
            ch, x0 = psim.SliderFloat("front x0 (at y = 0)", d.x0, 5.0, b.length); changed_any |= ch
            ch, slope = psim.SliderFloat("front slope (x shift over height)", d.slope, -2.0, 3.0); changed_any |= ch
            ch, width = psim.SliderFloat("front width", d.width, 0.02, 1.5); changed_any |= ch
            ch, crit = psim.SliderFloat("dead above severity", d.crit, 0.05, 0.95); changed_any |= ch
            self.damage = sp.Damage(x0=x0, slope=slope, width=width, crit=crit)

        if psim.CollapsingHeader("statics", True):
            st = self.statics
            ch, mu = psim.SliderFloat("friction mu", st.mu, 0.0, 1.5); changed_any |= ch
            ch, cap = psim.SliderFloat("cap on sum(f_n)", st.cap, 5.0, 400.0); changed_any |= ch
            self.statics = sp.Statics(mu=mu, cap=cap)
            ch, self.show_forces = psim.Checkbox("show LP contact forces", self.show_forces); changed_any |= ch

        if psim.CollapsingHeader("objective and landscape", True):
            names = [q.name for q in self.fam.params]
            ch, self.lam = psim.SliderFloat("lambda (weight on sound wood removed)", self.lam, 0.0, 4.0); changed_any |= ch
            if len(names) > 1:
                ch0, self.sweep[0] = psim.Combo("sweep (horizontal)", self.sweep[0], names)
                ch1, self.sweep[1] = psim.Combo("sweep (vertical)", self.sweep[1], names)
                if (ch0 or ch1) and self.sweep[0] == self.sweep[1]:
                    self.sweep[1] = (self.sweep[0] + 1) % len(names)
                changed_any |= ch0 or ch1
            ch, self.land_mode = psim.Combo("landscape shows", self.land_mode, LAND_MODES); changed_any |= ch
            ch, self.land_res = psim.SliderInt("landscape resolution", self.land_res, 15, 121)
            ch, self.land_height = psim.SliderFloat("landscape relief (0 = flat)", self.land_height, 0.0, 4.0); changed_any |= ch
            if len(names) > 1 and psim.Button("compute landscape"):
                self.compute_landscape(); changed_any = True
            psim.SameLine()
            if psim.Button("reset family defaults"):
                self.params[self.fam.key] = self.fam.defaults(); self.land = None; changed_any = True
            if self.land is not None and psim.Button("go to landscape optimum"):
                vi, vj, _, _ = self.land_optimum()
                ni, nj = self.swept_names()
                self.params[self.fam.key] = self.fam.feasible(dict(self.p, **{ni: vi, nj: vj}), self.block)
                changed_any = True

        psim.PopItemWidth()
        if changed_any:
            self.refresh()
        psim.Separator()
        psim.TextUnformatted(self.msg)


def main():
    if "--headless" in sys.argv[1:]:
        print("sanity checks:", fm.sanity_checks())
        v = Viewer(headless=True)
        for k in KEYS:
            v.fam_idx = KEYS.index(k); v.refresh()
            M, R, cs, _ = v.evaluate()
            print(); print(fm.summary(v.fam, v.p, v.damage, v.statics, v.grid, v.n_samp))
            print(fm.ascii_layers(v.fam, v.p, v.damage, cs, v.block))
        return
    import polyscope as ps
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")
    ps.set_navigation_style("turntable")
    ps.set_view_projection_mode("perspective")
    viewer = Viewer()
    ps.set_user_callback(viewer.gui)
    ps.look_at((6.0, -10.5, 9.0), (6.0, -1.2, 0.3))
    ps.show()


if __name__ == "__main__":
    main()
