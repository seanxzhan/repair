"""Polyscope playground for the 2D repair-interface model behind docs/proposal.md.

One block, partly rotten, is the input. The interface -- a mortise-and-tenon
splice with a shoulder at `a`, a tenon of length `ell` between two cheeks -- is
the output, and every one of its parameters is a slider here. So are the damage
front, friction and the force cap. On each change the contact set is rebuilt and
the frictional-equilibrium LP re-solved, and you see which contacts are live
(green), which are dead against rot (red), the forces the LP puts on them, the
moment the interface can transfer, and the sound wood it throws away.

    python examples/inspect_splice.py              # open the viewer
    python examples/inspect_splice.py --headless   # print the readouts, no window

"compute landscape" sweeps (a, ell) with everything else held at the current
sliders and draws the result below the block: capacity, the objective, or the
sound wood removed, with the active-set cliffs overlaid in black and two
markers -- where you are, and the landscape's optimum. "go to landscape
optimum" moves the (a, ell) sliders onto that orange marker. Moving a damage
or statics slider afterwards marks it stale until you recompute.

The model is `repair.splice2d`; docs/figures/proposal.py is the same model
rendered once, at the defaults this viewer starts from.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    from repair import splice2d as sp
except ImportError:                       # not installed: fall back to the source tree
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from repair import splice2d as sp

# Colours as (r, g, b) in [0, 1], matching docs/figures/_style.py.
GOOD = (0.05, 0.64, 0.05)
BAD = (0.82, 0.23, 0.23)
INK = (0.04, 0.04, 0.04)
INK2 = (0.32, 0.32, 0.30)
AQUA = (0.11, 0.69, 0.48)
ORANGE = (0.92, 0.41, 0.20)
NEW_WOOD = (0.97, 0.95, 0.93)
FRAME = (0.55, 0.54, 0.52)

RENDER_NX, RENDER_NY = 240, 40            # the block is drawn as a clipped quad grid
LAND_ORIGIN = (2.0, -3.6)                 # where the landscape is drawn, below the block
LAND_SIZE = (8.0, 2.4)                    # (a extent, ell extent) in scene units
LAND_MODES = ["capacity", "objective", "sound wood removed"]


def _quad_grid(x0, x1, y0, y1, nx, ny, z=0.0):
    """Vertices and quad faces of an (nx x ny) grid, plus cell centres."""
    xs, ys = np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1)
    X, Y = np.meshgrid(xs, ys)
    V = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    i, j = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    v00 = i * (nx + 1) + j
    F = np.column_stack([v00.ravel(), (v00 + 1).ravel(), (v00 + nx + 2).ravel(), (v00 + nx + 1).ravel()])
    cx = 0.5 * (xs[:-1] + xs[1:]); cy = 0.5 * (ys[:-1] + ys[1:])
    CX, CY = np.meshgrid(cx, cy)
    return V, F, CX.ravel(), CY.ravel()


class Viewer:
    def __init__(self, headless=False):
        self.headless = headless
        self.reset()
        self.land = None                  # (a_grid, l_grid, M, R, signature)
        self.land_mode = 0
        self.land_res = 49
        self.land_height = 1.5            # relief of the landscape, in scene units; 0 is flat
        self.show_forces = True
        self.msg = ""
        self.grid = sp.Grid(self.block)
        self.refresh()

    def reset(self):
        self.block, self.damage, self.statics = sp.Block(), sp.Damage(), sp.Statics()
        self.splice = sp.Splice()
        self.lam = 1.5

    # ---------------------------------------------------------------- model
    def signature(self):
        """What the landscape depends on besides (a, ell)."""
        s = self.splice
        return (self.damage, self.statics, s.y1, s.y2, s.n_samp, self.land_res)

    def evaluate(self):
        cs = sp.contact_set(self.splice, self.damage, self.block)
        M, forces = sp.solve_moment(cs, self.statics)
        R = self.grid.sound_removed(self.splice, self.damage)
        return cs, M, forces, R

    def compute_landscape(self):
        n = self.land_res
        a_grid = np.linspace(7.5, 11.5, n)
        l_grid = np.linspace(0.0, 3.5, max(9, int(round(n * 0.62))))
        M, R = sp.landscape(a_grid, l_grid, self.splice, self.damage, self.statics, self.grid)
        self.land = (a_grid, l_grid, M, R, self.signature())

    # -------------------------------------------------------------- drawing
    def refresh(self):
        cs, M, forces, R = self.evaluate()
        if not self.headless:
            import polyscope as ps
            ps.remove_all_structures()
            self.draw_block()
            self.draw_contacts(cs, forces)
            if self.land is not None:
                self.draw_landscape()
        self.msg = self.readout(cs, M, R)

    def draw_block(self):
        import polyscope as ps
        b, s, d = self.block, self.splice, self.damage
        V, F, cx, cy = _quad_grid(0, b.length, 0, b.height, RENDER_NX, RENDER_NY)
        removed = (cx > s.a) | ((cx > s.a - s.ell) & (cy > s.y1) & (cy < s.y2))
        sev = d.severity(V[:, 0], V[:, 1], b)

        kept = ps.register_surface_mesh("retained wood", V, F[~removed], smooth_shade=False)
        kept.add_scalar_quantity("damage severity", sev, cmap="reds", vminmax=(0.0, 1.0), enabled=True)
        kept.set_edge_width(0.0)
        new = ps.register_surface_mesh("new wood", V, F[removed], smooth_shade=False)
        new.set_color(NEW_WOOD); new.set_edge_width(0.0)

        ys = np.linspace(0, b.height, 40)
        front = np.column_stack([d.front_x(ys, b), ys, np.full(ys.size, 0.01)])
        ps.register_curve_network("decay front", front, "line", radius=0.006).set_color(INK2)

        outline = np.array([[0, 0, 0.01], [b.length, 0, 0.01], [b.length, b.height, 0.01], [0, b.height, 0.01]])
        ps.register_curve_network("block outline", outline, "loop", radius=0.008).set_color(INK)

        # the interface itself, as a polyline
        xb = s.a - s.ell
        iface = np.array([[s.a, 0], [s.a, s.y1], [xb, s.y1], [xb, s.y2], [s.a, s.y2], [s.a, b.height]])
        iface = np.column_stack([iface, np.full(len(iface), 0.012)])
        ps.register_curve_network("interface", iface, "line", radius=0.010).set_color(INK)

    def draw_contacts(self, cs, forces):
        import polyscope as ps
        P = np.array([[c.point[0], c.point[1], 0.02] for c in cs])
        col = np.array([GOOD if c.live else BAD for c in cs])
        pc = ps.register_point_cloud("contacts", P)
        pc.set_radius(0.035, relative=False)
        pc.add_color_quantity("live / dead", col, enabled=True)
        if forces is not None and np.any(forces):
            vec = np.column_stack([forces, np.zeros(len(forces))])
            # length and radius are relative to the scene scale (polyscope's convention)
            pc.add_vector_quantity("LP contact force", vec, enabled=self.show_forces,
                                   color=ORANGE, length=0.12, radius=0.004)

    def land_values(self):
        a_grid, l_grid, M, R, _ = self.land
        if self.land_mode == 0:
            return M, "capacity M_max"
        if self.land_mode == 1:
            return M / max(M.max(), 1e-9) - self.lam * R / max(R.max(), 1e-9), "objective"
        return R, "sound wood removed"

    def land_optimum(self):
        """(a, ell, i, j) of the best cell of the shown landscape: the argmax of
        capacity or the objective, the argmin of sound wood removed."""
        a_grid, l_grid = self.land[0], self.land[1]
        vals, _ = self.land_values()
        i, j = np.unravel_index(np.argmax(vals if self.land_mode != 2 else -vals), vals.shape)
        return float(a_grid[j]), float(l_grid[i]), int(i), int(j)

    def land_heights(self, vals):
        """z of the landscape surface: relief proportional to the shown value."""
        lo, hi = float(vals.min()), float(vals.max())
        rel = (vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals)
        return self.land_height * rel

    def to_scene(self, a, ell, heights=None):
        """Map an interface (a, ell) to the landscape's scene position (x, y, z)."""
        a_grid, l_grid = self.land[0], self.land[1]
        x = LAND_ORIGIN[0] + LAND_SIZE[0] * (a - a_grid[0]) / (a_grid[-1] - a_grid[0])
        y = LAND_ORIGIN[1] + LAND_SIZE[1] * (ell - l_grid[0]) / (l_grid[-1] - l_grid[0])
        z = 0.0
        if heights is not None:              # sit on the surface at the nearest grid node
            i = int(np.clip(np.argmin(np.abs(l_grid - ell)), 0, len(l_grid) - 1))
            j = int(np.clip(np.argmin(np.abs(a_grid - a)), 0, len(a_grid) - 1))
            z = float(heights[i, j])
        return x, y, z

    def draw_landscape(self):
        import polyscope as ps
        a_grid, l_grid, M, _, _ = self.land
        vals, label = self.land_values()
        H = self.land_heights(vals)
        x0, y0 = LAND_ORIGIN
        x1, y1 = x0 + LAND_SIZE[0], y0 + LAND_SIZE[1]

        # One vertex per grid node so the surface is a continuous height field;
        # values, heights and colours all live on vertices.
        xs = np.linspace(x0, x1, len(a_grid)); ys = np.linspace(y0, y1, len(l_grid))
        X, Y = np.meshgrid(xs, ys)
        V = np.column_stack([X.ravel(), Y.ravel(), H.ravel()])
        na, nl = len(a_grid), len(l_grid)
        i, j = np.meshgrid(np.arange(nl - 1), np.arange(na - 1), indexing="ij")
        v00 = i * na + j
        F = np.column_stack([v00.ravel(), (v00 + 1).ravel(), (v00 + na + 1).ravel(), (v00 + na).ravel()])
        land = ps.register_surface_mesh("landscape", V, F, smooth_shade=True)
        land.add_scalar_quantity(label, vals.ravel(), cmap="blues", enabled=True)
        land.set_edge_width(0.0)

        # cliffs: a face is a cliff if any of its corner nodes is; lifted a hair
        cliffs = sp.cliff_mask(M, 0.03 * max(M.max(), 1e-9), robust=True).ravel()
        fc = cliffs[F].any(axis=1)
        if fc.any():
            Vc = V.copy(); Vc[:, 2] += 0.01 + 0.004 * self.land_height
            c = ps.register_surface_mesh("landscape cliffs", Vc, F[fc], smooth_shade=True)
            c.set_color(INK); c.set_transparency(0.55); c.set_edge_width(0.0)

        # base frame, plus a drop line at each corner if there is relief
        base = np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]])
        ps.register_curve_network("landscape frame", base, "loop", radius=0.008).set_color(FRAME)
        if self.land_height > 0:
            top = base.copy(); top[:, 2] = self.land_height
            posts = np.vstack([base, top]); edges = np.array([[k, k + 4] for k in range(4)])
            ps.register_curve_network("landscape posts", posts, edges, radius=0.005).set_color(FRAME)

        here = self.to_scene(self.splice.a, self.splice.ell, H)
        pc = ps.register_point_cloud("you are here", np.array([[here[0], here[1], here[2] + 0.04]]))
        pc.set_radius(0.09, relative=False); pc.set_color(AQUA)
        a_opt, l_opt, _, _ = self.land_optimum()
        opt = self.to_scene(a_opt, l_opt, H)
        po = ps.register_point_cloud("landscape optimum", np.array([[opt[0], opt[1], opt[2] + 0.04]]))
        po.set_radius(0.07, relative=False); po.set_color(ORANGE)

    # -------------------------------------------------------------- readout
    def readout(self, cs, M, R):
        s, d, st = self.splice, self.damage, self.statics
        obj = None
        lines = [f"interface  a = {s.a:.2f}  ell = {s.ell:.2f}  cheeks [{s.y1:.2f}, {s.y2:.2f}]"
                 f"  ({s.n_samp} samples/face)",
                 f"damage     front x0 = {d.x0:.2f}  slope = {d.slope:.2f}  width = {d.width:.2f}"
                 f"  crit = {d.crit:.2f}",
                 f"statics    mu = {st.mu:.2f}  cap sum(f_n) <= {st.cap:.0f}", ""]
        live = sum(c.live for c in cs)
        lines.append(f"M_max = {M:7.2f}      sound wood removed = {R:.3f}      "
                     f"{live}/{len(cs)} contacts live")
        per = []
        for fi, name in enumerate(sp.FACES):
            n = sum(1 for c in cs if c.face == fi)
            if n:
                per.append(f"{name} {sum(1 for c in cs if c.face == fi and c.live)}/{n}")
        lines.append("  " + "   ".join(per))
        if self.land is not None:
            a_grid, l_grid, Ml, Rl, sig = self.land
            stale = sig != self.signature()
            _, label = self.land_values()
            a_opt, l_opt, _, _ = self.land_optimum()
            obj = M / max(Ml.max(), 1e-9) - self.lam * R / max(Rl.max(), 1e-9)
            cliffs = sp.cliff_mask(Ml, 0.03 * max(Ml.max(), 1e-9), robust=True)
            ci, cj = np.nonzero(cliffs)
            i = int(np.argmin(np.abs(l_grid - s.ell))); j = int(np.argmin(np.abs(a_grid - s.a)))
            dist = np.min(np.hypot(ci - i, cj - j)) if ci.size else float("nan")
            lines += ["",
                      f"landscape  {len(a_grid)} x {len(l_grid)}, a in [{a_grid[0]:.1f}, {a_grid[-1]:.1f}], "
                      f"ell in [{l_grid[0]:.1f}, {l_grid[-1]:.1f}]" + ("   ** STALE: recompute **" if stale else ""),
                      f"  showing {label}; its optimum (orange) at a = {a_opt:.2f}, ell = {l_opt:.2f}",
                      f"  objective here = {obj:.3f} (lambda = {self.lam:.2f}); "
                      f"nearest cliff {dist:.0f} cells away; cliffs = {100 * cliffs.mean():.0f}% of cells"]
        else:
            lines += ["", "no landscape yet: press 'compute landscape'"]
        return "\n".join(lines)

    # ------------------------------------------------------------------ GUI
    def gui(self):
        import polyscope.imgui as psim
        s, d, st = self.splice, self.damage, self.statics
        b = self.block
        changed_any = False
        psim.PushItemWidth(220)

        if psim.CollapsingHeader("interface (the output)", True):
            ch, a = psim.SliderFloat("shoulder a", s.a, 6.0, b.length - 0.2); changed_any |= ch
            ch, ell = psim.SliderFloat("tenon length ell", s.ell, 0.0, 4.0); changed_any |= ch
            ell = min(ell, a - 0.2)
            center, thick = 0.5 * (s.y1 + s.y2), s.y2 - s.y1
            ch, center = psim.SliderFloat("cheek centre", center, 0.1, b.height - 0.1); changed_any |= ch
            ch, thick = psim.SliderFloat("tenon thickness", thick, 0.05, b.height - 0.1); changed_any |= ch
            thick = min(thick, 2 * min(center, b.height - center) - 0.02)
            ch, n = psim.SliderInt("samples per face", s.n_samp, 3, 15); changed_any |= ch
            self.splice = replace(s, a=a, ell=ell, n_samp=n).with_cheeks(center, thick)

        if psim.CollapsingHeader("damage (the input)", True):
            ch, x0 = psim.SliderFloat("front x0 (at y = 0)", d.x0, 5.0, b.length); changed_any |= ch
            ch, slope = psim.SliderFloat("front slope (x shift over height)", d.slope, -2.0, 3.0); changed_any |= ch
            ch, width = psim.SliderFloat("front width", d.width, 0.02, 1.5); changed_any |= ch
            ch, crit = psim.SliderFloat("dead above severity", d.crit, 0.05, 0.95); changed_any |= ch
            self.damage = sp.Damage(x0=x0, slope=slope, width=width, crit=crit)

        if psim.CollapsingHeader("statics", True):
            ch, mu = psim.SliderFloat("friction mu", st.mu, 0.0, 1.5); changed_any |= ch
            ch, cap = psim.SliderFloat("cap on sum(f_n)", st.cap, 5.0, 400.0); changed_any |= ch
            self.statics = sp.Statics(mu=mu, cap=cap)
            ch, self.show_forces = psim.Checkbox("show LP contact forces", self.show_forces); changed_any |= ch

        if psim.CollapsingHeader("objective and landscape", True):
            ch, self.lam = psim.SliderFloat("lambda (weight on sound wood removed)", self.lam, 0.0, 4.0)
            changed_any |= ch
            ch, self.land_mode = psim.Combo("landscape shows", self.land_mode, LAND_MODES); changed_any |= ch
            ch, self.land_res = psim.SliderInt("landscape resolution (a)", self.land_res, 15, 121)
            ch, self.land_height = psim.SliderFloat("landscape relief (0 = flat)", self.land_height, 0.0, 4.0)
            changed_any |= ch
            if psim.Button("compute landscape (a x ell)"):
                self.compute_landscape(); changed_any = True
            psim.SameLine()
            if psim.Button("reset to Figure 1"):
                self.reset(); self.land = None; changed_any = True
            if self.land is not None:
                # Jump (a, ell) to the orange marker; cheeks and everything else stay put.
                if psim.Button("go to landscape optimum"):
                    a_opt, l_opt, _, _ = self.land_optimum()
                    self.splice = replace(self.splice, a=a_opt, ell=l_opt); changed_any = True

        psim.PopItemWidth()
        if changed_any:
            self.refresh()
        psim.Separator()
        psim.TextUnformatted(self.msg)


def main():
    if "--headless" in sys.argv[1:]:
        v = Viewer(headless=True)
        print(v.msg)
        v.land_res = 25
        v.compute_landscape(); v.refresh()
        print(); print(v.msg)
        return
    import polyscope as ps
    ps.init()
    ps.set_ground_plane_mode("none")
    # The block lies flat in the xy plane; the landscape's relief rises in z.
    # Set relief to 0 in the panel for the flat, map-like view.
    ps.set_up_dir("z_up")
    ps.set_navigation_style("turntable")
    ps.set_view_projection_mode("perspective")
    viewer = Viewer()
    ps.set_user_callback(viewer.gui)
    ps.look_at((6.0, -10.5, 9.0), (6.0, -1.2, 0.3))
    ps.show()


if __name__ == "__main__":
    main()
