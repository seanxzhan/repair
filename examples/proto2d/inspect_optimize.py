"""Polyscope playground for Step 4a: watch each route climb the repair objective.

    python examples/proto2d/inspect_optimize.py                 # opens on the tenon, first held-out field
    python examples/proto2d/inspect_optimize.py --headless      # one run per route, printed

Pick a family and a held-out field, set lambda and the required load, choose a
start (random, or the sliders), and run a route: gradient ascent through the
surrogate, on the LP with softened contact flags, or with finite differences
on the hard LP; or the reference search on the hard LP. The block shows the
interface at the current step with live/dead contacts and the LP's forces;
a step slider scrubs the trajectory; the readout gives the route's own
estimate against the hard LP's score at every step, and which parameters sit
on a bound. Uses repair.proto2d.optimize; the comparison table is
examples/proto2d/compare_optimizers.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

try:
    from repair.proto2d import dataset as ds, families as fm, model as sp, optimize as opt
    from repair.proto2d.train import load_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from repair.proto2d import dataset as ds, families as fm, model as sp, optimize as opt
    from repair.proto2d.train import load_model

from inspect_splice import BAD, GOOD, INK, INK2, NEW_WOOD, ORANGE, RENDER_NX, RENDER_NY, _quad_grid

ROUTES = ["surrogate", "soft_lp", "hard_fd", "reference"]


class Viewer:
    def __init__(self, data="out/proto2d", models="out/proto2d/models", headless=False):
        self.headless = headless
        self.fields, self.fams = ds.load(data)
        self.models_dir = Path(models)
        self.keys = sorted(p.parent.name for p in self.models_dir.glob("*/model.pt"))
        self.b, self.grid, self.st = sp.Block(), sp.Grid(), sp.Statics()
        self.test_fields = [int(x) for x in np.unique(self.fams[self.keys[0]]["field"][self.fams[self.keys[0]]["test"]])]
        self.fam_idx, self.field_idx, self.route_idx = self.keys.index("tenon") if "tenon" in self.keys else 0, 0, 0
        self.lam, self.m_req, self.tau = 1.5, 0.0, 0.1
        self.steps, self.lr, self.h = 80, 0.05, 0.02
        self.seed = 0
        self.models = {}
        self.run = None; self.step = 0
        self.msg = ""
        self.load()
        self.x = self.prob.project(np.random.default_rng(self.seed).random(len(self.prob.names)))
        self.refresh()

    # ---------------------------------------------------------------- model
    def load(self):
        key = self.keys[self.fam_idx]
        if key not in self.models:
            self.models[key] = load_model(self.models_dir / key / "model.pt", "cpu")
        model, ck = self.models[key]
        fi = self.test_fields[self.field_idx]
        f = ds.field_of(self.fields, fi, self.b)
        raster = torch.tensor(self.fields["raster"][fi].astype(np.float32)).unsqueeze(0)
        obj = opt.Objective(lam=self.lam, m_req=self.m_req, m_ref=ck["normalization"]["peak"])
        self.prob = opt.Problem(fm.FAMILIES[key], f, obj, model, ck, raster, "cpu", tau=self.tau)
        self.field = f

    def evaluate(self, x):
        p = self.prob.denorm(x)
        M, R, cs, F = fm.evaluate(self.prob.fam, p, self.field, self.st, self.grid)
        return p, M, R, cs, F

    def do_run(self):
        route = ROUTES[self.route_idx]
        self.prob.lp_calls = 0
        if route == "reference":
            self.run = opt.reference(self.prob, 800, 2, seed=self.seed)
        else:
            self.run = opt.ascend(self.prob, route, self.x, self.steps, self.lr, self.h)
        self.step = len(self.run.traj) - 1
        self.x = self.run.traj[self.step][0]

    # -------------------------------------------------------------- drawing
    def refresh(self):
        p, M, R, cs, F = self.evaluate(self.x)
        if not self.headless:
            import polyscope as ps
            ps.remove_all_structures()
            self.draw_block(p, cs, F)
        self.msg = self.readout(p, M, R, cs)

    def draw_block(self, p, cs, F):
        import polyscope as ps, shapely
        b, d = self.b, self.field
        V, Fq, cx, cy = _quad_grid(0, b.length, 0, b.height, RENDER_NX, RENDER_NY)
        sev = d.severity(V[:, 0], V[:, 1], b)
        layers = self.prob.fam.layers(p, b)
        removed = np.zeros(len(cx), bool)
        for lay in layers:
            removed |= shapely.contains_xy(lay.removed, cx, cy)
        kept = ps.register_surface_mesh("retained wood", V, Fq[~removed], smooth_shade=False)
        kept.add_scalar_quantity("damage severity", sev, cmap="reds", vminmax=(0.0, 1.0), enabled=True); kept.set_edge_width(0.0)
        if removed.any():
            new = ps.register_surface_mesh("new wood", V, Fq[removed], smooth_shade=False); new.set_color(NEW_WOOD); new.set_edge_width(0.0)
        fp = d.front_points() if hasattr(d, "front_points") else None
        if fp is not None and len(fp):
            pc = ps.register_point_cloud("crit contour", np.column_stack([fp, np.full(len(fp), 0.015)])); pc.set_radius(0.012, relative=False); pc.set_color(INK2)
        segs = [s for lay in layers for s in fm.interface_segments(lay, b)]
        if segs:
            P = np.array([[s[0][0], s[0][1], 0.012] for s in segs] + [[s[1][0], s[1][1], 0.012] for s in segs])
            E = np.array([[k, k + len(segs)] for k in range(len(segs))])
            ps.register_curve_network("interface", P, E, radius=0.005).set_color(INK)
        if cs:
            P = np.array([[c.point[0], c.point[1], 0.02] for c in cs])
            pc = ps.register_point_cloud("contacts", P); pc.set_radius(0.035, relative=False)
            pc.add_color_quantity("live / dead", np.array([GOOD if c.live else BAD for c in cs]), enabled=True)
            if F is not None and np.any(F):
                pc.add_vector_quantity("LP contact force", np.column_stack([F, np.zeros(len(F))]), enabled=True, color=ORANGE, length=0.12, radius=0.004)
        # the trajectory, as the shoulder's path drawn under the block
        if self.run and len(self.run.traj) > 1:
            xs = np.array([self.prob.denorm(t[0])["a"] for t in self.run.traj])
            ys = -0.4 - 0.08 * np.arange(len(xs)) / max(1, len(xs) - 1) * 10
            pts = np.column_stack([xs, ys, np.zeros(len(xs))])
            ps.register_curve_network("trajectory (shoulder position vs step)", pts, "line", radius=0.004).set_color(ORANGE)

    def readout(self, p, M, R, cs):
        prob, obj = self.prob, self.prob.obj
        key, fi = self.keys[self.fam_idx], self.test_fields[self.field_idx]
        lines = [f"{key}  on held-out field {fi}   ({len(cs)} contacts, {sum(c.live for c in cs)} live)",
                 "  " + "  ".join(f"{k}={v:.2f}" for k, v in p.items()) + f"   on bound: {prob.on_bound(self.x) or 'none'}",
                 f"  hard LP: M = {M:7.2f}   sound removed = {R:.3f}   objective = {obj(M, R):.3f}"]
        M_nn = prob.surrogate(self.x)[0]; M_soft = prob.soft(self.x)
        lines.append(f"  surrogate says M = {M_nn:7.2f}   soft LP (tau {self.tau}) says M = {M_soft:7.2f}")
        if self.run:
            r = self.run
            lines += ["", f"run: {r.route}   step {self.step}/{len(r.traj) - 1}   LP calls {r.lp_calls}",
                      f"  end: true objective {r.J_true:.3f}  (own estimate {r.J_est:.3f})   M {r.M_true:.1f}  R {r.R_true:.2f}   on bound: {r.on_bound or 'none'}"]
            if r.route != "reference":
                J0 = obj(*prob.hard(r.traj[0][0]))
                lines.append(f"  start objective {J0:.3f}  ->  gain {r.J_true - J0:+.3f}")
        else:
            lines += ["", "no run yet: pick a route and press 'run'"]
        return "\n".join(lines)

    # ------------------------------------------------------------------ GUI
    def gui(self):
        import polyscope.imgui as psim
        changed = False; reload = False
        psim.PushItemWidth(220)
        ch, self.fam_idx = psim.Combo("family", self.fam_idx, self.keys); reload |= ch
        ch, self.field_idx = psim.Combo("held-out field", self.field_idx, [str(f) for f in self.test_fields]); reload |= ch
        if psim.CollapsingHeader("objective", True):
            ch, self.lam = psim.SliderFloat("lambda (sound wood removed)", self.lam, 0.0, 4.0); reload |= ch
            ch, self.m_req = psim.SliderFloat("required capacity", self.m_req, 0.0, 200.0); reload |= ch
            ch, self.tau = psim.SliderFloat("tau (soft-LP contact softness)", self.tau, 0.01, 0.5); reload |= ch
        if reload:
            self.load(); self.run = None; self.x = self.prob.project(self.x); changed = True
        if psim.CollapsingHeader("interface (normalized parameters)", True):
            x = self.x.copy()
            for i, q in enumerate(self.prob.fam.params):
                ch, x[i] = psim.SliderFloat(f"{q.name}: {q.doc}", float(x[i]), 0.0, 1.0); changed |= ch
            if changed:
                self.x = self.prob.project(x); self.run = None
            if psim.Button("random start"):
                self.seed += 1; self.x = self.prob.project(np.random.default_rng(self.seed).random(len(self.prob.names))); self.run = None; changed = True
        if psim.CollapsingHeader("route", True):
            _, self.route_idx = psim.Combo("route", self.route_idx, ROUTES)
            _, self.steps = psim.SliderInt("steps", self.steps, 5, 300)
            _, self.lr = psim.SliderFloat("learning rate", self.lr, 0.005, 0.2)
            _, self.h = psim.SliderFloat("finite-difference half-step", self.h, 0.002, 0.1)
            if psim.Button("run"):
                self.do_run(); changed = True
            if self.run and len(self.run.traj) > 1:
                ch, self.step = psim.SliderInt("step (scrub the trajectory)", self.step, 0, len(self.run.traj) - 1)
                if ch:
                    self.x = self.run.traj[self.step][0]; changed = True
        psim.PopItemWidth()
        if changed:
            self.refresh()
        psim.Separator(); psim.TextUnformatted(self.msg)


def main():
    if "--headless" in sys.argv[1:]:
        v = Viewer(headless=True)
        for i, route in enumerate(ROUTES):
            v.route_idx = i; v.do_run(); v.refresh(); print(v.msg); print()
        return
    import polyscope as ps
    ps.init(); ps.set_ground_plane_mode("none"); ps.set_up_dir("z_up"); ps.set_navigation_style("turntable"); ps.set_view_projection_mode("perspective")
    viewer = Viewer()
    ps.set_user_callback(viewer.gui)
    ps.look_at((6.0, -8.0, 7.0), (6.0, 0.0, 0.3))
    ps.show()


if __name__ == "__main__":
    main()
