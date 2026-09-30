"""Figures for whiting2012-structural-optimization-masonry.md.

  -1  the paper's T example in 2D: the QP force solution on the original and
      on the stretched base (Fig. 3a/b of the paper), computed
  -2  the 2009 tension energy vs the 2012 torque energy as the base is
      stretched (futile) or the overhang is reduced (real fix), computed
  -3  Table 3 (material increase) and Table 1 (gradient timing) of the paper
Run from docs/figures/:  python whiting2012-structural-optimization-masonry.py
"""
from _style import *  # noqa: F401,F403

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.optimize import minimize

SLUG = "whiting2012-structural-optimization-masonry"
MU = 0.7


# --------------------------------------------------------------------------
# 2D rigid-block QP of Eq. (5): min 1/2 f^T H f  s.t. equilibrium, friction,
# fn+, fn- >= 0.  H is diagonal: large weight on tension, small on the rest.
# Then the 2009 energy (sum of squared tension) and the 2012 torque energy
# (Sec. 4: tension above the interface minimum, times the lever arm to the
# hinge line through the minimum-tension vertex) are evaluated on f*.
# --------------------------------------------------------------------------
def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def solve_forces(centroid, W, verts, n, t, mu=MU, w_tension=1.0, w_other=1e-3):
    """One free block on a fixed support; contact vertices `verts` share normal n, tangent t.

    Variables per vertex: (fn+, fn-, ft).  Returns array (m, 3).
    """
    m = len(verts)
    A = np.zeros((3, 3 * m))
    for i, p in enumerate(verts):
        r = np.asarray(p) - np.asarray(centroid)
        A[:, 3 * i] = [n[0], n[1], cross2(r, n)]
        A[:, 3 * i + 1] = [-n[0], -n[1], -cross2(r, n)]
        A[:, 3 * i + 2] = [t[0], t[1], cross2(r, t)]
    w = np.array([0.0, -W, 0.0])
    H = np.tile([w_other, w_tension, w_other], m)
    G = np.zeros((2 * m, 3 * m))
    for i in range(m):
        G[2 * i, 3 * i], G[2 * i, 3 * i + 2] = mu, -1.0
        G[2 * i + 1, 3 * i], G[2 * i + 1, 3 * i + 2] = mu, 1.0
    cons = [{"type": "eq", "fun": lambda x: A @ x + w, "jac": lambda x: A},
            {"type": "ineq", "fun": lambda x: G @ x, "jac": lambda x: G}]
    x0 = np.zeros(3 * m)
    x0[0::3] = W / m
    res = minimize(lambda x: 0.5 * float(np.sum(H * x * x)), x0, jac=lambda x: H * x,
                   bounds=[(0, None), (0, None), (None, None)] * m, constraints=cons,
                   method="SLSQP", options={"ftol": 1e-15, "maxiter": 500})
    x = res.x
    assert np.allclose(A @ x + w, 0, atol=1e-8), "equilibrium residual"
    return x.reshape(m, 3)


def energies(forces, verts):
    """(2009 tension energy, 2012 torque energy, 2012 uniform energy) for one interface.

    2009: y = sum (fn-)^2.
    2012 (Sec. 4.1-4.2): f_min = min tension on the interface; y_uniform = 1/2 * m * f_min^2;
    y_torque = 1/2 * sum ((fn- - f_min) * d_i)^2 with d_i the distance to the hinge line
    through the minimum-tension vertex (2D: a point).
    """
    fm = forces[:, 1]
    y09 = float(np.sum(fm ** 2))
    fmin = fm.min()
    hinge = np.asarray(verts[int(np.argmin(fm))])
    d = np.array([np.linalg.norm(np.asarray(p) - hinge) for p in verts])
    y_torque = 0.5 * float(np.sum(((fm - fmin) * d) ** 2))
    y_uniform = 0.5 * len(verts) * fmin ** 2
    return y09, y_torque, y_uniform


# The T in miniature: a slab of weight W = 1 whose centre of mass is at x = -d,
# resting on a base whose top edge runs from x = 0 to x = L (base = fixed support).
SLAB_W, SLAB_H = 5.0, 0.5
BASE_H = 1.4
D0, L0 = 0.3, 1.0


def t_forces(d, L):
    centroid = (-d, SLAB_H / 2)
    verts = [(0.0, 0.0), (L, 0.0)]
    f = solve_forces(centroid, 1.0, verts, n=(0, 1), t=(1, 0))
    return f, verts


def draw_t(ax, d, L, title):
    f, verts = t_forces(d, L)
    y09, ytq, _ = energies(f, verts)
    x0 = -d - SLAB_W / 2
    ax.add_patch(Rectangle((0, -BASE_H), L, BASE_H, fc=STONE, ec=INK2, lw=1, zorder=2))
    ax.add_patch(Rectangle((x0, 0), SLAB_W, SLAB_H, fc=WOOD, ec=INK2, lw=1, zorder=3))
    fscale = 0.9
    # weight
    ax.annotate("", xy=(-d, SLAB_H / 2 - fscale), xytext=(-d, SLAB_H / 2),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=2, mutation_scale=12), zorder=6)
    ax.text(-d - 0.08, SLAB_H / 2 - 0.55, "W", color=INK2, ha="right", fontsize=10)
    for (px, py), (fp, fm, ft) in zip(verts, f):
        if fp > 1e-6:
            ax.annotate("", xy=(px, py + fp * fscale), xytext=(px, py),
                        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2, mutation_scale=12), zorder=7)
            ax.text(px + 0.07, py + fp * fscale * 0.55, f"{fp:.2f} W", color=INK, fontsize=8.5)
        if fm > 1e-6:
            ax.annotate("", xy=(px, py - fm * fscale), xytext=(px, py),
                        arrowprops=dict(arrowstyle="-|>", color=BAD, lw=2, mutation_scale=12), zorder=7)
            ax.text(px + 0.07, py - fm * fscale * 0.7, f"tension {fm:.2f} W\nlever arm {L:g}", color=INK,
                    fontsize=8.5, va="top")
    # hinge marker and lever arm
    ax.plot([0], [0], marker="o", ms=6, mfc=SURFACE, mec=INK, zorder=8)
    ax.text(0.1, -0.12, "hinge", color=INK2, fontsize=8.5, ha="left", va="top")
    ax.set_title(title + f"\n2009 energy (fₙ⁻)² = {y09:.4f} W²\n2012 torque energy ½(fₙ⁻·L)² = {ytq:.4f} (W·m)²", fontsize=9)
    ax.set_xlim(-3.0, 2.6)
    ax.set_ylim(-BASE_H - 0.1, SLAB_H + 1.0)
    clean(ax, equal=True, hide_axes=True)
    return f


def figure1():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.78, bottom=0.12, wspace=0.05)
    fa = draw_t(axes[0], D0, L0, f"Base L = {L0:g}: right corner pulls with 0.30 W")
    fb = draw_t(axes[1], D0, 2 * L0, f"Base stretched to L = {2 * L0:g}: pull halves, moment does not")
    # closed-form checks: f_L = -dW/L
    assert abs(fa[1, 1] - D0 / L0) < 1e-6 and abs(fb[1, 1] - D0 / (2 * L0)) < 1e-6
    handles = [Line2D([], [], color=BLUE, lw=2, label="compression (solved)"),
               Line2D([], [], color=BAD, lw=2, label="tension needed"),
               Line2D([], [], color=INK2, lw=2, label="weight W at x = −d"),
               Line2D([], [], marker="o", ms=6, mfc=SURFACE, mec=INK, ls="none", label="hinge (min-tension corner)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), columnspacing=1.2)
    fig.suptitle("Stretching the base lowers the tension force but not its moment about the hinge",
                 fontsize=10.5, fontweight="semibold", y=0.97)
    save(fig, f"{SLUG}-1.png")


def figure2():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    f0, v0 = t_forces(D0, L0)
    y09_0, ytq_0, _ = energies(f0, v0)
    # (a) stretch the base to the right, overhang fixed
    Ls = np.linspace(0.5, 3.0, 60)
    e = np.array([energies(*t_forces(D0, L)) for L in Ls])
    ax = axes[0]
    ax.plot(Ls, e[:, 0] / y09_0, color=BLUE, label="2009 energy Σ(fₙ⁻)²")
    ax.plot(Ls, e[:, 1] / ytq_0, color=ORANGE, label="2012 torque energy")
    ax.axvline(L0, color=MUTED, lw=1, ls=":")
    ax.text(L0 + 0.05, 3.3, "start", color=INK2, fontsize=8.5)
    ax.set_xlabel("base length L (m), overhang d = 0.3 fixed")
    ax.set_ylabel("energy / energy at start")
    ax.set_title("Stretch the base (futile):\nonly the 2009 energy drops", fontsize=10)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 4.2)
    clean(ax)
    # (b) move the base's left edge under the load: overhang d shrinks
    ds = np.linspace(0.6, -0.2, 60)
    e = np.array([energies(*t_forces(d, L0)) for d in ds])
    ax = axes[1]
    ax.plot(ds, e[:, 0] / y09_0, color=BLUE, label="2009 energy Σ(fₙ⁻)²")
    ax.plot(ds, e[:, 1] / ytq_0, color=ORANGE, label="2012 torque energy")
    ax.axvline(D0, color=MUTED, lw=1, ls=":")
    ax.text(D0 + 0.02, 3.3, "start", color=INK2, fontsize=8.5)
    ax.set_xlabel("overhang d (m), base length L = 1 fixed")
    ax.set_ylabel("energy / energy at start")
    ax.set_title("Move support under the load (real fix):\nboth energies reach zero at d = 0", fontsize=10)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 4.2)
    ax.invert_xaxis()
    clean(ax)
    # check the closed forms: y09 = (d/L)^2, y_torque = 1/2 d^2  (W = 1)
    d, L = 0.45, 1.7
    y09, ytq, _ = energies(*t_forces(d, L))
    assert abs(y09 - (d / L) ** 2) < 1e-8 and abs(ytq - 0.5 * d ** 2) < 1e-8
    fig.tight_layout()
    save(fig, f"{SLUG}-2.png")


def figure3():
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    # Table 3: weight increase after reaching feasibility, new energy vs Whiting 2009
    models = ["T", "arch", "stack", "shell", "slabs", "buttress"]
    new = [105, 9.9, 148, 256, 180, 34.5]
    old = [129, 31.1, 189, 367, 500, 66.7]
    ax = axes[0]
    y = np.arange(len(models))
    ax.barh(y - 0.19, old, height=0.36, color=ORANGE, label="2009 tension energy", zorder=3)
    ax.barh(y + 0.19, new, height=0.36, color=BLUE, label="2012 torque energy", zorder=3)
    for i, (a, b) in enumerate(zip(old, new)):
        ax.text(a + 8, i - 0.19, f"{a:g}%", va="center", fontsize=8, color=INK)
        ax.text(b + 8, i + 0.19, f"{b:g}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, models)
    ax.invert_yaxis()
    ax.set_xlim(0, 600)
    ax.set_xlabel("weight added to the modified blocks (%)")
    ax.set_title("Torque energy reaches feasibility with\nless material in every model (Table 3)", fontsize=10)
    ax.legend(loc="lower right")
    clean(ax)
    # Table 1: analytic gradient vs finite differences
    labels = ["arch (12)", "stack (8)", "shell (24)", "torus (232)", "torus (400)", "torus (576)"]
    analytic = [0.41, 0.36, 1.19, 80.6, 138.6, 720.6]
    fd = [127, 84.5, 269, 8.37e3, 2.28e4, 4.95e4]
    ax = axes[1]
    y = np.arange(len(labels))
    ax.barh(y - 0.19, fd, height=0.36, color=ORANGE, label="finite differences", zorder=3)
    ax.barh(y + 0.19, analytic, height=0.36, color=BLUE, label="analytic (this paper)", zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 3e5)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("time for one full gradient (s, log scale)")
    ax.set_title("Closed-form gradient is roughly 70–300× faster\nthan finite differences (Table 1)", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    clean(ax)
    fig.tight_layout()
    save(fig, f"{SLUG}-3.png")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
