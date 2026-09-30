"""Figures for wang2025-learning-to-assemble.md.

Figure 1 runs the paper's held-part stability check on a 5-voussoir arch for
two disassembly orders (in the spirit of the paper's Fig. 2). The verdict is the
feasibility of the equilibrium constraints (linprog); for an infeasible state
the minimum-residual QP of Sec. "The stability check" is solved (SLSQP) and the
leftover motion v = M^-1 r is drawn. Figures 2 and 3 chart Tables 2 and 3.
Run from docs/figures/:  python wang2025-learning-to-assemble.py
"""
from _style import *  # noqa: F401,F403

import numpy as np
from matplotlib.patches import Polygon, Rectangle
from scipy.optimize import linprog, minimize

SLUG = "wang2025-learning-to-assemble"
GRAV = 9.81
MU = 0.55          # the paper's coefficient
RI, RO = 2.0, 2.7  # arch intrados / extrados radius (m)
N = 5              # voussoirs P1..P5 spanning 0..180 degrees
LIMIT = 1e5        # the paper's lambda^u / f^u bound

# ------------------------------------------------------------------ geometry
def sector(i, npts=8):
    a0, a1 = np.pi * i / N, np.pi * (i + 1) / N
    ang = np.linspace(a0, a1, npts)
    outer = np.c_[RO * np.cos(ang), RO * np.sin(ang)]
    inner = np.c_[RI * np.cos(ang[::-1]), RI * np.sin(ang[::-1])]
    return np.vstack([outer, inner])


def poly_props(pts, rho=1.0):
    """Area, centroid and polar second moment of a polygon (shoelace formulas)."""
    x, y = pts[:, 0], pts[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    A = 0.5 * cr.sum()
    cx = (x + xn) @ cr / (6 * A); cy = (y + yn) @ cr / (6 * A)
    Ix = ((y**2 + y * yn + yn**2) @ cr) / 12; Iy = ((x**2 + x * xn + xn**2) @ cr) / 12
    Ic = rho * (Ix + Iy) - rho * A * (cx**2 + cy**2)
    return rho * A, np.array([cx, cy]), Ic


PARTS = [sector(i, 40) for i in range(N)]
MASS, COMS, INERT = zip(*[poly_props(p) for p in PARTS])
MASS, INERT = np.array(MASS), np.array(INERT)

# Joints j = 0..N at angle pi*j/N. Joint 0 sits on the left... no: joint 0 is at angle 0
# (right abutment), joint N at angle pi (left abutment). Contact k: (part A, part B, point,
# normal into B, tangent). A = j-1 or "R" abutment, B = j or "L" abutment.
CONTACTS = []
for j in range(N + 1):
    th = np.pi * j / N
    nrm = np.array([-np.sin(th), np.cos(th)])      # into the part at larger angle
    tan = np.array([np.cos(th), np.sin(th)])       # radial (sliding direction)
    A = "R" if j == 0 else j - 1
    B = "L" if j == N else j
    for r in (RI, RO):
        CONTACTS.append((A, B, r * np.array([np.cos(th), np.sin(th)]), nrm, tan))


def wrench(part, point, force):
    r = point - COMS[part]
    return np.array([force[0], force[1], r[0] * force[1] - r[1] * force[0]])


def build(present, held):
    """Rows of the residual r = J^T lambda + g for the parts that are present and not held.
    Columns: per active contact (lambda_n, lambda_t+, lambda_t-)."""
    rows = [p for p in present if p not in held]
    active = [k for k, (A, B, *_) in enumerate(CONTACTS)
              if (A in present or A in ("L", "R")) and (B in present or B in ("L", "R"))]
    J = np.zeros((3 * len(rows), 3 * len(active)))
    g = np.zeros(3 * len(rows))
    for i, p in enumerate(rows):
        g[3 * i + 1] = -MASS[p] * GRAV
        for c, k in enumerate(active):
            A, B, pt, nrm, tan = CONTACTS[k]
            for part, sgn in ((B, 1.0), (A, -1.0)):
                if part != p:
                    continue
                J[3 * i:3 * i + 3, 3 * c] += sgn * wrench(p, pt, nrm)
                J[3 * i:3 * i + 3, 3 * c + 1] += sgn * wrench(p, pt, tan)
                J[3 * i:3 * i + 3, 3 * c + 2] += sgn * wrench(p, pt, -tan)
    Minv = np.concatenate([[1 / MASS[p], 1 / MASS[p], 1 / INERT[p]] for p in rows]) if rows else np.zeros(0)
    return rows, active, J, g, Minv


def check(present, held):
    """Returns (stable, |v|_inf, per-part v). Stable iff the constraints admit r = 0."""
    rows, active, J, g, Minv = build(present, held)
    if not rows:
        return True, 0.0, {}
    nv = 3 * len(active)
    # friction: lambda_t+ + lambda_t- <= mu lambda_n
    A_ub = np.zeros((len(active), nv))
    for c in range(len(active)):
        A_ub[c, 3 * c] = -MU; A_ub[c, 3 * c + 1] = 1; A_ub[c, 3 * c + 2] = 1
    bounds = [(0, LIMIT)] * nv
    lp = linprog(np.zeros(nv), A_ub=A_ub, b_ub=np.zeros(len(active)), A_eq=J, b_eq=-g,
                 bounds=bounds, method="highs")
    if lp.status == 0:
        return True, 0.0, {p: np.zeros(3) for p in rows}
    # infeasible: minimise the weighted residual as the paper does, to draw the motion
    W = np.diag(Minv)
    obj = lambda lam: 0.5 * (J @ lam + g) @ W @ (J @ lam + g)
    jac = lambda lam: J.T @ (W @ (J @ lam + g))
    cons = [{"type": "ineq", "fun": lambda lam: -A_ub @ lam, "jac": lambda lam: -A_ub}]
    best = None
    for start in (np.zeros(nv), np.full(nv, 1.0)):
        res = minimize(obj, start, jac=jac, bounds=bounds, constraints=cons, method="SLSQP",
                       options={"ftol": 1e-12, "maxiter": 1000})
        if best is None or res.fun < best.fun:
            best = res
    r = J @ best.x + g
    v = Minv * r
    per = {p: v[3 * i:3 * i + 3] for i, p in enumerate(rows)}
    return False, float(np.abs(v).max()), per


# ------------------------------------------------------------------ sequences
full = set(range(N))
assert check(full, set())[0], "the full arch must stand"
assert not check(full - {2}, set())[0], "the arch without its keystone must fall"

top_down = [  # (present, held, label)
    (full, set(), "start:\nfull arch"),
    (full, {2}, "hold P3\n(keystone)"),
    (full - {2}, set(), "remove P3\n"),
]
left_right = [
    (full, {0, 1}, "hold P1,\nhold P2"),
    (full - {0}, {1}, "remove P1\n"),
    (full - {0, 1}, {2}, "hold P3,\nremove P2"),
    (full - {0, 1, 2}, {3}, "hold P4,\nremove P3"),
    (full - {0, 1, 2, 3}, {4}, "hold P5,\nremove P4"),
    (set(), set(), "remove P5:\ndone, reward +1"),
]


def draw_state(ax, present, held, label):
    stable, vmax, per = check(present, held)
    for side in (-1, 1):
        ax.add_patch(Rectangle((side * RO if side < 0 else RI, -0.9), RO - RI, 0.9, fc=STONE, ec=INK2, lw=0.8, hatch="///"))
    for i, pts in enumerate(PARTS):
        if i in present:
            fc = STONE if i in held else WOOD
            ax.add_patch(Polygon(pts, fc=fc, ec=INK2, lw=0.9, zorder=2))
            if i in held:
                c = COMS[i]
                ax.plot([c[0], c[0] * 1.35], [c[1], c[1] * 1.35], color=INK2, lw=1.6, zorder=3)
                ax.plot(c[0] * 1.35, c[1] * 1.35, "s", color=INK2, ms=4.5, zorder=3)
        else:
            ax.add_patch(Polygon(pts, fc="none", ec=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1))
        c = COMS[i]
        ax.text(c[0], c[1], f"P{i + 1}", fontsize=7, color=INK if i in present else MUTED,
                ha="center", va="center", zorder=4)
    if not stable:
        for p, v in per.items():
            c = COMS[p]
            u = v[:2] / max(np.abs(v[:2]).max(), 1e-9) * 0.8
            ax.annotate("", xy=c + u, xytext=c, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6, shrinkA=0, shrinkB=0), zorder=5)
    col = GOOD if stable else BAD
    verdict = "stable" if stable else f"unstable\n‖v‖∞ = {vmax:.1f}"
    ax.text(0, -0.5, verdict, fontsize=7.5, color=col, ha="center", va="center", fontweight="semibold", linespacing=1.0)
    ax.set_title(label, fontsize=8, color=INK)
    ax.set_xlim(-3.0, 3.0); ax.set_ylim(-1.0, 3.1)
    ax.set_aspect("equal", adjustable="box")
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(col); sp.set_linewidth(1.5)
    ax.set_xticks([]); ax.set_yticks([])


fig, axes = plt.subplots(2, 6, figsize=(8, 3.6))
for ax in axes.ravel():
    ax.set_axis_off()
for k, st in enumerate(top_down):
    axes[0, k].set_axis_on(); draw_state(axes[0, k], *st)
axes[0, 3].text(0.0, 0.5, "episode ends,\nreward −1:\ntop-down fails", fontsize=8.5, color=INK, va="center")
for k, st in enumerate(left_right):
    axes[1, k].set_axis_on(); draw_state(axes[1, k], *st)
fig.text(0.01, 0.74, "top-down\n(2 robots)", fontsize=8.5, color=INK2, rotation=90, va="center", ha="left")
fig.text(0.01, 0.27, "left-to-right\n(2 robots)", fontsize=8.5, color=INK2, rotation=90, va="center", ha="left")
fig.suptitle(f"Where stability enters sequencing: every state after an action must pass the equilibrium check (μ = {MU})",
             fontsize=10.5, fontweight="semibold")
fig.tight_layout(rect=(0.02, 0, 1, 0.94), w_pad=0.4, h_pad=0.3)
save(fig, f"{SLUG}-1.png")

# ------------------------------------------------------------------ Figure 3: Table 3
rows = [("Bottle-12", 95.70, 86.53, 0.28), ("Tetris-17", 98.04, 92.64, 0.25), ("Tetris-14", 94.29, 86.55, 0.21),
        ("Tetris-13", 97.33, 92.32, 0.15), ("Cube-13", 93.74, 90.85, 0.29), ("Dog-35", 99.15, 98.98, 1.14),
        ("Dome-37", 97.36, 90.96, 1.6), ("Dome-72", 97.64, 96.40, 16.05),
        ("Vault-62, 2 robots", 92.32, 38.90, 21.3), ("Vault-62, 3 robots", 97.86, 53.44, 9.5),
        ("Vault-62, 4 robots", 96.18, 69.04, 7.62), ("Vault-62, 5 robots", 94.59, 70.77, 9.85),
        ("Vault-62, 6 robots", 91.24, 78.39, 9.84)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 4.2), gridspec_kw=dict(width_ratios=[2.2, 1]), sharey=True)
y = np.arange(len(rows))[::-1]
for yi, (name, tr, te, h) in zip(y, rows):
    a1.plot([te, tr], [yi, yi], color=GRID, lw=3, zorder=1)
a1.scatter([r[1] for r in rows], y, color=BLUE, s=36, zorder=3, label="training states (beam width 64)")
a1.scatter([r[2] for r in rows], y, color=ORANGE, s=36, zorder=3, label="unseen test states (beam width 128)")
a1.set_yticks(y); a1.set_yticklabels([r[0] for r in rows], fontsize=8.5)
a1.set_xlim(30, 101); a1.set_xlabel("disassembly-policy accuracy (% of tasks solved)")
a1.set_title("Vault test accuracy collapses with 2 robots", fontsize=10)
a1.legend(loc="upper left", fontsize=8)
clean(a1)
a2.barh(y, [r[3] for r in rows], color=BLUE, height=0.6, zorder=3)
for yi, r in zip(y, rows):
    a2.text(r[3] + 0.4, yi, f"{r[3]:g}", fontsize=7.5, color=INK2, va="center")
a2.set_xlim(0, 27); a2.set_xlabel("training time (h)")
a2.set_title("Training time", fontsize=10)
clean(a2)
fig.suptitle("Paper's Table 3: policies trained per structure with PPO", fontsize=11, fontweight="semibold")
fig.tight_layout()
save(fig, f"{SLUG}-3.png")

# ------------------------------------------------------------------ Figure 2: Table 2
models = ["Bottle-12", "Dog-35", "Dome-37", "Vault-62", "Dome-72"]
admm = np.array([4.21, 5.52, 4.56, 16.84, 15.13])
gurobi = np.array([6.44, 8.40, 15.79, 90.32, 101.16])
acc = [99.86, 99.99, 99.82, 99.99, 99.96]
nstates = ["2,919", "47,345", "49,405", "86,112", "155,853"]
fig, ax = plt.subplots(figsize=(7, 3.4))
x = np.arange(len(models)); wb = 0.36
ax.bar(x - wb / 2, gurobi, wb, color=BLUE, label="Gurobi (ground truth, one state at a time)", zorder=3)
ax.bar(x + wb / 2, admm, wb, color=ORANGE, label="ADMM-QP on GPU (batches of 256)", zorder=3)
for i in range(len(models)):
    ax.text(x[i], max(gurobi[i], admm[i]) + 2, f"{gurobi[i] / admm[i]:.1f}× faster\n{acc[i]:.2f} % agree", fontsize=7.5, color=INK2, ha="center")
ax.set_xticks(x); ax.set_xticklabels([f"{m}\n({n} states)" for m, n in zip(models, nstates)], fontsize=8)
ax.set_ylabel("time per state (ms)"); ax.set_ylim(0, 118)
ax.set_title("Paper's Table 2: batching makes the check 1.5–6.7× faster per state at ≥ 99.8 % agreement", fontsize=10)
ax.legend(loc="upper left", fontsize=8)
clean(ax)
fig.tight_layout()
save(fig, f"{SLUG}-2.png")
