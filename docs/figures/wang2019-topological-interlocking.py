"""Figures for docs/wang2019-topological-interlocking.md (Wang, Song, Isvoranu, Pauly 2019).

All stability results below are computed with the paper's frictionless model in 2D:
one compressive point force per vertex of each contact segment, force + torque
balance per free part (A_eq F = -W, F >= 0, checked with scipy.optimize.linprog), the
kinematic side B_in = A_eq^T, and the tilt-angle measure Phi found by sweeping /
binary-searching the gravity direction.
Run from docs/figures/:  python wang2019-topological-interlocking.py
"""
from _style import *  # noqa: F401,F403

import numpy as np
from matplotlib.patches import Polygon, Wedge
from scipy.optimize import linprog

# --------------------------------------------------------------------------- 2D rigid-body statics
def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def poly_centroid_area(P):
    P = np.asarray(P, float); x, y = P[:, 0], P[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    A = 0.5 * cr.sum()
    cx = ((x + xn) * cr).sum() / (6 * A); cy = ((y + yn) * cr).sum() / (6 * A)
    return np.array([cx, cy]), A


def equilibrium_matrix(contacts, centroid):
    """Columns: one compressive force per contact point (r, n) acting on the free part along n.
    Rows: Fx, Fy, torque about the centroid.  (A_eq of Eq. 8; B_in = A_eq^T is Eq. 5.)"""
    cols = []
    for r, n in contacts:
        cols.append([n[0], n[1], cross2(np.asarray(r) - centroid, n)])
    return np.array(cols).T


def holds(A, W):
    """Is there F >= 0 with A F = -W ?  (equilibrium under external wrench W)"""
    r = linprog(np.zeros(A.shape[1]), A_eq=A, b_eq=-np.asarray(W), bounds=[(0, None)] * A.shape[1], method="highs")
    return r.status == 0


def escape(A, W):
    """Farkas alternative: motion Y with B_in Y >= 0 and W^T Y > 0.  Returns (power, Y)."""
    B = A.T
    r = linprog(-np.asarray(W), A_ub=-B, b_ub=np.zeros(B.shape[0]), bounds=[(-1, 1)] * 3, method="highs")
    return -r.fun, r.x


def critical_tilt(A, mass, lo=0.0, hi=180.0, iters=40):
    """Binary search the polar angle phi (deg from straight down) at which self-weight equilibrium fails."""
    W0 = np.array([0.0, -mass, 0.0])
    if not holds(A, W0):
        return np.nan
    if holds(A, np.array([mass * np.sin(np.radians(hi)), -mass * np.cos(np.radians(hi)), 0.0])):
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        W = np.array([mass * np.sin(np.radians(mid)), -mass * np.cos(np.radians(mid)), 0.0])
        if holds(A, W):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- block configurations
def wedge_in_groove(gamma_deg, depth=1.0, top=0.0):
    """Convex block whose lower part is a wedge seated in a V-groove; gamma = wall angle from vertical."""
    g = np.radians(gamma_deg); w = depth * np.tan(g)
    block = [(0, 0), (w, depth), (w, depth + top), (-w, depth + top), (-w, depth)]
    if top == 0:
        block = [(0, 0), (w, depth), (-w, depth)]
    nL = np.array([np.cos(g), np.sin(g)]); nR = np.array([-np.cos(g), np.sin(g)])
    contacts = [((0, 0), nL), ((-w, depth), nL), ((0, 0), nR), ((w, depth), nR)]
    base = [(-2.0, -0.6), (2.0, -0.6), (2.0, depth), (w, depth), (0, 0), (-w, depth), (-2.0, depth)]
    return block, contacts, base


def block_on_table():
    block = [(-0.5, 0), (0.5, 0), (0.5, 1), (-0.5, 1)]
    n = np.array([0.0, 1.0])
    contacts = [((-0.5, 0), n), ((0.5, 0), n)]
    base = [(-2.0, -0.6), (2.0, -0.6), (2.0, 0), (-2.0, 0)]
    return block, contacts, base


def block_in_pocket():
    block = [(-0.5, 0), (0.5, 0), (0.5, 1), (-0.5, 1)]
    contacts = [((-0.5, 0), np.array([0.0, 1.0])), ((0.5, 0), np.array([0.0, 1.0])),
                ((-0.5, 0), np.array([1.0, 0.0])), ((-0.5, 1), np.array([1.0, 0.0])),
                ((0.5, 0), np.array([-1.0, 0.0])), ((0.5, 1), np.array([-1.0, 0.0])),
                ((-0.5, 1), np.array([0.0, -1.0])), ((0.5, 1), np.array([0.0, -1.0]))]
    base = [(-2.0, -0.6), (2.0, -0.6), (2.0, 1.6), (-2.0, 1.6), (-2.0, 1.0), (-0.5, 1.0), (-0.5, 0.0),
            (0.5, 0.0), (0.5, 1.0), (-2.0, 1.0)]
    return block, contacts, base


def draw_config(ax, block, base, contacts, feasible_deg, title):
    ax.add_patch(Polygon(base, closed=True, fc=STONE, ec=INK2, lw=1.0, zorder=1))
    ax.add_patch(Polygon(block, closed=True, fc=WOOD, ec=INK2, lw=1.0, zorder=2))
    c, _ = poly_centroid_area(block)
    # feasible / infeasible gravity directions drawn as arcs of a circle around the centroid
    R = 1.35
    deg = np.arange(-180, 180)
    for lo in deg:
        ok = feasible_deg[(lo + 180) % 360]
        th0 = -90 + lo          # gravity d(phi) = (sin phi, -cos phi): plot angle -90 deg + phi
        ax.add_patch(Wedge(c, R, th0, th0 + 1, width=0.16, fc=GOOD if ok else BAD, ec="none", zorder=3, alpha=0.9))
    ax.plot([c[0]], [c[1]], "o", ms=3, color=INK, zorder=4)
    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-0.8, 2.2)
    ax.set_title(title, fontsize=9, fontweight="normal")
    clean(ax, equal=True, hide_axes=True)


# --------------------------------------------------------------------------- Figure 1: feasible gravity cones
configs = [
    ("(a) flat contact\nΦ = 0°", block_on_table()),
    ("(b) V-groove, walls 45°\nfrom vertical: Φ = 45°", wedge_in_groove(45.0)),
    ("(c) V-groove, walls 20°\nfrom vertical: Φ = 70°", wedge_in_groove(20.0)),
    ("(d) closed pocket\nΦ = 180°", block_in_pocket()),
]
fig, axs = plt.subplots(1, 4, figsize=(8, 2.7))
phis = []
for ax, (title, (block, contacts, base)) in zip(axs, configs):
    c, area = poly_centroid_area(block)
    A = equilibrium_matrix(contacts, c)
    feas = []
    for phi in range(-180, 180):
        W = np.array([area * np.sin(np.radians(phi)), -area * np.cos(np.radians(phi)), 0.0])
        feas.append(holds(A, W))
    Phi = critical_tilt(A, area)
    phis.append(Phi)
    print(f"{title.splitlines()[0]}: Phi = {Phi:.2f} deg, feasible directions = {sum(feas)} of 360")
    draw_config(ax, block, base, contacts, feas, title)
from matplotlib.patches import Patch
axs[0].legend(handles=[Patch(color=GOOD, label="gravity direction held"), Patch(color=BAD, label="not held (slides / tips)")],
              loc="upper center", bbox_to_anchor=(2.3, 0.02), ncol=2, fontsize=8.5)
fig.suptitle("More blocking faces widen the cone of gravity directions a frictionless block can hold (cf. paper Fig. 2)",
             fontsize=10.5, fontweight="semibold", color=INK, y=1.08)
save(fig, "wang2019-topological-interlocking-1.png")

# --------------------------------------------------------------------------- Figure 2: Phi vs groove angle, with torque
gammas = np.linspace(2, 88, 44)
curves = {}
for label, top in [("wedge only (squat)", 0.0), ("wedge + tall body (3× groove depth)", 3.0)]:
    vals = []
    for g in gammas:
        block, contacts, base = wedge_in_groove(g, depth=1.0, top=top)
        c, area = poly_centroid_area(block)
        vals.append(critical_tilt(equilibrium_matrix(contacts, c), area))
    curves[label] = np.array(vals)
    print(label, "Phi at gamma=45:", np.interp(45, gammas, vals))

fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.plot(gammas, curves["wedge only (squat)"], color=BLUE, lw=2.6, label="full 2D equilibrium, squat wedge", zorder=2)
ax.plot(gammas, 90 - gammas, "--", color=INK2, lw=1.2, label="forces only, no torque: Φ = 90° − γ (coincides with squat wedge)", zorder=3)
ax.plot(gammas, curves["wedge + tall body (3× groove depth)"], color=ORANGE, lw=2,
        label="full 2D equilibrium, tall block (tips first)")
ax.set_xlabel("groove wall angle γ from vertical (°)")
ax.set_ylabel("critical tilt Φ (°)")
ax.set_xlim(0, 90); ax.set_ylim(0, 95)
ax.set_xticks(range(0, 91, 15)); ax.set_yticks(range(0, 91, 15))
ax.legend(loc="lower left", fontsize=8)
ax.set_title("Steeper groove walls hold a wider tilt; a tall block loses stability to tipping",
             fontsize=10.5)
clean(ax)
save(fig, "wang2019-topological-interlocking-2.png")

# --------------------------------------------------------------------------- Figure 3: Farkas duality, translations only
gam = 45.0
block, contacts, base = wedge_in_groove(gam)
c, area = poly_centroid_area(block)
A2 = np.array([[n[0], n[1]] for r, n in contacts]).T                  # translational part only (2 x 4)
normals = np.array([[np.cos(np.radians(gam)), np.sin(np.radians(gam))], [-np.cos(np.radians(gam)), np.sin(np.radians(gam))]])


def holds2(W):
    r = linprog(np.zeros(A2.shape[1]), A_eq=A2, b_eq=-np.asarray(W), bounds=[(0, None)] * A2.shape[1], method="highs")
    return r.status == 0, (r.x if r.status == 0 else None)


def escape2(W):
    B = A2.T
    r = linprog(-np.asarray(W), A_ub=-B, b_ub=np.zeros(B.shape[0]), bounds=[(-1, 1)] * 2, method="highs")
    return -r.fun, r.x


angles = np.arange(0, 360, 1)
held = np.array([holds2([np.cos(np.radians(a)), np.sin(np.radians(a))])[0] for a in angles])
motion = np.array([all(np.dot(normals, [np.cos(np.radians(a)), np.sin(np.radians(a))]) >= -1e-12) for a in angles])
print("held load directions:", held.sum(), "deg; admissible motion directions:", motion.sum(), "deg")

fig, axs = plt.subplots(1, 2, figsize=(8, 3.6), gridspec_kw={"wspace": 0.35})
# (a) the two cones on one circle
ax = axs[0]
th = np.linspace(0, 2 * np.pi, 361)
ax.plot(np.cos(th), np.sin(th), color=GRID, lw=1.2, zorder=1)
for a, ok in zip(angles, held):
    if ok:
        ax.add_patch(Wedge((0, 0), 1.0, a - 0.5, a + 0.5, fc=GOOD, ec="none", zorder=2, alpha=0.35))
for a, ok in zip(angles, motion):
    if ok:
        ax.add_patch(Wedge((0, 0), 1.0, a - 0.5, a + 0.5, fc=BLUE, ec="none", zorder=2, alpha=0.35))
for n in normals:
    ax.annotate("", xy=0.72 * n, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2, mutation_scale=10), zorder=3)
ax.text(0.70, 0.52, "n₁", color=INK2, fontsize=9, ha="left"); ax.text(-0.70, 0.52, "n₂", color=INK2, fontsize=9, ha="right")
ax.text(0.0, -0.62, "loads W that the\ncontact forces\ncan balance", ha="center", va="center", fontsize=8.5, color=INK)
ax.text(0.0, 0.62, "admissible\nmotions v", ha="center", va="center", fontsize=8.5, color=INK)
ax.text(1.06, 0.0, "neither: load not\nheld, motion\nnot admissible", ha="left", va="center", fontsize=7.5, color=INK2)
ax.set_xlim(-1.2, 1.9); ax.set_ylim(-1.2, 1.2)
ax.set_title("(a) Balanced loads (green) and admissible motions\n(blue) are polar cones: W·v ≤ 0 for every pair",
             fontsize=9, fontweight="normal")
clean(ax, equal=True, hide_axes=True)
# (b) two sample loads: one balanced, one escaping
ax = axs[1]
ax.add_patch(Polygon(base, closed=True, fc=STONE, ec=INK2, lw=1.0, zorder=1))
ax.add_patch(Polygon(block, closed=True, fc=WOOD, ec=INK2, lw=1.0, zorder=2))
samples = [("W₁ (tilt 30°)", 30.0, GOOD), ("W₂ (tilt 60°)", 60.0, BAD)]
for lab, tilt, col in samples:
    W = area * np.array([np.sin(np.radians(tilt)), -np.cos(np.radians(tilt))])
    ok, F = holds2(W)
    pw, v = escape2(W)
    print(f"{lab}: held={ok}, F={None if F is None else np.round(F, 3)}, escape power={pw:.3f}, v={np.round(v, 3)}")
    Wn = W / np.linalg.norm(W) * 0.9
    ax.annotate("", xy=c + Wn, xytext=c, arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, mutation_scale=12), zorder=4)
    ax.text(*(c + Wn + np.array([0.1, -0.04])), lab, color=INK, fontsize=8.5, ha="left", va="top", zorder=5)
    if ok:
        # contact forces (summed per face) that balance W1
        fL = (F[0] + F[1]) * normals[0]; fR = (F[2] + F[3]) * normals[1]
        for f, foot, lab2 in [(fL, np.array([-0.5, 0.5]), "f₁n₁"), (fR, np.array([0.5, 0.5]), "f₂n₂")]:
            fn = f / area * 0.9
            ax.annotate("", xy=foot + fn, xytext=foot, arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6, mutation_scale=11), zorder=4)
            ax.text(*(foot + fn + np.array([-0.1, 0.05])), lab2, color=INK2, fontsize=8, ha="right", va="bottom")
    else:
        vn = v / np.linalg.norm(v) * 0.9
        ax.annotate("", xy=c + vn, xytext=c, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.0, mutation_scale=12), zorder=4)
        ax.text(*(c + vn + np.array([0.08, 0.06])), "escaping motion v\n(W₂·v > 0)", color=INK, fontsize=8.5, ha="left", va="bottom")
ax.plot([c[0]], [c[1]], "o", ms=3, color=INK, zorder=5)
ax.set_xlim(-1.6, 2.4); ax.set_ylim(-0.7, 2.0)
ax.set_title("(b) Walls 45° from vertical: W₁ is balanced by\ncontact forces f₁n₁ + f₂n₂; for W₂ the LP finds a motion",
             fontsize=9, fontweight="normal")
clean(ax, equal=True, hide_axes=True)
fig.suptitle("Farkas' lemma: a load is either balanced by compressive contact forces or does positive work on an admissible motion",
             fontsize=10, fontweight="semibold", color=INK, y=1.06)
save(fig, "wang2019-topological-interlocking-3.png")

# --------------------------------------------------------------------------- Figure 4: Table 1 (before -> after optimization)
# (surface, parts, Phi before [deg or None = no equilibrium], Phi after, globally interlocking after)
table1 = [
    ("Roof", 62, 180.0, 180.0, True), ("Igloo", 64, 22.2, 29.4, False), ("Bump", 110, 0.8, 13.4, False),
    ("Lilium", 105, None, 16.2, False), ("Blob", 113, None, 31.7, False), ("Spindle", 133, 1.4, 31.8, False),
    ("Flower", 346, 180.0, 180.0, True), ("Torus", 120, 10.6, 25.7, False), ("Hyperbolic", 110, 33.8, 180.0, True),
    ("Peanut", 122, 5.2, 35.3, False), ("Pentagon", 190, 31.9, 78.5, False), ("Six", 138, 33.1, 65.0, False),
    ("Buga Pavilion", 136, 26.2, 77.1, False), ("Vase", 95, 14.6, 36.5, False), ("Surface Vouga", 104, None, 53.8, False),
    ("Free Holes", 345, 18.3, 46.6, False),
]
rows = sorted(table1, key=lambda r: r[3])
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ys = np.arange(len(rows))
for y, (name, n, b, a, gi) in zip(ys, rows):
    b0 = 0.0 if b is None else b
    ax.plot([b0, a], [y, y], color=GRID, lw=2.4, zorder=1)
    if b is None:
        ax.plot([0], [y], "o", ms=6, mfc=SURFACE, mec=BLUE, mew=1.5, zorder=3)
    else:
        ax.plot([b], [y], "o", ms=6, color=BLUE, zorder=3)
    ax.plot([a], [y], "o", ms=6, color=ORANGE, zorder=3)
    ax.text(a + 4, y, ("globally interlocking" if gi else f"{a:.1f}°") + f"  (N = {n})", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
ax.set_xlim(-3, 245); ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
ax.set_xlabel("stability measure Φ, critical tilt angle (°)")
ax.plot([], [], "o", color=BLUE, label="before optimization"); ax.plot([], [], "o", mfc=SURFACE, mec=BLUE, mew=1.5, label="before: no equilibrium under gravity (shown at 0°)")
ax.plot([], [], "o", color=ORANGE, label="after optimization")
ax.legend(loc="lower right", fontsize=8)
ax.set_title("Optimizing face tilts raises Φ on every surface that is not already interlocking (Table 1)", fontsize=10.5)
clean(ax)
save(fig, "wang2019-topological-interlocking-4.png")
