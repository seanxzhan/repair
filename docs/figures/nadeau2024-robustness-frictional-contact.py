"""Figures for nadeau2024-robustness-frictional-contact.md.

A 2D toy (slab S on legs L1, L2 on a fixed floor) run through the paper's
pipeline: minimum-energy contact forces (eqs. 8-11, solved with SLSQP), the
closed-form per-contact slip limit (eqs. 14-20), interface sums (21), max-flow
over the contact interface graph (22) and cut/pivot enumeration for toppling
(23-27). Everything drawn is computed. Figure 3 charts the paper's Table I.
Run from docs/figures/:  python nadeau2024-robustness-frictional-contact.py
"""
from _style import *  # noqa: F401,F403

import itertools
import numpy as np
import networkx as nx
from matplotlib.patches import Polygon, Rectangle
from scipy.optimize import minimize, linprog

SLUG = "nadeau2024-robustness-frictional-contact"
G = 9.81
MU = 0.5

# ------------------------------------------------------------------ scene (m, kg)
OBJ = {  # name: (xmin, xmax, ymin, ymax, mass)
    "S": (0.0, 6.0, 2.0, 2.6, 3.0),
    "L1": (0.5, 1.3, 0.0, 2.0, 1.0),
    "L2": (4.7, 5.5, 0.0, 2.0, 1.0),
}
FIXED = "F"


def com(name):
    x0, x1, y0, y1, m = OBJ[name]
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2])


# Interfaces: (upper object, lower object, y, xmin, xmax). Contact forces act on the
# upper object with inward normal (0, +1); the lower object gets the reaction.
NPTS = 3  # contact points per interface (the paper samples 20 by farthest-point sampling)
INTERFACES = [
    ("S", "L1", 2.0, 0.5, 1.3),
    ("S", "L2", 2.0, 4.7, 5.5),
    ("L1", FIXED, 0.0, 0.5, 1.3),
    ("L2", FIXED, 0.0, 4.7, 5.5),
]
CONTACTS = []  # (interface index, upper, lower, point, normal on upper)
for t, (u, l, y, x0, x1) in enumerate(INTERFACES):
    for x in np.linspace(x0, x1, NPTS):
        CONTACTS.append((t, u, l, np.array([x, y]), np.array([0.0, 1.0])))
NC = len(CONTACTS)
FREE = list(OBJ)


def equilibrium_matrices():
    """A f + w = 0 for every free object; f = [ft_0, fn_0, ft_1, fn_1, ...]."""
    A = np.zeros((3 * len(FREE), 2 * NC))
    w = np.zeros(3 * len(FREE))
    for j, name in enumerate(FREE):
        w[3 * j + 1] = -OBJ[name][4] * G
        c = com(name)
        for k, (t, u, l, p, n) in enumerate(CONTACTS):
            tvec = np.array([-n[1], n[0]])
            for obj, sgn in ((u, 1.0), (l, -1.0)):
                if obj != name:
                    continue
                r = p - c
                A[3 * j:3 * j + 2, 2 * k] += sgn * tvec
                A[3 * j:3 * j + 2, 2 * k + 1] += sgn * n
                A[3 * j + 2, 2 * k] += sgn * (r[0] * tvec[1] - r[1] * tvec[0])
                A[3 * j + 2, 2 * k + 1] += sgn * (r[0] * n[1] - r[1] * n[0])
    return A, w


def solve_forces():
    """Eqs. (8)-(11): min sum |f_i|^2 s.t. equilibrium, |ft| <= mu fn, fn >= 0."""
    A, w = equilibrium_matrices()
    # feasible start from an LP
    G_ub, h = [], []
    for k in range(NC):
        r1 = np.zeros(2 * NC); r1[2 * k] = 1; r1[2 * k + 1] = -MU; G_ub.append(r1); h.append(0)
        r2 = np.zeros(2 * NC); r2[2 * k] = -1; r2[2 * k + 1] = -MU; G_ub.append(r2); h.append(0)
    G_ub, h = np.array(G_ub), np.array(h)
    bounds = [(None, None), (0, None)] * NC
    lp = linprog(np.ones(2 * NC) * 0, A_ub=G_ub, b_ub=h, A_eq=A, b_eq=-w, bounds=bounds, method="highs")
    assert lp.status == 0, "assembly is unstable under gravity"
    cons = [{"type": "eq", "fun": lambda f: A @ f + w, "jac": lambda f: A},
            {"type": "ineq", "fun": lambda f: h - G_ub @ f, "jac": lambda f: -G_ub}]
    res = minimize(lambda f: f @ f, lp.x, jac=lambda f: 2 * f, constraints=cons, bounds=bounds,
                   method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
    f = res.x
    assert np.abs(A @ f + w).max() < 1e-6, "equilibrium residual too large"
    assert (f[1::2] >= -1e-9).all() and (np.abs(f[0::2]) <= MU * f[1::2] + 1e-9).all()
    return f.reshape(NC, 2)  # (ft, fn) per contact


FORCES = solve_forces()
# sanity: total normal force on the floor equals the total weight
floor = [k for k, c in enumerate(CONTACTS) if c[2] == FIXED]
assert abs(FORCES[floor, 1].sum() - sum(o[4] for o in OBJ.values()) * G) < 1e-6


def slip_contact(k, e_world):
    """Eqs. (14)-(20) with the v components zero (2D). Returns R_k in N or inf.

    e_world is the push on the object. Equilibrium f_new = f - s e means the contact
    force tip moves along -e, so the paper's closed form (written for the increment
    of the contact force) is evaluated at -e. This is the sense used in the paper's
    own examples (pushing a cube into its support improves the contact condition).
    """
    t, u, l, p, n = CONTACTS[k]
    tvec = np.array([-n[1], n[0]])
    ft, fn = FORCES[k]
    et, en = -(e_world @ tvec), -(e_world @ n)
    d = MU**2 * en**2 - et**2
    b = ft * et - MU**2 * fn * en                   # = -(mu^2 fn en - ft.et)
    disc = MU**2 * (en * ft - et * fn) ** 2         # eq. (17) squared, the cross term is 0 in 2D
    if disc < 0:
        return np.inf
    nn = np.sqrt(disc)
    if abs(d) < 1e-12:                               # eq. (19)
        den = 2 * (MU**2 * en * fn - et * ft)
        if abs(den) < 1e-12:
            return np.inf
        sm = (ft**2 - MU**2 * fn**2) / den
    else:                                            # eq. (16)
        sm = (b - nn) / d
    return sm if sm >= 0 else np.inf                 # eq. (20)


def slip_interfaces(e_world):
    """Eq. (21): sum of contact limits per interface."""
    caps = np.zeros(len(INTERFACES))
    for k, c in enumerate(CONTACTS):
        caps[c[0]] += slip_contact(k, e_world)
    return caps


def slip_assembly(source, e_world):
    """Eq. (22): max-flow from the pushed object to the fixed node. Returns (R, min-cut edges)."""
    caps = slip_interfaces(e_world)
    g = nx.DiGraph()
    for t, (u, l, *_) in enumerate(INTERFACES):
        if np.isfinite(caps[t]):
            g.add_edge(u, l, capacity=float(caps[t]))
            g.add_edge(l, u, capacity=float(caps[t]))
        else:
            g.add_edge(u, l); g.add_edge(l, u)  # no capacity key = unbounded
    try:
        value, (side_s, side_t) = nx.minimum_cut(g, source, FIXED)
    except nx.NetworkXUnbounded:
        return np.inf, []
    cut = [t for t, (u, l, *_) in enumerate(INTERFACES) if (u in side_s) != (l in side_s)]
    return value, cut


def hull_vertices(points):
    from shapely.geometry import MultiPoint
    hull = MultiPoint([tuple(p) for p in points]).convex_hull
    if hull.geom_type == "Point":
        return [np.array(hull.coords[0])]
    if hull.geom_type == "LineString":
        return [np.array(c) for c in hull.coords]
    return [np.array(c) for c in hull.exterior.coords[:-1]]


def topple(source, p, e_world):
    """Eqs. (23)-(27) in 2D: enumerate cuts (groups containing the pushed object),
    candidate pivots (hull vertices of the cut interfaces' contact points) and both
    rotation senses. Returns (R_top, (group, pivot, sense))."""
    best, arg = np.inf, None
    others = [o for o in FREE if o != source]
    for r in range(len(others) + 1):
        for extra in itertools.combinations(others, r):
            group = {source, *extra}
            cut_ifaces = [t for t, (u, l, *_) in enumerate(INTERFACES) if (u in group) != (l in group)]
            pts = [c[3] for c in CONTACTS if c[0] in cut_ifaces]
            for a in hull_vertices(pts):
                for sense in (+1.0, -1.0):
                    tau = 0.0
                    for k, (t, u, l, q, n) in enumerate(CONTACTS):
                        if t not in cut_ifaces:
                            continue
                        # normal pointing into the group's object at this contact
                        n_g, f_g = (n, FORCES[k]) if u in group else (-n, -FORCES[k])
                        rr = q - a
                        vel = sense * np.array([-rr[1], rr[0]])   # velocity of the group's point
                        vn = vel @ n_g
                        if vn < -1e-9:
                            tau = np.inf; break                    # kinematically blocked
                        if vn > 1e-9:
                            tvec = np.array([-n[1], n[0]])
                            fw = f_g[0] * tvec + f_g[1] * n         # world force on the group's object
                            tau += fw @ vel                         # moment of the contact force
                        # vn == 0: point on the pivot or sliding; contributes nothing here
                    if not np.isfinite(tau):
                        continue
                    rp = p - a
                    drive = sense * (rp[0] * e_world[1] - rp[1] * e_world[0])
                    if drive <= 1e-9:
                        continue                                    # push does not drive this rotation
                    R = tau / drive
                    if R < best:
                        best, arg = R, (tuple(sorted(group)), a, sense)
    return best, arg


def robustness(source, p, e_world):
    rs, cut = slip_assembly(source, e_world)
    rt, arg = topple(source, p, e_world)
    return min(rs, rt), rs, rt, cut, arg


# ------------------------------------------------------------------ hand checks
P = np.array([6.0, 2.6])  # push point: top-right corner of the slab
WS = OBJ["S"][4] * G
WT = sum(o[4] for o in OBJ.values()) * G
r_right = robustness("S", P, np.array([1.0, 0.0]))
assert abs(r_right[1] - MU * WS) < 1e-6, r_right       # slab slides on both legs: mu * W_S
r_down = robustness("S", P, np.array([0.0, -1.0]))
assert np.isinf(r_down[1]) and abs(r_down[2] - WS * 2.5 / 0.5) < 1e-6, r_down
r_up = robustness("S", P, np.array([0.0, 1.0]))
assert abs(r_up[1] - WS) < 1e-6 and abs(r_up[2] - WS * 2.5 / 5.5) < 1e-6, r_up
print("hand checks passed:", {k: np.round(v[:3], 2) for k, v in
                              {"right": r_right, "down": r_down, "up": r_up}.items()})

# ------------------------------------------------------------------ Figure 1: scene + CIG
fig, (ax, axg) = plt.subplots(1, 2, figsize=(8, 3.6), gridspec_kw=dict(width_ratios=[1.45, 1]))
ax.add_patch(Rectangle((-0.6, -0.5), 7.2, 0.5, fc=STONE, ec=INK2, lw=0.8, hatch="///", zorder=1))
for name, (x0, x1, y0, y1, m) in OBJ.items():
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=WOOD, ec=INK2, lw=1.0, zorder=2))
    c = com(name)
    ax.plot(*c, "o", color=INK, ms=3.5, zorder=4)
    ax.text(c[0] + 0.12, c[1] + 0.02, f"{name}  {m:g} kg", fontsize=8.5, color=INK, va="center", zorder=5)
ax.text(3.0, -0.27, "floor (fixed)", fontsize=8, color=INK2, ha="center", va="center",
        bbox=dict(fc=STONE, ec="none", pad=1.5))
scale = 0.06  # m per N
for k, (t, u, l, p, n) in enumerate(CONTACTS):
    ft, fn = FORCES[k]
    F = ft * np.array([-n[1], n[0]]) + fn * n
    ax.plot(*p, "o", color=INK2, ms=3, zorder=5)
    ax.annotate("", xy=p + scale * F, xytext=p,
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6, shrinkA=0, shrinkB=0), zorder=6)
ax.text(3.0, 1.55, f"QP contact forces (blue);\nfloor total {FORCES[floor, 1].sum():.0f} N = weight",
        fontsize=8, color=INK2, ha="center", va="center")
ax.annotate("", xy=P + np.array([0, 0]), xytext=P + np.array([-1.2, 0.0]),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2, shrinkA=0, shrinkB=0), zorder=6)
ax.plot(*P, "s", color=ORANGE, ms=5, zorder=7)
ax.text(P[0] + 0.1, P[1] + 0.05, "push p, ê = +x", fontsize=8.5, color=INK, ha="right", va="bottom")
ax.set_xlim(-0.7, 6.7); ax.set_ylim(-0.6, 3.5)
ax.set_aspect("equal", adjustable="box")
ax.set_title(f"Toy table, μ = {MU}: forces under gravity", fontsize=10)
clean(ax, hide_axes=True)

# CIG with capacities for the +x push (eq. 21) and the min cut
caps = slip_interfaces(np.array([1.0, 0.0]))
Rs, cut = slip_assembly("S", np.array([1.0, 0.0]))
pos = {"S": (0.5, 1.0), "L1": (0.0, 0.5), "L2": (1.0, 0.5), FIXED: (0.5, 0.0)}
for t, (u, l, *_) in enumerate(INTERFACES):
    (xa, ya), (xb, yb) = pos[u], pos[l]
    col = ORANGE if t in cut else INK2
    axg.plot([xa, xb], [ya, yb], color=col, lw=2.0 if t in cut else 1.5, zorder=1)
    lab = f"{caps[t]:.1f} N" if np.isfinite(caps[t]) else "∞"
    mx, my = (xa + xb) / 2, (ya + yb) / 2
    dx = -0.13 if xa + xb < 1 else 0.13
    axg.text(mx + dx, my, lab, fontsize=9, color=INK, ha="center", va="center",
             bbox=dict(fc=SURFACE, ec="none", pad=1))
for name, (x, y) in pos.items():
    lab = "floor\n(sink)" if name == FIXED else (f"{name}\n(source)" if name == "S" else name)
    axg.text(x, y, lab, fontsize=9, color=INK, ha="center", va="center",
             bbox=dict(boxstyle="circle,pad=0.35", fc=STONE if name == FIXED else WOOD, ec=INK2, lw=0.8), zorder=3)
axg.plot([0.05, 0.95], [0.72, 0.72], color=ORANGE, lw=1.2, ls=(0, (4, 3)))
axg.text(0.5, -0.22, f"min cut (dashed): R_slip = {Rs:.1f} N = μ·W_S", fontsize=8.5, color=INK, ha="center")
axg.set_xlim(-0.35, 1.35); axg.set_ylim(-0.3, 1.22)
axg.set_title("Contact interface graph, +x push", fontsize=10)
clean(axg, hide_axes=True)
fig.suptitle("Slip capacity flows like water: the two slab–leg edges are the bottleneck", fontsize=11,
             fontweight="semibold")
fig.tight_layout()
save(fig, f"{SLUG}-1.png")

# ------------------------------------------------------------------ Figure 2: rose plot of R(p, e)
angles = np.radians(np.arange(0, 360, 1.0))
Rmin, Rs_all, Rt_all, mech = [], [], [], []
for a in angles:
    e = np.array([np.cos(a), np.sin(a)])
    R, rs, rt, _, _ = robustness("S", P, e)
    Rmin.append(R); Rs_all.append(rs); Rt_all.append(rt); mech.append("slip" if rs <= rt else "topple")
Rmin, Rs_all, Rt_all = map(np.array, (Rmin, Rs_all, Rt_all))
RMAX = 160.0

fig = plt.figure(figsize=(8, 4.4))
axp = fig.add_subplot(1, 2, 1, projection="polar")
axc = fig.add_subplot(1, 2, 2)


def clipped(v):
    v = np.array(v, float)
    return np.where(np.isfinite(v), np.minimum(v, RMAX), RMAX)


axp.plot(angles, clipped(Rs_all), color=BLUE, lw=2, label="R_slip (max-flow)")
axp.plot(angles, clipped(Rt_all), color=ORANGE, lw=2, label="R_top (cuts × pivots)")
axp.fill(angles, clipped(Rmin), color=STONE, alpha=0.6, lw=0, label="R = min (survives)")
axp.set_rmax(RMAX); axp.set_rticks([50, 100, 150]); axp.set_rlabel_position(112)
axp.tick_params(labelsize=8, colors=INK2)
axp.set_thetagrids([0, 90, 180, 270], ["+x", "+y (up)", "−x", "−y (down)"])
axp.grid(True, color=GRID)
axp.set_title(f"Push at p = ({P[0]:g}, {P[1]:g}) m; radius in N (∞ clipped at {RMAX:.0f})", fontsize=9.5, pad=12)
axp.legend(loc="upper left", bbox_to_anchor=(-0.25, 1.02), fontsize=8)
for a_deg, txt in [(0, f"{Rmin[0]:.1f} N\nslab slides"), (90, f"{Rmin[90]:.1f} N\nslab lifts off L1"),
                   (270, f"{Rmin[270]:.0f} N\nslab tips over L2")]:
    r = min(Rmin[a_deg], RMAX)
    axp.annotate(txt, (np.radians(a_deg), r), xytext=(np.radians(a_deg), r + 70 if r < 80 else r - 60),
                 fontsize=7.5, color=INK, ha="center", va="center",
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))

# Cartesian view: which mechanism governs, log scale
deg = np.degrees(angles)
axc.plot(deg, np.where(np.isfinite(Rs_all), Rs_all, np.nan), color=BLUE, lw=2, label="R_slip")
axc.plot(deg, np.where(np.isfinite(Rt_all), Rt_all, np.nan), color=ORANGE, lw=2, label="R_top")
gov_slip = np.array([m == "slip" for m in mech]) & np.isfinite(Rmin)
gov_top = np.array([m == "topple" for m in mech]) & np.isfinite(Rmin)
axc.fill_between(deg, 5, 1e4, where=gov_slip, color=BLUE, alpha=0.08, lw=0, label="slip governs")
axc.fill_between(deg, 5, 1e4, where=gov_top, color=ORANGE, alpha=0.10, lw=0, label="toppling governs")
axc.set_yscale("log"); axc.set_ylim(8, 3000); axc.set_xlim(0, 360)
axc.set_xticks([0, 90, 180, 270, 360]); axc.set_xticklabels(["+x", "+y", "−x", "−y", "+x"])
axc.set_xlabel("push direction ê (degrees from +x)"); axc.set_ylabel("largest push before motion (N)")
axc.set_title("Slip governs sideways, toppling up and down", fontsize=9.5)
axc.text(356, 300, "gaps = ∞ (push goes\ninto the friction cone)", fontsize=7.5, color=INK2, ha="right")
axc.legend(loc="upper left", fontsize=7.5, ncol=1)
clean(axc)
fig.tight_layout()
save(fig, f"{SLUG}-2.png")

# ------------------------------------------------------------------ Figure 3: Table I
methods = ["Heuristic\n(prior work)", "Simulation\n(PyBullet)", "Optim.\n4-gon", "Optim.\n8-gon", "This\nwork"]
time_s = np.array([[0.07, 0.23, 0.24], [0.80, 10.1, 41.7], [10.9, 151, 118], [23.6, 205, 118], [0.13, 0.20, 0.23]])
err = np.array([[0, 30, 44], [22, 26, 67], [16, 17, 23.6], [11, 10, 9], [0.00, 1.00, 1.20]])
scenes = ["cube", "stack of 3 cubes", "slab on 2 legs"]
cols = [BLUE, ORANGE, AQUA]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.6))
x = np.arange(len(methods)); wbar = 0.26
for j in range(3):
    a1.bar(x + (j - 1) * wbar, time_s[:, j], wbar, color=cols[j], label=scenes[j], zorder=3)
    a2.bar(x + (j - 1) * wbar, err[:, j], wbar, color=cols[j], label=scenes[j], zorder=3)
for j in range(3):
    for i in range(len(methods)):
        a2.text(x[i] + (j - 1) * wbar, err[i, j] + 1, f"{err[i, j]:g}", fontsize=6.5, color=INK2, ha="center")
a1.set_yscale("log"); a1.set_ylim(0.05, 400)
a1.set_ylabel("running time per scene (s, log)"); a2.set_ylabel("error vs hand-computed ground truth (%)")
for a in (a1, a2):
    a.set_xticks(x); a.set_xticklabels(methods, fontsize=7.5)
    clean(a)
a1.set_title("Under 0.25 s versus minutes", fontsize=10); a2.set_title("Error ≤ 1.2 % versus 9–67 %", fontsize=10)
a1.legend(loc="upper left", fontsize=8)
a2.set_ylim(0, 75)
fig.suptitle("Paper's Table I: three box scenes, 10 runs per method", fontsize=11, fontweight="semibold")
fig.tight_layout()
save(fig, f"{SLUG}-3.png")
