"""Figures for mosemann1997-stability-assemblies-friction.md.

Everything here is computed: the paper's Problem-1 LP (min sum f_n subject to
equilibrium, f_n >= 0, linearised Coulomb cone) is solved with scipy.linprog
for a sweep of gravity directions, and the feasible directions are drawn.
Run from docs/figures/:  python mosemann1997-stability-assemblies-friction.py
"""
from _style import *  # noqa: F401,F403

import numpy as np
from matplotlib.patches import Polygon, Wedge, Circle
from scipy.optimize import linprog

SLUG = "mosemann1997-stability-assemblies-friction"
MG = 1.0  # weight of the free block (units of m|g|); only the direction of g matters

# ---------------------------------------------------------------- 2D toys
# Free block B: width 2, height 1, bottom-left corner at the origin, c.o.m. (1, 0.5).
W, H = 2.0, 1.0
COM = np.array([1.0, 0.5])
FLOOR = [((0, 0), (0, 1)), ((W, 0), (0, 1))]                 # (point, inward normal on B)
LWALL = [((0, 0), (1, 0)), ((0, H), (1, 0))]
RWALL = [((W, 0), (-1, 0)), ((W, H), (-1, 0))]
TOYS = {
    "flat": FLOOR,
    "corner": FLOOR + LWALL,
    "slot": FLOOR + LWALL + RWALL,
}


def lp2d(contacts, g_dir, mu, cap=None):
    """Problem 1 of the paper in 2D. Returns (feasible, forces[(fn, ft)...]).

    Unknowns per contact: f_n >= 0 and f_t with |f_t| <= mu f_n (the 2D cone is
    exactly two half-planes, so no pyramid approximation is needed).
    Equilibrium: sum of forces + m g = 0, sum of torques about c.o.m. = 0.
    Optional cap: sum f_n <= cap * m|g| (the paper's force-limit filter).
    """
    n = len(contacts)
    A_eq = np.zeros((3, 2 * n))
    for i, (p, nrm) in enumerate(contacts):
        p, nrm = np.asarray(p, float), np.asarray(nrm, float)
        t = np.array([-nrm[1], nrm[0]])
        r = p - COM
        A_eq[0:2, 2 * i] = nrm
        A_eq[0:2, 2 * i + 1] = t
        A_eq[2, 2 * i] = r[0] * nrm[1] - r[1] * nrm[0]
        A_eq[2, 2 * i + 1] = r[0] * t[1] - r[1] * t[0]
    b_eq = np.array([-MG * g_dir[0], -MG * g_dir[1], 0.0])
    A_ub, b_ub = [], []
    for i in range(n):
        row = np.zeros(2 * n); row[2 * i] = -mu; row[2 * i + 1] = 1; A_ub.append(row); b_ub.append(0)
        row = np.zeros(2 * n); row[2 * i] = -mu; row[2 * i + 1] = -1; A_ub.append(row); b_ub.append(0)
    if cap is not None:
        row = np.zeros(2 * n); row[0::2] = 1; A_ub.append(row); b_ub.append(cap * MG)
    c = np.zeros(2 * n); c[0::2] = 1  # minimise sum f_n
    bounds = [(0, None), (None, None)] * n
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    return res.status == 0, (res.x.reshape(n, 2) if res.status == 0 else None)


def stable_mask(contacts, mu, angles_deg, cap=None):
    out = []
    for a in angles_deg:
        th = np.radians(a)
        out.append(lp2d(contacts, np.array([np.sin(th), -np.cos(th)]), mu, cap)[0])
    return np.array(out)


ANG = np.arange(0.0, 360.0, 0.5)  # gravity tilt from straight down, clockwise positive


def sanity_checks():
    # Block on a slope slips exactly at tan(theta) = mu (for mu below the tipping limit).
    for mu in (0.3, 0.6):
        m = stable_mask(TOYS["flat"], mu, ANG)
        stable = ANG[m]
        span = stable[(stable <= 180)].max()
        assert abs(span - np.degrees(np.arctan(mu))) <= 0.5, (mu, span)
    # Tipping limit for a large mu: gravity line through c.o.m. reaches the corner at atan(W/H).
    m = stable_mask(TOYS["flat"], 10.0, ANG)
    tip = ANG[m & (ANG <= 180)].max()
    assert abs(tip - np.degrees(np.arctan(W / H))) <= 0.5, tip
    # Solved forces balance gravity.
    ok, f = lp2d(TOYS["flat"], np.array([0.2, -0.98]), 0.3)
    tot = np.zeros(2)
    for (p, nrm), (fn, ft) in zip(TOYS["flat"], f):
        nrm = np.asarray(nrm, float); t = np.array([-nrm[1], nrm[0]])
        tot += fn * nrm + ft * t
    assert np.allclose(tot, -MG * np.array([0.2, -0.98]), atol=1e-8)
    print("sanity checks passed")


sanity_checks()


def draw_toy(ax, name, contacts, g_dir, mu, title):
    """Block B on/in fixed block A, contact points, friction cones, solved forces."""
    # Fixed block A (floor plus optional walls).
    ax.add_patch(Polygon([(-1.2, -0.7), (3.2, -0.7), (3.2, 0), (-1.2, 0)], fc=STONE, ec=INK2, lw=0.8))
    if name in ("corner", "slot"):
        ax.add_patch(Polygon([(-1.2, 0), (0, 0), (0, 1.5), (-1.2, 1.5)], fc=STONE, ec=INK2, lw=0.8))
    if name == "slot":
        ax.add_patch(Polygon([(2, 0), (3.2, 0), (3.2, 1.5), (2, 1.5)], fc=STONE, ec=INK2, lw=0.8))
    ax.add_patch(Polygon([(0, 0), (W, 0), (W, H), (0, H)], fc=WOOD, ec=INK2, lw=1.0))
    ax.plot(*COM, "o", color=INK, ms=4)
    ax.text(COM[0] + 0.1, COM[1] + 0.05, "B", color=INK, fontsize=9)
    ax.text(-0.6, -0.4, "A (fixed)", color=INK2, fontsize=8, ha="center")
    ok, f = lp2d(contacts, g_dir, mu)
    scale = 0.9
    for (p, nrm), (fn, ft) in zip(contacts, f):
        p = np.asarray(p, float); nrm = np.asarray(nrm, float); t = np.array([-nrm[1], nrm[0]])
        # friction cone (half-angle atan mu) drawn as a wedge behind the contact
        ang = np.degrees(np.arctan2(nrm[1], nrm[0])); half = np.degrees(np.arctan(mu))
        ax.add_patch(Wedge(p, 0.45, ang - half, ang + half, fc=AQUA, ec="none", alpha=0.45, zorder=3))
        ax.plot(*p, "o", color=INK2, ms=3.5, zorder=4)
        F = fn * nrm + ft * t
        if np.linalg.norm(F) > 1e-6:
            ax.annotate("", xy=p + scale * F, xytext=p,
                        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2, shrinkA=0, shrinkB=0), zorder=5)
    ax.annotate("", xy=COM + 0.8 * g_dir, xytext=COM,
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2, shrinkA=0, shrinkB=0), zorder=5)
    ax.text(*(COM + 0.8 * g_dir + np.array([0.12, 0])), "mg", color=INK, fontsize=9, va="center")
    ax.set_xlim(-1.3, 3.3); ax.set_ylim(-0.8, 1.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    clean(ax, hide_axes=True)


def draw_arc(ax, masks, labels, title):
    """Stable gravity directions on a circle: each mask drawn as a ring band."""
    ax.add_patch(Circle((0, 0), 1.0, fc="none", ec=GRID, lw=1))
    radii = [1.0, 0.8, 0.6]
    for k, (mask, lab) in enumerate(zip(masks, labels)):
        r_out, r_in = radii[k], radii[k] - 0.16
        # bands of consecutive stable angles
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        # group runs, treating the array as circular
        runs, start = [], idx[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if b != a + 1:
                runs.append((start, a)); start = b
        runs.append((start, idx[-1]))
        if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == len(mask) - 1:
            runs = [(runs[-1][0], runs[0][1] + len(mask))] + runs[1:-1]
        for a, b in runs:
            a1 = ANG[a % len(ANG)]; a2 = ANG[b % len(ANG)] + (360 if b >= len(ANG) else 0)
            # g = (sin th, -cos th): a tilt th from straight down is matplotlib angle th - 90
            t1, t2 = a1 - 90 - 0.25, a2 - 90 + 0.25
            if a2 - a1 <= 0.6:  # a single direction: draw as a spoke
                ax.plot([0.9 * r_in * np.cos(np.radians(t1 + 0.25)), r_out * np.cos(np.radians(t1 + 0.25))],
                        [0.9 * r_in * np.sin(np.radians(t1 + 0.25)), r_out * np.sin(np.radians(t1 + 0.25))],
                        color=GOOD, lw=3, solid_capstyle="butt")
            else:
                ax.add_patch(Wedge((0, 0), r_out, t1, t2, width=r_out - r_in, fc=GOOD, ec="none", alpha=0.85))
        frac = mask.mean() * 360
        txt = f"{lab}: {frac:.0f}°" if frac < 359.9 else f"{lab}: 360° (all)"
        if frac < 1:
            txt = f"{lab}: one direction"
        y = -1.42 - 0.24 * k
        ax.text(-0.95, y, txt, fontsize=8.5, color=INK, ha="left", va="center")
        ax.plot([-1.25, -1.05], [y, y], color=GOOD, lw=4, solid_capstyle="butt", alpha=0.85)
    ax.annotate("", xy=(0, -0.5), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.5))
    ax.text(0.06, -0.28, "g", fontsize=9, color=INK2)
    for a, lab in [(0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")]:
        t = np.radians(a - 90)
        ax.text(1.16 * np.cos(t), 1.16 * np.sin(t), lab, fontsize=7.5, color=MUTED, ha="center", va="center")
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-2.2, 1.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    clean(ax, hide_axes=True)


# ---------------------------------------------------------------- Figure 1
fig, axes = plt.subplots(2, 3, figsize=(8, 6.0), gridspec_kw=dict(height_ratios=[1, 1.6]))
g_demo = np.array([np.sin(np.radians(12)), -np.cos(np.radians(12))])
titles = ["B on a flat A", "B in a corner of A", "B in a slot of A"]
for j, (name, contacts) in enumerate(TOYS.items()):
    draw_toy(axes[0, j], name, contacts, g_demo, 0.3, titles[j])
    m0 = stable_mask(contacts, 0.0, ANG)
    m3 = stable_mask(contacts, 0.3, ANG)
    masks, labels = [m0, m3], ["μ = 0", "μ = 0.3"]
    if name == "slot":
        masks.append(stable_mask(contacts, 0.3, ANG, cap=3.0)); labels.append("μ = 0.3, Σfₙ ≤ 3mg")
    draw_arc(axes[1, j], masks, labels, "LP-feasible directions of g")
fig.suptitle("Friction turns a point into an arc; parallel walls let the LP call every direction stable",
             fontsize=11, fontweight="semibold", y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=3.0)
y_mid = 0.5 * (axes[0, 0].get_position().y0 + axes[1, 0].get_position().y1) + 0.02
fig.text(0.5, y_mid, "Row 1: contact points, friction cones (green) and LP contact forces (blue) for g tilted 12°, μ = 0.3.\n"
         "Row 2: outer band μ = 0, middle band μ = 0.3, inner band μ = 0.3 with the force cap.",
         ha="center", va="center", fontsize=8, color=INK2)
save(fig, f"{SLUG}-1.png")

# ---------------------------------------------------------------- Figure 3: stable fraction vs mu / cap
mus = np.concatenate([[0.0], np.geomspace(0.02, 5.0, 40)])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.4))
colors = [BLUE, ORANGE, AQUA]
for (name, contacts), col in zip(TOYS.items(), colors):
    fr = [stable_mask(contacts, mu, ANG).mean() * 360 for mu in mus]
    ax1.plot(mus, fr, color=col, label=name)
for mu in (0.95, 0.99, 1.0, 1.01, 1.05):
    print(f"corner mu={mu}: {stable_mask(TOYS['corner'], mu, ANG).mean() * 360:.0f} deg")
tip = 2 * np.degrees(np.arctan(W / H))
ax1.axhline(tip, color=MUTED, lw=1, ls="--")
ax1.text(0.021, tip + 6, f"flat block tips beyond ±{tip / 2:.0f}° (= atan(W/H))", fontsize=7.5, color=INK2)
ax1.set_xscale("log"); ax1.set_xlim(0.02, 5); ax1.set_ylim(0, 375)
ax1.set_xlabel("friction coefficient μ (–)"); ax1.set_ylabel("stable gravity directions (degrees of 360)")
ax1.set_title("Stable arc grows as 2·atan μ until tipping", fontsize=10)
ax1.legend(loc="center left", title=None)
clean(ax1)

caps = np.linspace(1.0, 10.0, 46)
fr_cap = [stable_mask(TOYS["slot"], 0.3, ANG, cap=c).mean() * 360 for c in caps]
ax2.plot(caps, fr_cap, color=AQUA, label="slot, μ = 0.3")
k_star = 2 / 0.3
ax2.axvline(k_star, color=MUTED, lw=1, ls="--")
ax2.text(k_star + 0.15, 200, "upside-down needs\nΣfₙ = 2mg/μ = 6.7 mg\nof wall squeeze", fontsize=7.5, color=INK2)
ax2.set_xlabel("force cap  Σfₙ / m|g| (–)"); ax2.set_ylabel("stable gravity directions (degrees of 360)")
ax2.set_ylim(0, 375); ax2.set_xlim(1, 10)
ax2.set_title("In the slot the answer depends on the cap", fontsize=10)
ax2.legend(loc="lower right")
clean(ax2)
fig.tight_layout()
save(fig, f"{SLUG}-3.png")

# ---------------------------------------------------------------- Figure 2: 3D sphere sweep with an 8-facet pyramid
# Block 2 x 1 x 1 (x, y, z) sitting on a floor (y = 0) against a wall (x = 0): the paper's
# L-bracket case. Contacts at the 4 floor corners and the 4 wall corners.
BOX = np.array([2.0, 1.0, 1.0])
COM3 = BOX / 2
CONTACTS3 = []
for x in (0, BOX[0]):
    for z in (0, BOX[2]):
        CONTACTS3.append((np.array([x, 0, z]), np.array([0, 1, 0])))       # floor
for y in (0, BOX[1]):
    for z in (0, BOX[2]):
        CONTACTS3.append((np.array([0, y, z]), np.array([1, 0, 0])))       # wall


def frame(nrm):
    a = np.array([0, 0, 1.0]) if abs(nrm[2]) < 0.9 else np.array([1.0, 0, 0])
    tx = np.cross(nrm, a); tx /= np.linalg.norm(tx)
    ty = np.cross(nrm, tx)
    return tx, ty


def lp3d(g, mu, facets=8):
    """Problem 1 with the friction cone replaced by an inscribed l-facet pyramid."""
    n = len(CONTACTS3)
    A_eq = np.zeros((6, 3 * n))
    for i, (p, nrm) in enumerate(CONTACTS3):
        tx, ty = frame(nrm)
        r = p - COM3
        for k, d in enumerate((nrm, tx, ty)):
            A_eq[0:3, 3 * i + k] = d
            A_eq[3:6, 3 * i + k] = np.cross(r, d)
    b_eq = np.concatenate([-MG * g, np.zeros(3)])
    rows = []
    for i in range(n):
        for k in range(facets):
            ph = 2 * np.pi * k / facets
            row = np.zeros(3 * n)
            # facet k of the inscribed pyramid: cos(ph) ftx + sin(ph) fty <= mu cos(pi/l) fn
            row[3 * i] = -mu * np.cos(np.pi / facets); row[3 * i + 1] = np.cos(ph); row[3 * i + 2] = np.sin(ph)
            rows.append(row)
    c = np.zeros(3 * n); c[0::3] = 1
    bounds = [(0, None), (None, None), (None, None)] * n
    res = linprog(c, A_ub=np.array(rows), b_ub=np.zeros(len(rows)), A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    return res.status == 0


# Sample the sphere on a lat/lon grid that contains the great circle g_z = 0 exactly.
lats = np.radians(np.arange(-90, 90.01, 2.0))
lons = np.radians(np.arange(0, 360, 2.0))
mu3 = 0.3
stab = np.zeros((len(lats), len(lons)), bool)
for i, la in enumerate(lats):
    for j, lo in enumerate(lons):
        g = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        g[np.abs(g) < 1e-12] = 0.0
        stab[i, j] = lp3d(g, mu3)
arc0 = np.array([lp3d(np.array([np.cos(lo), np.sin(lo), 0.0]), 0.0) for lo in lons])
print("mu=0 arc:", arc0.sum() * 2, "deg;  mu=0.3 belt covers", stab.mean() * 100, "% of samples")

# Orthographic view from direction v (looking at the -x,-y side, where the stable belt lives).
view = np.array([-1.0, -1.0, 0.6]); view /= np.linalg.norm(view)
up = np.array([0, 0, 1.0]); right = np.cross(up, view); right /= np.linalg.norm(right); up2 = np.cross(view, right)


def proj(p):
    p = np.asarray(p, float)
    return p @ right, p @ up2, p @ view


fig, ax = plt.subplots(figsize=(6.4, 5.2))
ax.add_patch(Circle((0, 0), 1, fc=SURFACE, ec=INK2, lw=0.8))
# graticule
for la in np.radians(np.arange(-60, 61, 30)):
    pts = np.array([proj([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]) for lo in np.linspace(0, 2 * np.pi, 181)])
    vis = pts[:, 2] > 0
    ax.plot(np.where(vis, pts[:, 0], np.nan), np.where(vis, pts[:, 1], np.nan), color=GRID, lw=0.8)
for lo in np.radians(np.arange(0, 360, 30)):
    pts = np.array([proj([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]) for la in np.linspace(-np.pi / 2, np.pi / 2, 91)])
    vis = pts[:, 2] > 0
    ax.plot(np.where(vis, pts[:, 0], np.nan), np.where(vis, pts[:, 1], np.nan), color=GRID, lw=0.8)
# stable samples (mu = 0.3) as small dots on the visible hemisphere
LA, LO = np.meshgrid(lats, lons, indexing="ij")
G = np.stack([np.cos(LA) * np.cos(LO), np.cos(LA) * np.sin(LO), np.sin(LA)], -1)
P = G.reshape(-1, 3) @ np.stack([right, up2, view], 1)
vis = P[:, 2] > 0
st = stab.reshape(-1)
ax.scatter(P[vis & st, 0], P[vis & st, 1], s=6, color=GOOD, lw=0, label=f"stable, μ = {mu3} (8-facet pyramid)", zorder=3)
ax.scatter(P[vis & ~st, 0], P[vis & ~st, 1], s=2.5, color=STONE, lw=0, label="unstable", zorder=2)
# mu = 0 arc
pts = np.array([proj([np.cos(lo), np.sin(lo), 0]) for lo in lons[arc0]])
order = np.argsort(np.degrees(lons[arc0]))
pts = pts[order]
vis0 = pts[:, 2] > 0
ax.plot(np.where(vis0, pts[:, 0], np.nan), np.where(vis0, pts[:, 1], np.nan), color=INK, lw=2.5,
        label="stable, μ = 0 (great-arc segment)", zorder=4)
# axis labels on the sphere
for name, v in [("−x (into wall)", [-1, 0, 0]), ("−y (into floor)", [0, -1, 0]), ("+z", [0, 0, 1])]:
    x, y, d = proj(v)
    r = np.hypot(x, y)
    ax.annotate(name, (x, y), xytext=(1.18 * x / r, 1.12 * y / r), fontsize=8.5,
                color=INK2, ha="center", va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.45, 1.3)
ax.set_title("Block in an L-bracket: the frictionless great arc widens into a belt (μ = 0.3)", fontsize=10)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=8, markerscale=1.8)
clean(ax, equal=True, hide_axes=True)
fig.tight_layout()
save(fig, f"{SLUG}-2.png")
