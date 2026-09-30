"""Figures for docs/ganeshan2025-migumi.md.

Figure 1: morphological opening (Sec. 3.2, Fig. 4) computed exactly with polygon
offsets (shapely buffers), on a pocket profile with a slot narrower than the bit,
and the square-peg / square-pocket overlap that opening alone (MO) leaves.
Figure 2: a two-parameter toy of the optimisation (Sec. 4): the Surface Gap (Eq. 3)
and Milling Path Distance (Eq. 7) landscapes over the corner radii of the two mill
paths, with gradient-descent trajectories with and without the path-distance term.
Figure 3: the corner geometry at the start and end of those descents.
Figure 4: Table 1 (main comparison) and Table 2 (ablation).
Run from docs/figures/:  python ganeshan2025-migumi.py
"""
from _style import *  # noqa: F401,F403

import numpy as np
import shapely
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon, box

np.random.seed(0)
R = 3.175          # bit radius used in all experiments (1/8 in), mm
QS = 64            # arc resolution of polygon offsets


# ----------------------------------------------------------------------------
# Exact 2D morphology with polygon offsets: erosion = buffer(-r), dilation = buffer(+r)
# ----------------------------------------------------------------------------
def erode(g, r):
    return g.buffer(-r, quad_segs=QS)


def dilate(g, r):
    return g.buffer(r, quad_segs=QS)


def opening(g, r):
    return dilate(erode(g, r), r)


def fill(ax, geom, color, alpha=1.0, z=1):
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    for p in polys:
        if p.is_empty:
            continue
        verts, codes = [], []
        for ring in [p.exterior] + list(p.interiors):
            xy = np.array(ring.coords)
            verts.extend(xy); codes.extend([Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY])
        ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=z))


def edge(ax, geom, color, lw=1.4, ls="-", z=3):
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    for p in polys:
        if p.is_empty:
            continue
        for ring in [p.exterior] + list(p.interiors):
            xy = np.array(ring.coords)
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, ls=ls, zorder=z)


def figure1():
    # (a-c) pocket with a slot narrower than the bit diameter
    f = box(-8, -6, 8, 4).union(box(-2, 4, 2, 12))
    g = erode(f, R)
    C = opening(f, R)
    lost = f.difference(C)
    print("Fig1 profile area %.2f mm^2, opened %.2f, lost %.2f (slot 32 + corners %.2f, minus the reachable slot mouth)"
          % (f.area, C.area, lost.area, 4 * (1 - np.pi / 4) * R * R))
    # (d) square peg in square pocket
    pocket = box(-8, -8, 8, 8)
    pocket_milled = opening(pocket, R)
    big = box(-40, -40, 40, 40)
    ring = big.difference(pocket)            # B's cut: everything around the peg
    ring_milled = opening(ring, R)           # unchanged: the bit wraps around a convex corner
    peg = box(-12, -12, 12, 12).difference(ring_milled)   # clip away the big box's own rounded corners
    overlap = peg.difference(pocket_milled)
    print("Fig1 peg unchanged by opening: %s  overlap area %.3f mm^2 (analytic (4-pi) r^2 = %.3f)"
          % (abs(peg.area - pocket.area) < 1e-2, overlap.area, (4 - np.pi) * R * R))
    assert abs(overlap.area - (4 - np.pi) * R * R) < 0.02

    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.9))
    for ax in axes:
        clean(ax, equal=True, hide_axes=True)
        ax.set_xlim(-11, 11); ax.set_ylim(-11.5, 13.5)
    ax = axes[0]
    fill(ax, f, STONE); edge(ax, f, INK2)
    ax.annotate("", xy=(-2, 8), xytext=(2, 8), arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
    ax.text(0, 9.3, "4 mm < 2r", ha="center", fontsize=7.5, color=INK)
    ax.text(0, -9.2, "cut region f, r = 0", ha="center", fontsize=8.5, color=INK, va="top")
    ax.set_title("(a) designed", fontsize=9)

    ax = axes[1]
    fill(ax, f, STONE); edge(ax, f, INK2, ls="--", lw=1)
    fill(ax, g, SEQ[1], z=2); edge(ax, g, BLUE)
    cx, cy = 4.825, 0.825
    ax.add_patch(plt.Circle((cx, cy), R, facecolor="none", edgecolor=INK, lw=0.9, ls=":", zorder=4))
    ax.plot([cx], [cy], ".", color=INK, ms=4, zorder=5)
    ax.text(0, -9.2, "erode by r = 3.175 mm:\nslot too narrow, it vanishes", ha="center", fontsize=8.5, color=INK, va="top")
    ax.set_title("(b) g = f ⊖ B_r", fontsize=9)

    ax = axes[2]
    fill(ax, C, STONE); fill(ax, lost, ORANGE, z=2)
    edge(ax, f, INK2, ls="--", lw=1); edge(ax, C, INK2)
    ax.text(0, -9.2, "lost %.1f of %.0f mm²" % (lost.area, f.area), ha="center", fontsize=8.5, color=INK, va="top")
    ax.set_title("(c) C = g ⊕ B_r", fontsize=9)

    ax = axes[3]
    dy = 2.5  # draw panel (d) level with the others
    stockA = translate(box(-10.5, -10.5, 10.5, 10.5).difference(pocket_milled), 0, dy)
    fill(ax, stockA, WOOD); fill(ax, translate(peg, 0, dy), WOOD2, z=2); fill(ax, translate(overlap, 0, dy), ORANGE, z=3)
    edge(ax, translate(pocket_milled, 0, dy), INK2); edge(ax, translate(peg, 0, dy), INK2)
    ax.text(0, -9.2, "overlap 4 × %.2f mm²" % (overlap.area / 4), ha="center", fontsize=8.5, color=INK, va="top")
    ax.set_title("(d) MO: peg + pocket", fontsize=9)
    handles = [Patch(facecolor=STONE, edgecolor=INK2, label="cut region"), Patch(facecolor=SEQ[1], edgecolor=BLUE, label="eroded region (mill path = its edge)"),
               Patch(facecolor=WOOD, edgecolor=INK2, label="part A (pocket)"), Patch(facecolor=WOOD2, edgecolor=INK2, label="part B (peg, unchanged)"),
               Patch(facecolor=ORANGE, edgecolor=INK2, label="unreachable / overlap")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), fontsize=7.5, columnspacing=1.2)
    fig.suptitle("Opening keeps only what a round bit can clear; applied part by part it breaks the fit",
                 fontsize=10.5, fontweight="semibold", y=1.0)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, wspace=0.05)
    save(fig, "ganeshan2025-migumi-1.png")


# ----------------------------------------------------------------------------
# Figure 2/3: two-parameter toy of the optimisation
# ----------------------------------------------------------------------------
W = 8.0  # designed half-width of the peg / pocket


def rounded_square(half, rho, n_arc=24):
    """Closed polyline of a square of half-width `half` with corner radius rho (0 = sharp)."""
    rho = float(np.clip(rho, 0.0, half))
    pts = []
    for cx, cy, a0 in [(half - rho, half - rho, 0), (-(half - rho), half - rho, 90),
                       (-(half - rho), -(half - rho), 180), (half - rho, -(half - rho), 270)]:
        for a in np.linspace(a0, a0 + 90, n_arc if rho > 0 else 2):
            pts.append((cx + rho * np.cos(np.radians(a)), cy + rho * np.sin(np.radians(a))))
    return np.array(pts)


def walls(rhoA, rhoB, r=R):
    """Mill paths -> walls.  Pocket path: half-width W - r, corner rhoA; dilating by r gives the pocket wall
    (half-width W, corner rhoA + r).  Ring path (the cut around the peg) has inner boundary half-width W + r,
    corner rhoB; dilating the ring by r moves that boundary inward: half-width W, corner max(rhoB - r, 0)."""
    pathA = rounded_square(W - r, rhoA)
    pathB = rounded_square(W + r, rhoB)
    wallA = rounded_square(W, rhoA + r)
    wallB = rounded_square(W, max(rhoB - r, 0.0))
    return pathA, pathB, wallA, wallB


def contour_integral(src, dst, fn):
    """sum over the points of polyline src of fn(distance to polyline dst) * ds."""
    ring = shapely.LinearRing(dst)
    d = shapely.distance(shapely.points(src), ring)
    seg = np.linalg.norm(np.roll(src, -1, axis=0) - src, axis=1)
    ds = 0.5 * (seg + np.roll(seg, 1))
    return float(np.sum(fn(d) * ds))


def losses(rhoA, rhoB, r=R):
    pathA, pathB, wallA, wallB = walls(rhoA, rhoB, r)
    MS = contour_integral(wallA, wallB, lambda d: d) + contour_integral(wallB, wallA, lambda d: d)      # Eq. 3, mm^2
    MP = contour_integral(pathA, pathB, lambda d: (d - 2 * r) ** 2) + contour_integral(pathB, pathA, lambda d: (d - 2 * r) ** 2)  # Eq. 7, mm^3
    k = 4 * (1 - np.pi / 4)
    Locc = k * ((rhoA + r) ** 2 + max(rhoB - r, 0.0) ** 2)   # area changed vs the sharp design, mm^2
    return MS, MP, Locc


def descend(start, lamP, lamOcc=0.02, lr=0.12, iters=160, h=1e-3):
    """Adam on the two corner radii with finite-difference gradients, parameters clipped to their ranges."""
    p = np.array(start, float)
    m = np.zeros(2); v = np.zeros(2)
    lo = np.array([0.0, 0.0]); hi = np.array([2 * R, 4 * R])
    path = [p.copy()]

    def L(q):
        MS, MP, Lo = losses(*q)
        return MS + lamP * MP + lamOcc * Lo

    for it in range(1, iters + 1):
        g = np.zeros(2)
        for i in range(2):
            e = np.zeros(2); e[i] = h
            qp = np.clip(p + e, lo, hi); qm = np.clip(p - e, lo, hi)
            g[i] = (L(qp) - L(qm)) / max(qp[i] - qm[i], 1e-12)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mh = m / (1 - 0.9 ** it); vh = v / (1 - 0.999 ** it)
        p = np.clip(p - lr * mh / (np.sqrt(vh) + 1e-8), lo, hi)
        path.append(p.copy())
    return np.array(path)


START = (R / 2, R / 2)   # a start inside the plateau, e.g. paths carried over from a smaller bit radius
LAM_P = 1.0


def figure2():
    rA = np.linspace(0, 2 * R, 41)
    rB = np.linspace(0, 4 * R, 61)
    MS = np.zeros((len(rB), len(rA))); MP = np.zeros_like(MS)
    for i, b in enumerate(rB):
        for j, a in enumerate(rA):
            MS[i, j], MP[i, j], _ = losses(a, b)
    iB, jA = np.unravel_index(MS.argmin(), MS.shape)
    print("Fig2 landscape: min M_S = %.3g at rhoB - rhoA = %.2f (2r = %.2f); plateau check M_S(0,0)=%.2f M_S(0,r)=%.2f"
          % (MS.min(), rB[iB] - rA[jA], 2 * R, MS[0, 0], MS[np.argmin(abs(rB - R)), 0]))
    traj_S = descend(START, lamP=0.0)
    traj_SP = descend(START, lamP=LAM_P)
    eS, ePS = losses(*traj_S[-1]), losses(*traj_SP[-1])
    print("Fig2 L_S only: end", traj_S[-1].round(3), "M_S=%.3f" % eS[0], " | with L_P: end", traj_SP[-1].round(3), "M_S=%.4f" % ePS[0])
    assert eS[0] > 1.0 and ePS[0] < 0.05 and abs(traj_SP[-1][1] - traj_SP[-1][0] - 2 * R) < 0.05

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)
    cmap = LinearSegmentedColormap.from_list("seq", [SURFACE] + SEQ)
    for ax, Z, name, unit in [(axes[0], MS, "Surface Gap M_S (Eq. 3)", "M_S (mm²)"), (axes[1], MP, "Milling Path Distance M_P (Eq. 7)", "M_P (mm³)")]:
        clean(ax)
        cf = ax.contourf(rA, rB, Z, levels=16, cmap=cmap, norm=PowerNorm(0.5))
        cb = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.06)
        cb.set_label(unit, fontsize=8); cb.ax.tick_params(labelsize=7)
        ax.plot(rA, rA + 2 * R, color=INK, lw=1, ls="--", label="ρ_B = ρ_A + 2r: walls coincide")
        ax.axhline(R, color=INK2, lw=0.8, ls=":")
        ax.plot(traj_S[:, 0], traj_S[:, 1], "-", color=ORANGE, lw=2, label="descend L_S + λ_occ L_occ")
        ax.plot(traj_SP[:, 0], traj_SP[:, 1], "-", color=BLUE, lw=2, label="descend L_S + λ_P L_P + λ_occ L_occ")
        ax.plot(*START, "o", color=INK, ms=5, mfc=SURFACE, zorder=5)
        ax.plot(*traj_S[-1], "s", color=ORANGE, ms=6, zorder=6); ax.plot(*traj_SP[-1], "s", color=BLUE, ms=6, zorder=6)
        ax.set_xlabel("ρ_A: pocket mill-path corner radius (mm)")
        ax.set_xlim(0, 2 * R); ax.set_ylim(0, 4 * R)
        ax.set_title(name, fontsize=9.5)
    axes[0].set_ylabel("ρ_B: peg mill-path corner radius (mm)")
    axes[0].text(2 * R - 0.2, R - 0.3, "ρ_B < r: peg wall stays sharp,\nM_S flat in ρ_B", fontsize=7, color=INK, va="top", ha="right",
                 bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5, alpha=0.85))
    axes[0].text(START[0] + 0.25, START[1] + 0.25, "start", fontsize=7.5, color=INK)
    axes[1].text(2 * R - 0.2, R - 0.3, "M_P still slopes\ntoward the dashed line", fontsize=7, color=INK, va="top", ha="right",
                 bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5, alpha=0.85))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04), fontsize=7.5)
    fig.suptitle("The surface gap alone stalls on its plateau; adding the path-distance term reaches zero gap",
                 fontsize=10, fontweight="semibold", y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    save(fig, "ganeshan2025-migumi-2.png")
    return traj_S, traj_SP


def figure3(traj_S, traj_SP):
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.9))
    cases = [(START, "start: ρ_A = ρ_B = r/2"), (tuple(traj_S[-1]), "end of descent, L_S + L_occ only"),
             (tuple(traj_SP[-1]), "end of descent, with λ_P L_P")]
    for ax, ((a, b), title) in zip(axes, cases):
        clean(ax, equal=True, hide_axes=True)
        pathA, pathB, wallA, wallB = walls(a, b)
        A_region = box(W - 12, W - 12, W + 6, W + 6).difference(Polygon(wallA))   # part A material around the pocket
        B_region = Polygon(wallB)                                                    # peg
        fill(ax, A_region, WOOD); fill(ax, B_region, WOOD2, z=2)
        ov = B_region.difference(Polygon(wallA)); gap = Polygon(wallA).difference(B_region)
        fill(ax, ov, ORANGE, z=3); fill(ax, gap, STONE, z=3)
        ax.plot(*wallA.T, color=BLUE, lw=2, zorder=4); ax.plot(*wallB.T, color=ORANGE, lw=1.4, ls=(0, (2.5, 1.5)), zorder=5)
        ax.plot(*pathA.T, color=INK2, lw=0.8, zorder=4); ax.plot(*pathB.T, color=INK2, lw=0.8, zorder=4)
        MSv = losses(a, b)[0]
        ax.text(W - 5.4, W + 4.6, "ρ_A = %.2f, ρ_B = %.2f mm\nM_S = %.2f mm², overlap %.2f mm²" % (a, b, MSv, ov.area),
                fontsize=7.5, color=INK, va="top")
        ax.set_xlim(W - 5.6, W + 5.2); ax.set_ylim(W - 5.6, W + 5.2)
        ax.set_title(title, fontsize=8.5)
    handles = [plt.Line2D([], [], color=BLUE, lw=2, label="pocket wall (A) = pocket path ⊕ B_r"),
               plt.Line2D([], [], color=ORANGE, lw=1.4, ls=(0, (2.5, 1.5)), label="peg wall (B) = ring path ⊕ B_r"),
               plt.Line2D([], [], color=INK2, lw=0.8, label="mill paths ∂g"),
               Patch(facecolor=ORANGE, label="overlap"), Patch(facecolor=STONE, edgecolor=INK2, label="gap")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08), fontsize=7.5, columnspacing=1.0)
    fig.suptitle("One corner of the peg-in-pocket toy: the fix is to round the peg to the pocket's radius",
                 fontsize=10, fontweight="semibold", y=1.0)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.03, wspace=0.05)
    save(fig, "ganeshan2025-migumi-3.png")


# ----------------------------------------------------------------------------
# Figure 4: Tables 1 and 2
# ----------------------------------------------------------------------------
def figure4():
    fig = plt.figure(figsize=(8.0, 3.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.0])
    ax = fig.add_subplot(gs[0]); clean(ax)
    methods = ["MO", "ODF", "MiGumi"]
    M = [100, 25, 100]; C = [6.25, 96.87, 90.62]
    x = np.arange(3); wd = 0.32
    ax.bar(x - wd / 2 - 0.02, M, wd, color=BLUE, label="millable %M")
    ax.bar(x + wd / 2 + 0.02, C, wd, color=ORANGE, label="tightly coupled %C_τ")
    for xi, m, c in zip(x, M, C):
        ax.text(xi - wd / 2 - 0.02, m + 2, "%g" % m, ha="center", fontsize=7.5, color=INK2)
        ax.text(xi + wd / 2 + 0.02, c + 2, "%g" % c, ha="center", fontsize=7.5, color=INK2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8))
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8.5)
    ax.set_ylabel("share of the 30 designs (%)")
    ax.set_ylim(0, 132)
    ax.legend(loc="upper center", fontsize=7, ncol=1)
    ax.set_title("Table 1: only MiGumi gets both", fontsize=9.5)

    rows = ["full method", "no L_S", "no L_P", "no L_occ", "no gradual r rollout", "no ODF initialisation"]
    V = [74.82, 138.23, 120.56, 83.16, 142.11, 238.64]
    D = [456.14, 483.52, 474.60, 543.27, 488.96, 569.59]
    y = np.arange(len(rows))
    for k, (vals, lab, ttl) in enumerate([(V, "median violation volume V (mm³)", "Table 2: each ablation adds gap + overlap"),
                                          (D, "median design deviation D (mm³)", "and drifts further from the design")]):
        ax = fig.add_subplot(gs[k + 1]); clean(ax)
        cols = [ORANGE] + [BLUE] * 5
        ax.barh(y, vals, color=cols, height=0.62)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.015, yi, "%.1f" % v, va="center", fontsize=7.5, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels(rows if k == 0 else [""] * len(rows), fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.25)
        ax.set_xlabel(lab, fontsize=8)
        ax.set_title(ttl, fontsize=9)
    fig.tight_layout()
    save(fig, "ganeshan2025-migumi-4.png")


if __name__ == "__main__":
    figure1()
    tS, tSP = figure2()
    figure3(tS, tSP)
    figure4()
