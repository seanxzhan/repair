"""Figures for docs/wang2021-mocca.md (MOCCA, Wang, Song, Pauly 2021).

Everything is computed on small 2D joints with the paper's definitions:
generalized normals n = (n_hat, r x n_hat) sampled on the contact (Eq. 1), the
motion cone V = N* as the dual cone (Theorem 3.1) built by half-space
intersection, the motion-based equilibrium test (Eqs. 5-6) solved with linprog,
and the infeasibility measure E (Eq. 7) via its dual (Eq. 8) with NNLS.
Run from docs/figures/:  python wang2021-mocca.py
"""
from _style import *  # noqa: F401,F403

import numpy as np
from matplotlib.patches import Polygon, Wedge
from scipy.interpolate import CubicSpline
from scipy.optimize import linprog, nnls
from scipy.spatial import HalfspaceIntersection, ConvexHull

# --------------------------------------------------------------------------- joints
def vnotch(beta_deg, depth=1.0, m=25):
    """Wedge B seated in a V-notch of fixed part A. beta = flank angle from vertical.
    Returns (polygon of B, list of (r, n_hat) contact samples with n_hat pointing into B)."""
    b = np.radians(beta_deg); w = depth * np.tan(b)
    B = np.array([(0.0, 0.0), (w, depth), (-w, depth)])
    nL = np.array([np.cos(b), np.sin(b)]); nR = np.array([-np.cos(b), np.sin(b)])
    s = np.linspace(0, 1, m)
    left = [((-w * t, depth * t), nL) for t in s]
    right = [((w * t, depth * t), nR) for t in s]
    return B, left + right


def ntype(h, m=50, width=1.0):
    """MOCCA n-type joint: contact from p1=(0,0) to p2=(1,0) turned into a cubic-spline bump of height h
    (height field along u = +y), sampled with m generalized normals pointing into the upper part."""
    xs_c = np.array([0.0, 0.3, 0.5, 0.7, 1.0]) * width
    ys_c = np.array([0.0, 0.55, 1.0, 0.55, 0.0]) * h
    cs = CubicSpline(xs_c, ys_c, bc_type="clamped")
    x = np.linspace(0, width, m)
    y = cs(x); dy = cs(x, 1)
    nh = np.stack([-dy, np.ones_like(dy)], 1); nh /= np.linalg.norm(nh, axis=1, keepdims=True)
    return cs, [((x[k], y[k]), nh[k]) for k in range(m)]


def generalized_normals(samples):
    """n = (n_hat, r x n_hat) about the origin of the joint frame (Eq. 1)."""
    return np.array([[n[0], n[1], r[0] * n[1] - r[1] * n[0]] for r, n in samples])


def conic_section(Ngen, vy=1.0, box=6.0):
    """Motion cone V = {v : n.v >= 0} cut by the plane v_y = vy: polygon in (v_x, omega)."""
    # n_x vx + n_y vy + n_w w >= 0  ->  -n_x vx - n_w w - n_y vy <= 0   (HalfspaceIntersection uses A x + b <= 0)
    hs = [[-n[0], -n[2], -n[1] * vy] for n in Ngen]
    hs += [[1, 0, -box], [-1, 0, -box], [0, 1, -box], [0, -1, -box]]
    hsi = HalfspaceIntersection(np.array(hs, float), np.array([0.0, 0.0]))
    pts = hsi.intersections
    return pts[ConvexHull(pts).vertices]


def translational_halfangle(Ngen):
    """Widest |v_x| with v_y = 1 and omega = 0 -> insertion cone half-angle (deg)."""
    A = Ngen[:, :2]
    best = 0.0
    for sgn in (1, -1):
        r = linprog([-sgn, 0], A_ub=-A, b_ub=np.zeros(len(A)), A_eq=[[0, 1]], b_eq=[1], bounds=[(None, None), (None, None)], method="highs")
        if r.status == 0:
            best = max(best, abs(r.x[0]))
        else:
            return 90.0     # unbounded: a half-plane
    return np.degrees(np.arctan(best))


# --------------------------------------------------------------------------- statics of a single free part
def centroid_area(P):
    P = np.asarray(P, float); x, y = P[:, 0], P[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1); cr = x * yn - xn * y
    A = 0.5 * cr.sum()
    return np.array([((x + xn) * cr).sum() / (6 * A), ((y + yn) * cr).sum() / (6 * A)]), A


def wrench_matrix(samples, c):
    """Columns: unit compressive force at each sample, as (Fx, Fy, torque about the centroid c)."""
    return np.array([[n[0], n[1], (r[0] - c[0]) * n[1] - (r[1] - c[1]) * n[0]] for r, n in samples]).T


def motion_test(Aeq, w):
    """Eqs. 5-6: is there v with B_in v >= 0 (B_in = Aeq^T) and w^T v > 0?  Returns the max power (0 = equilibrium)."""
    B = Aeq.T
    r = linprog(-np.asarray(w), A_ub=-B, b_ub=np.zeros(B.shape[0]), bounds=[(-1, 1)] * 3, method="highs")
    return -r.fun


def infeasibility_E(Aeq, w):
    """Eq. 7 via its dual Eq. 8: E = min_{f>=0} 1/2 ||Aeq f + w||^2  (extra wrench s needed for balance)."""
    f, res = nnls(Aeq, -np.asarray(w))
    return 0.5 * res ** 2


def tilt_tolerance(samples, P, iters=40):
    """Largest gravity tilt phi (deg) with E = 0, by binary search on the motion-based test."""
    c, area = centroid_area(P)
    A = wrench_matrix(samples, c)
    def ok(phi):
        w = np.array([area * np.sin(np.radians(phi)), -area * np.cos(np.radians(phi)), 0.0])
        return motion_test(A, w) <= 1e-9
    if not ok(0.0):
        return 0.0
    lo, hi = 0.0, 180.0
    if ok(hi):
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return 0.5 * (lo + hi)


# =========================================================================== Figure 1: dual cone
fig, axs = plt.subplots(1, 3, figsize=(8, 3.3), gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.2})
beta = 30.0
B, samp = vnotch(beta)
N2 = np.array([n for r, n in samp])
# (a) geometry
ax = axs[0]
w = np.tan(np.radians(beta))
ax.add_patch(Polygon([(-1.2, -0.4), (1.2, -0.4), (1.2, 1.0), (w, 1.0), (0, 0), (-w, 1.0), (-1.2, 1.0)], closed=True, fc=STONE, ec=INK2, lw=1))
ax.add_patch(Polygon(B, closed=True, fc=WOOD, ec=INK2, lw=1, zorder=2))
for r, n in samp[::6]:
    ax.annotate("", xy=np.array(r) + 0.28 * n, xytext=r, arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, mutation_scale=8), zorder=3)
ax.text(-0.22, 0.82, "B", ha="center", fontsize=9, color=INK); ax.text(-1.1, 0.1, "A (fixed)", fontsize=8.5, color=INK)
ax.plot([0, 0], [0, 1.0], color=INK2, lw=0.8, ls=":")
ax.add_patch(Wedge((0, 0), 0.42, 90 - beta, 90, fc="none", ec=INK2, lw=0.8))
ax.text(0.17, 0.27, "β", fontsize=9, color=INK2)
ax.annotate("", xy=(0, 1.35), xytext=(0, 1.05), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
ax.text(0.08, 1.2, "u (removal)", fontsize=8, color=INK2, va="center")
ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.5, 1.5)
ax.set_title(f"(a) V-notch, flank angle β = {beta:.0f}°\nsampled normals n̂ point into B", fontsize=9, fontweight="normal")
clean(ax, equal=True, hide_axes=True)
# (b) normal cone and its dual (translations only) on the unit circle
ax = axs[1]
th = np.linspace(0, 2 * np.pi, 361)
ax.plot(np.cos(th), np.sin(th), color=GRID, lw=1.2)
angs = np.arange(0, 360, 1)
dirs = np.stack([np.cos(np.radians(angs)), np.sin(np.radians(angs))], 1)
in_motion = (dirs @ N2.T >= -1e-12).all(axis=1)
nang = np.degrees(np.arctan2(N2[:, 1], N2[:, 0]))
in_normal = (angs >= nang.min() - 0.5) & (angs <= nang.max() + 0.5)
for a, ok in zip(angs, in_normal):
    if ok:
        ax.add_patch(Wedge((0, 0), 1.0, a - 0.5, a + 0.5, width=0.13, fc=ORANGE, ec="none", zorder=1))
for a, ok in zip(angs, in_motion):
    if ok:
        ax.add_patch(Wedge((0, 0), 0.78, a - 0.5, a + 0.5, fc=BLUE, ec="none", alpha=0.3, zorder=2))
for n in (N2[0], N2[-1]):
    ax.annotate("", xy=0.86 * n, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2, mutation_scale=10), zorder=3)
ax.text(0.95, 0.42, "n̂₁", fontsize=9, color=INK2); ax.text(-0.95, 0.42, "n̂₂", fontsize=9, color=INK2, ha="right")
mw = in_motion.sum(); nw = in_normal.sum()
ax.text(0, 0.42, f"motion cone\nV = N*\n{mw - 1}° wide", ha="center", va="center", fontsize=8.5, color=INK)
ax.text(0, 1.18, f"normal cone (ring): {nw - 1}° wide", ha="center", va="center", fontsize=8, color=INK)
ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.1, 1.35)
ax.set_title("(b) V-notch: motion cone V = N*\n(translations only)", fontsize=9, fontweight="normal")
clean(ax, equal=True, hide_axes=True)
print(f"beta={beta}: motion cone width {mw-1} deg (expect {2*beta:.0f}), normal cone width {nw-1} deg (expect {180-2*beta:.0f})")
# (c) same for an n-type spline joint: normals are a curve, the dual is still convex
ax = axs[2]
cs, samp_n = ntype(0.18)
Nn = np.array([n for r, n in samp_n])
in_motion_n = (dirs @ Nn.T >= -1e-12).all(axis=1)
nang_n = np.degrees(np.arctan2(Nn[:, 1], Nn[:, 0]))
in_normal_n = (angs >= nang_n.min() - 0.5) & (angs <= nang_n.max() + 0.5)
ax.plot(np.cos(th), np.sin(th), color=GRID, lw=1.2)
for a, ok in zip(angs, in_normal_n):
    if ok:
        ax.add_patch(Wedge((0, 0), 1.0, a - 0.5, a + 0.5, width=0.13, fc=ORANGE, ec="none", zorder=1))
for a, ok in zip(angs, in_motion_n):
    if ok:
        ax.add_patch(Wedge((0, 0), 0.78, a - 0.5, a + 0.5, fc=BLUE, ec="none", alpha=0.3, zorder=2))
ax.plot(0.935 * Nn[:, 0], 0.935 * Nn[:, 1], "o", ms=2.0, color=INK2, zorder=3)
ax.text(0, 0.42, f"motion cone\nV = N*\n{in_motion_n.sum()-1}° wide", ha="center", va="center", fontsize=8.5, color=INK)
ax.text(0, 1.18, f"50 sampled n̂ (dots): {in_normal_n.sum()-1}° fan", ha="center", va="center", fontsize=8, color=INK)
ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.1, 1.35)
ax.set_title("(c) n-type bump, h = 0.18:\nnormals on a curve, V still convex", fontsize=9, fontweight="normal")
clean(ax, equal=True, hide_axes=True)
fig.suptitle("A joint's motion cone is the dual cone of its contact normals: wider normal fan → narrower escape cone",
             fontsize=10.5, fontweight="semibold", color=INK, y=1.08)
save(fig, "wang2021-mocca-1.png")

# =========================================================================== Figure 2: conic sections with rotation
heights = [0.06, 0.18, 0.40]
cols = [BLUE, ORANGE, AQUA]
fig, axs = plt.subplots(1, 3, figsize=(8, 3.3), gridspec_kw={"width_ratios": [1.15, 1, 1], "wspace": 0.5})
ax = axs[0]
xx = np.linspace(0, 1, 200)
for h, col in zip(heights, cols):
    cs, _ = ntype(h)
    ax.plot(xx, cs(xx), color=col, lw=2, label=f"h = {h:.2f}")
ax.plot([0, 1], [0, 0], color=INK2, lw=1, ls=":")
ax.set_xlabel("x along contact (length 1)"); ax.set_ylabel("height")
ax.set_xlim(0, 1); ax.set_ylim(-0.05, 0.5)
ax.legend(loc="upper right", fontsize=8)
ax.set_title("(a) n-type profiles\n(cubic spline, true aspect)", fontsize=9, fontweight="normal")
clean(ax)
ax.set_aspect("equal", adjustable="box")
ax = axs[1]
alphas = {}
for h, col in zip(heights, cols):
    _, samp_h = ntype(h, m=50)
    Ng = generalized_normals(samp_h)
    poly = conic_section(Ng)
    alphas[h] = translational_halfangle(Ng)
    ax.add_patch(Polygon(poly, closed=True, fc=col, ec=col, alpha=0.25, lw=1.5, zorder=2))
    ax.plot(*np.vstack([poly, poly[:1]]).T, color=col, lw=1.8, zorder=3)
    print(f"h={h}: insertion half-angle alpha = {alphas[h]:.2f} deg, section vertices = {len(poly)}")
ax.axhline(0, color=INK2, lw=0.8, ls=":")
ax.set_xlabel("v_x  (with v_y = 1)"); ax.set_ylabel("ω (rad per unit v_y)")
ax.set_xlim(-2.2, 2.2); ax.set_ylim(-1.6, 2.6)
ax.set_title("(b) Cut of V at v_y = 1 (50 normals):\nα = " + ", ".join(f"{alphas[h]:.0f}°" for h in heights) + " (open above)", fontsize=9, fontweight="normal")
clean(ax)
ax = axs[2]
h = 0.18
for m, col, lab in [(4, ORANGE, "4 samples (over-estimate)"), (50, BLUE, "50 samples")]:
    _, samp_h = ntype(h, m=m)
    Ng = generalized_normals(samp_h)
    poly = conic_section(Ng)
    ax.add_patch(Polygon(poly, closed=True, fc=col, ec=col, alpha=0.25, lw=1.5, zorder=2))
    ax.plot(*np.vstack([poly, poly[:1]]).T, color=col, lw=1.8, zorder=3, label=lab)
    print(f"h={h}, m={m}: alpha = {translational_halfangle(Ng):.2f} deg")
ax.axhline(0, color=INK2, lw=0.8, ls=":")
ax.set_xlabel("v_x  (with v_y = 1)")
ax.set_xlim(-2.2, 2.2); ax.set_ylim(-1.6, 2.6)
ax.legend(loc="upper left", fontsize=7, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.95)
ax.set_title("(c) Fewer samples → bigger cone\n(Theorem 3.2), h = 0.18", fontsize=9, fontweight="normal")
clean(ax)
fig.suptitle("Taller bumps shrink the motion cone (translation and rotation); sampling can only enlarge it",
             fontsize=10.5, fontweight="semibold", color=INK, y=1.06)
save(fig, "wang2021-mocca-2.png")

# =========================================================================== Figure 3: the trade-off
betas = np.linspace(2, 88, 44)
alpha_ins, tilt_tol = [], []
for b in betas:
    B, samp = vnotch(b)
    Ng = generalized_normals(samp)
    alpha_ins.append(translational_halfangle(Ng))
    tilt_tol.append(tilt_tolerance(samp, B))
alpha_ins = np.array(alpha_ins); tilt_tol = np.array(tilt_tol)
print("beta=30: alpha =", np.interp(30, betas, alpha_ins), "tilt tol =", np.interp(30, betas, tilt_tol))
# infeasibility measure E as a function of tilt for beta = 30
B30, samp30 = vnotch(30.0)
c30, a30 = centroid_area(B30); A30 = wrench_matrix(samp30, c30)
phis = np.linspace(0, 90, 181)
E = [infeasibility_E(A30, [a30 * np.sin(np.radians(p)), -a30 * np.cos(np.radians(p)), 0.0]) / a30 ** 2 for p in phis]

fig, axs = plt.subplots(1, 2, figsize=(8, 3.3), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axs[0]
a_req, t_req = 5.0, 30.0
lo_b = betas[np.argmax(alpha_ins >= a_req)]
hi_b = betas[np.where(tilt_tol >= t_req)[0].max()]
ax.axvspan(lo_b, hi_b, color=GRID, zorder=0)
ax.plot(betas, alpha_ins, color=BLUE, lw=2, label="insertion cone half-angle α (from V)")
ax.plot(betas, tilt_tol, color=ORANGE, lw=2, label="frictionless tilt tolerance (E = 0)")
ax.axhline(a_req, color=BLUE, lw=0.9, ls=":"); ax.axhline(t_req, color=ORANGE, lw=0.9, ls=":")
ax.text(20, a_req + 2, "required α ≥ 5°", fontsize=7.5, color=INK2)
ax.text(62, t_req + 2, "required tilt ≥ 30°", fontsize=7.5, color=INK2)
ax.text(0.5 * (lo_b + hi_b), 84, f"feasible flank\nangles {lo_b:.0f}°–{hi_b:.0f}°", ha="center", va="top", fontsize=7.5, color=INK2)
ax.set_xlabel("flank angle β from the removal direction (°)")
ax.set_ylabel("angle (°)")
ax.set_xlim(0, 90); ax.set_ylim(0, 92)
ax.set_xticks(range(0, 91, 15)); ax.set_yticks(range(0, 91, 15))
ax.legend(loc="lower center", bbox_to_anchor=(0.42, 0.1), fontsize=7.5)
ax.set_title("(a) V-notch: α + tilt tolerance = 90°", fontsize=9, fontweight="normal")
clean(ax)
ax = axs[1]
ax.plot(phis, E, color=BLUE, lw=2)
ax.axvline(60, color=INK2, lw=0.9, ls=":")
ax.text(58, max(E) * 0.95, "φ = 90° − β = 60°", ha="right", va="top", fontsize=7.5, color=INK2)
ax.set_xlabel("gravity tilt φ (°), β = 30°")
ax.set_ylabel("infeasibility E (Eq. 7) / weight²")
ax.set_xlim(0, 90)
ax.set_title("(b) E stays 0 until the load leaves the\nnormal cone, then grows", fontsize=9, fontweight="normal")
clean(ax)
fig.suptitle("Assemblability and stability trade one-for-one; MOCCA's sandwich α ≤ β ≤ γ is the grey band",
             fontsize=10.5, fontweight="semibold", color=INK, y=1.06)
save(fig, "wang2021-mocca-3.png")

# =========================================================================== Figure 4: Table 1
# (result, dim, parts, joints, alpha, feature, time in min) from Table 1 of the paper
table1 = [
    ("M", "2D", 10, 9, 5, "Equilibrium", 1.00), ("Horse", "3D", 13, 9, 5, "Equilibrium", 9.82),
    ("Pavilion", "3D", 48, 62, 5, "Equilibrium", 2.19), ("Lilium Tower", "3D", 139, 325, 5, "Equilibrium", 203.78),
    ("Scarecrow", "2D", 4, 5, 25, "Equilibrium", 0.08), ("Sphere", "3D", 6, 12, 3, "Interlocking", 2.98),
    ("Stack", "2D", 10, 10, None, "Equilibrium", 0.04), ("Tree", "2D", 7, 9, 10, "Equilibrium", 0.37),
    ("Leaning Tower (left)", "2D", 14, 28, 5, "Equilibrium", 0.23), ("Leaning Tower (mid)", "2D", 14, 28, 5, "Equilibrium", 0.45),
    ("Leaning Tower (right)", "2D", 13, 27, 5, "Equilibrium", 0.38), ("Leaning Tower", "2D", 14, 28, 5, "Support-free", 0.68),
    ("Deer", "2D", 14, 23, 5, "Support-free", 0.70), ("Igloo", "3D", 139, 326, 5, "Laterally stable", 61.11),
]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for dim, col, mk in [("2D", BLUE, "o"), ("3D", ORANGE, "s")]:
    rows = [r for r in table1 if r[1] == dim]
    ax.plot([r[3] for r in rows], [r[6] for r in rows], mk, ms=6.5, color=col, label=dim, zorder=3, ls="none")
for r in table1:
    if r[3] >= 60 or r[0] in ("Horse", "Sphere", "M", "Scarecrow"):
        ax.text(r[3] * 1.12, r[6], r[0], fontsize=7.5, color=INK2, va="center")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(4, 700); ax.set_ylim(0.03, 400)
ax.set_xlabel("number of cone joints (log scale)")
ax.set_ylabel("optimization time (min, log scale)")
ax.legend(loc="upper left", fontsize=8.5, title="joint dimension", title_fontsize=8.5)
ax.set_title("Table 1: 2D assemblies optimize in under a minute, 3D ones take up to hours", fontsize=10.5)
clean(ax)
save(fig, "wang2021-mocca-4.png")
