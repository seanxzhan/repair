"""Figures for docs/yao2017-decorative-joinery.md.

Everything here is computed with a small 2D re-implementation of the paper's
variational static analysis (Sec. 5.3): Gauss's least constraint (Eq. 13) and
maximal dissipation (Eq. 14), solved by alternating the two QPs (Staggered
Projections).  One free rigid body, fixed environment, point contacts.
The equilibrium method (EM, Sec. 5.2) is a plain LP feasibility problem.
Run from docs/figures/:  python yao2017-decorative-joinery.py
"""
from _style import *  # noqa: F401,F403

import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from scipy.optimize import linprog, minimize

np.random.seed(0)
G = 9.81
MU = np.tan(np.radians(43.0))  # friction angle 43 deg, as in the paper's benchmarks (Fig. 8)


# ----------------------------------------------------------------------------
# 2D rigid body + point contacts
# ----------------------------------------------------------------------------
class Body:
    def __init__(self, corners, mass=1.0):
        self.c = np.mean(corners, axis=0)            # centre of mass (uniform density)
        w = np.linalg.norm(corners[1] - corners[0])   # side lengths (corners in order)
        h = np.linalg.norm(corners[2] - corners[1])
        self.m = mass
        self.I = mass * (w * w + h * h) / 12.0       # rectangle
        self.corners = corners
        self.Minv = np.diag([1 / mass, 1 / mass, 1 / self.I])


def gamma(body, p):
    """Map generalized acceleration (x, y, theta) to the acceleration of point p."""
    r = p - body.c
    return np.array([[1.0, 0.0, -r[1]], [0.0, 1.0, r[0]]])


def build(body, contacts, extra_force=(0.0, 0.0)):
    """contacts: list of (point, unit normal into the body).  Returns N, D, g."""
    N, D = [], []
    for p, n in contacts:
        n = np.asarray(n, float)
        t = np.array([-n[1], n[0]])
        Gk = gamma(body, p)
        N.append(Gk.T @ n)
        D.append(Gk.T @ t)
    N = np.array(N).T
    D = np.array(D).T
    g = np.array([extra_force[0], -body.m * G + extra_force[1], 0.0])
    return N, D, g


def em_feasible(body, contacts):
    """Equilibrium method (Eqs. 6-8): does ANY admissible force set balance the load?
    LP: minimise the largest normal force (gives the uniform solution of Sec. 5.2)."""
    N, D, g = build(body, contacts)
    K = N.shape[1]
    # variables: alpha (K), beta (K), s
    c = np.zeros(2 * K + 1)
    c[-1] = 1.0
    A_eq = np.hstack([N, D, np.zeros((3, 1))])
    b_eq = -g
    rows, rhs = [], []
    for k in range(K):
        r = np.zeros(2 * K + 1); r[K + k] = 1; r[k] = -MU; rows.append(r); rhs.append(0)   # beta <= mu alpha
        r = np.zeros(2 * K + 1); r[K + k] = -1; r[k] = -MU; rows.append(r); rhs.append(0)  # -beta <= mu alpha
        r = np.zeros(2 * K + 1); r[k] = 1; r[-1] = -1; rows.append(r); rhs.append(0)       # alpha <= s
    bounds = [(0, None)] * K + [(None, None)] * K + [(0, None)]
    res = linprog(c, A_ub=np.array(rows), b_ub=rhs, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        return False, None, None
    return True, res.x[:K], res.x[K:2 * K]


def variational(body, contacts, extra_force=(0.0, 0.0), tol=1e-9, iters=200):
    """Staggered projections between Eq. (13) and Eq. (14).  Returns alpha, beta, r*."""
    N, D, g = build(body, contacts, extra_force)
    Minv = body.Minv
    K = N.shape[1]
    A = N.T @ Minv @ N
    B = D.T @ Minv @ D
    # A is rank <= 3 for K > 3 contacts: an equal-and-opposite "squeeze" pair is invisible to the
    # kinetic norm.  A tiny Tikhonov term picks the minimum-norm alpha (least squeeze), which is
    # what "minimal constraint forces" means physically; it does not change any non-degenerate answer.
    Areg = A + 1e-6 * np.trace(A) / K * np.eye(K)
    beta = np.zeros(K)
    alpha = np.zeros(K)
    for _ in range(iters):
        # --- Gauss's least constraint (Eq. 13): min 1/2 a'Aa  s.t. N'M^-1(Na + Db + g) >= 0
        h = N.T @ Minv @ (D @ beta + g)
        res = minimize(lambda a: 0.5 * a @ Areg @ a, np.zeros(K), jac=lambda a: Areg @ a,
                       constraints=[{"type": "ineq", "fun": lambda a: A @ a + h, "jac": lambda a: A}],
                       bounds=[(0, None)] * K, method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
        alpha_new = np.clip(res.x, 0, None)
        # --- Maximal dissipation (Eq. 14) in the projection form of Staggered Projections
        #     (Kaufman et al. 2008): min 1/2 b'Bb + b'D'M^-1(Na + g)  s.t. |b_k| <= mu a_k.
        #     (Eq. 14 as printed has no 1/2 on the quadratic term; with it, a body inside the
        #     friction cone would keep half its tangential acceleration.)
        c = D.T @ Minv @ (N @ alpha_new + g)
        lim = MU * alpha_new
        res = minimize(lambda b: 0.5 * b @ B @ b + b @ c, np.clip(beta, -lim, lim), jac=lambda b: B @ b + c,
                       bounds=list(zip(-lim, lim)), method="L-BFGS-B", options={"ftol": 1e-15, "gtol": 1e-12})
        beta_new = res.x
        done = np.linalg.norm(alpha_new - alpha) + np.linalg.norm(beta_new - beta) < tol * (1 + np.linalg.norm(alpha_new))
        alpha, beta = alpha_new, beta_new
        if done:
            break
    r = N @ alpha + D @ beta + g
    return alpha, beta, r, N, D


def classify(body, contacts, r, eps=1e-6):
    """Relative accelerations a_k = Gamma_k M^-1 r* (Eq. 15) and the paper's three states."""
    qdd = body.Minv @ r
    aks = [gamma(body, p) @ qdd for p, _ in contacts]
    rel = max(np.linalg.norm(a) for a in aks) / G
    if rel < eps and np.linalg.norm(qdd[:2]) / G < eps and abs(qdd[2]) < eps:
        state = "fully stable"
    elif rel < eps:
        state = "part-stable"
    else:
        state = "part-unstable"
    return state, aks, qdd


def arrow(ax, p, v, color, lw=1.6, scale=1.0, ms=9, **kw):
    if np.linalg.norm(v) * scale < 2e-3:
        return
    ax.add_patch(FancyArrowPatch(p, p + scale * np.asarray(v), color=color, lw=lw, mutation_scale=ms,
                                 arrowstyle="-|>", shrinkA=0, shrinkB=0, zorder=5, **kw))


# ----------------------------------------------------------------------------
# Figure 1: the paper's didactic example, block between two fixed walls
# ----------------------------------------------------------------------------
def block_between_walls():
    d = 1.0
    corners = np.array([[-d / 2, -d / 2], [d / 2, -d / 2], [d / 2, d / 2], [-d / 2, d / 2]])
    body = Body(corners)
    contacts = [(corners[0], (1, 0)), (corners[3], (1, 0)), (corners[1], (-1, 0)), (corners[2], (-1, 0))]
    W = body.m * G

    ok, a_em, b_em = em_feasible(body, contacts)
    alpha, beta, r, N, D = variational(body, contacts)
    state, aks, qdd = classify(body, contacts, r)
    print("Fig1 EM feasible:", ok, "alpha/W =", np.round(a_em / W, 3), "beta/W =", np.round(b_em / W, 3))
    print("Fig1 variational: alpha =", np.round(alpha, 6), "beta =", np.round(beta, 6),
          "state =", state, "accel/g =", np.round(qdd / G, 4))
    assert ok and state == "part-unstable" and abs(qdd[1] / G + 1) < 1e-6

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    fscale = 0.9  # drawing length per unit weight W
    for ax, title, al, be, acc in [
        (axes[0], "EM: a force balance exists → \"stable\"", a_em / W, b_em / W, None),
        (axes[1], "Variational: no push, no friction → falls", alpha / W, beta / W, qdd),
    ]:
        clean(ax, equal=True, hide_axes=True)
        ax.add_patch(Rectangle((-1.0, -1.1), 0.5, 2.2, facecolor=STONE, edgecolor=INK2, lw=0.8, hatch="///"))
        ax.add_patch(Rectangle((0.5, -1.1), 0.5, 2.2, facecolor=STONE, edgecolor=INK2, lw=0.8, hatch="///"))
        ax.add_patch(Polygon(corners, closed=True, facecolor=WOOD, edgecolor=INK2, lw=1.0))
        for xw in (-0.75, 0.75):
            ax.text(xw, 0.85, "fixed wall", fontsize=7.5, color=INK2, ha="center", va="center", rotation=90,
                    bbox=dict(facecolor=STONE, edgecolor="none", pad=1.5))
        # gravity
        arrow(ax, body.c, (0, -1.0), INK, scale=fscale * 0.5)
        ax.text(0.06, -0.25, "W", fontsize=9, color=INK)
        for (p, n), a_k, b_k in zip(contacts, al, be):
            n = np.asarray(n, float)
            t = np.array([-n[1], n[0]])
            arrow(ax, p - fscale * a_k * n, fscale * a_k * n, BLUE, scale=1.0)          # normal force
            arrow(ax, p, fscale * b_k * t, ORANGE, scale=1.0)                            # friction
        if acc is not None:
            arrow(ax, body.c + np.array([0.28, 0.0]), (0, acc[1] / G), BLUE, scale=0.6, lw=2.4, ms=13)
            ax.text(0.0, -0.62, "a = %.2f g, sliding down" % (-acc[1] / G), fontsize=8.5, color=INK, ha="center", va="top",
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
            ax.text(-0.98, -1.35, "α = β = 0 at all four corners → part-unstable", fontsize=8.5, color=INK2)
        else:
            ax.text(-0.98, -1.35, "α = %.2f W (each), β = %.2f W (each), Σβ = W" % (al[0], be[0]),
                    fontsize=8.5, color=INK2)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.45, 1.15)
        ax.set_title(title, fontsize=9.5, pad=4)
    # legend proxies
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BLUE, lw=2, label="normal force α (EM) / acceleration (variational)"),
               Line2D([], [], color=ORANGE, lw=2, label="friction force β"),
               Line2D([], [], color=INK, lw=2, label="weight W")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.suptitle("Same geometry, μ = tan 43°: EM finds a force balance, the variational solver does not",
                 fontsize=10.5, fontweight="semibold", y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    save(fig, "yao2017-decorative-joinery-1.png")


# ----------------------------------------------------------------------------
# Figure 2: snug tenon in a slot, sweep the slot angle relative to gravity
# ----------------------------------------------------------------------------
def slot_sweep():
    d = 1.0
    phis = np.linspace(0, 90, 91)
    acc, n_low, n_up, em_ok = [], [], [], []
    for phi in phis:
        # slot axis makes angle phi with the vertical; rotate contact normals with it
        th = np.radians(phi)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        corners0 = np.array([[-d / 2, -d / 2], [d / 2, -d / 2], [d / 2, d / 2], [-d / 2, d / 2]])
        corners = corners0 @ R.T
        body = Body(corners)
        nl, nr = R @ np.array([1.0, 0.0]), R @ np.array([-1.0, 0.0])
        contacts = [(corners[0], nl), (corners[3], nl), (corners[1], nr), (corners[2], nr)]
        alpha, beta, r, N, D = variational(body, contacts)
        state, aks, qdd = classify(body, contacts, r)
        acc.append(np.linalg.norm(qdd[:2]) / G)
        W = body.m * G
        n_low.append((alpha[0] + alpha[1]) / W)  # wall that gravity leans on (left wall, normal rotated)
        n_up.append((alpha[2] + alpha[3]) / W)
        em_ok.append(em_feasible(body, contacts)[0])
    acc, n_low = np.array(acc), np.array(n_low)
    phi_c = 90 - 43.0
    analytic = np.clip(np.cos(np.radians(phis)) - MU * np.sin(np.radians(phis)), 0, None)
    print("Fig2 max |solver - analytic| accel:", np.abs(acc - analytic).max(), " EM stable everywhere:", all(em_ok))
    print("Fig2 first stable angle (solver):", phis[np.argmax(acc < 1e-6)], "analytic:", phi_c)
    assert np.abs(acc - analytic).max() < 1e-4 and all(em_ok)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    clean(ax)
    ax.plot(phis, analytic, color=INK2, lw=1.2, ls="--", label="analytic  g(cos φ − μ sin φ)")
    ax.plot(phis[::3], acc[::3], "o", color=BLUE, ms=4.5, mfc="none", mew=1.4, label="variational solver (this figure)")
    ax.axvline(phi_c, color=INK2, lw=0.8, ls=":")
    ax.text(phi_c + 1.5, 0.5, "φ = 90° − 43°\n= %.0f°" % phi_c, fontsize=8.5, color=INK2)
    ax.axhspan(-0.05, 0.0, color=GRID, alpha=0.0)
    ax.text(2, -0.09, "EM: \"stable\" at every angle", fontsize=8.5, color=INK2)
    ax.set_xlabel("slot angle from vertical φ (deg)")
    ax.set_ylabel("sliding acceleration of the tenon (g)")
    ax.set_ylim(-0.15, 1.08)
    ax.set_xlim(0, 90)
    ax.legend(loc="upper right")
    ax.set_title("A snug tenon slides until the slot leans more than 47°", fontsize=10)

    ax = axes[1]
    clean(ax)
    ax.plot(phis, n_low, color=BLUE, label="normal force on the lower wall")
    ax.plot(phis, MU * n_low, color=ORANGE, label="friction limit μ·N")
    ax.plot(phis, np.cos(np.radians(phis)), color=INK2, ls="--", lw=1.2, label="load along the slot W cos φ")
    ax.axvline(phi_c, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("slot angle from vertical φ (deg)")
    ax.set_ylabel("force (multiples of the weight W)")
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title("Normal force only appears when gravity\npresses the tenon into a wall", fontsize=9.5)
    fig.tight_layout()
    save(fig, "yao2017-decorative-joinery-2.png")


# ----------------------------------------------------------------------------
# Figure 3: sliding vs hinging on a tilted ground plane
# ----------------------------------------------------------------------------
def slope_case(w, h, theta_deg, extra=(0, 0)):
    th = np.radians(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    # block sits on a plane through the origin, tilted by theta (uphill toward -x after rotation)
    corners0 = np.array([[-w / 2, 0], [w / 2, 0], [w / 2, h], [-w / 2, h]])
    corners = corners0 @ R.T
    body = Body(corners)
    n = R @ np.array([0.0, 1.0])
    contacts = [(corners[0], n), (corners[1], n)]
    alpha, beta, r, N, D = variational(body, contacts, extra)
    state, aks, qdd = classify(body, contacts, r)
    # sub-classify: hinging if the contacts have different normal accelerations
    kind = state
    if state == "part-unstable":
        an = [a @ n for a in aks]
        kind = "hinging" if max(an) > 1e-6 else "sliding"
    return body, contacts, R, alpha, beta, aks, qdd, kind


def slope_figure():
    tall = (1.0, 2.5)
    squat = (2.5, 1.0)
    thetas = np.arange(0, 61, 1)
    kinds = {}
    for name, (w, h) in [("tall", tall), ("squat", squat)]:
        kinds[name] = [slope_case(w, h, t)[-1] for t in thetas]
    first = {k: {kk: (thetas[v.index(kk)] if kk in v else None) for kk in ("sliding", "hinging")} for k, v in kinds.items()}
    print("Fig3 first angles:", first, " analytic: tall topples at %.1f, squat slides at 43.0" % np.degrees(np.arctan(tall[0] / tall[1])))
    assert first["tall"]["hinging"] == 22 and first["squat"]["sliding"] == 44

    fig = plt.figure(figsize=(7.8, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35])
    ax_t = fig.add_subplot(gs[0]); ax_s = fig.add_subplot(gs[1]); ax_c = fig.add_subplot(gs[2])

    for ax, (w, h), theta, label in [(ax_t, tall, 30.0, "tall block, θ = 30°"), (ax_s, squat, 50.0, "squat block, θ = 50°")]:
        body, contacts, R, alpha, beta, aks, qdd, kind = slope_case(w, h, theta)
        clean(ax, equal=True, hide_axes=True)
        # ground
        gx = np.array([-2.2, 2.2]); ground = np.array([gx, [0, 0]]).T @ R.T
        below = np.vstack([ground, ground[::-1] - 0.35 * R @ np.array([0, 1.0])])
        ax.add_patch(Polygon(below, closed=True, facecolor=STONE, edgecolor=INK2, lw=0.8, hatch="///"))
        ax.add_patch(Polygon(body.corners, closed=True, facecolor=WOOD, edgecolor=INK2, lw=1.0))
        W = body.m * G
        arrow(ax, body.c, (0, -0.6), INK, scale=1.0)
        for (p, n), a_k, al in zip(contacts, aks, alpha):
            arrow(ax, p - al / W * n, al / W * n, ORANGE, scale=1.0)
            arrow(ax, p, a_k / G, BLUE, scale=4.0, lw=2.0, ms=10)
        txt = {"sliding": "sliding: both corners accelerate\nalong the slope (%.2f g)" % (np.linalg.norm(aks[0]) / G),
               "hinging": "hinging: uphill corner lifts,\ndownhill corner stays (θ̈ = %.2f rad/s²)" % qdd[2]}[kind]
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=8, va="top", color=INK)
        ax.set_xlim(-2.3, 2.3); ax.set_ylim(-1.6, 3.0)
        ax.set_title(label, fontsize=9.5)
        print("Fig3", label, kind, "alpha/W =", np.round(alpha / W, 3), "a_k/g =", [np.round(a / G, 3) for a in aks])

    # classification sweep
    clean(ax_c)
    ymap = {"fully stable": 0, "sliding": 1, "hinging": 2}
    for name, col, off in [("tall", BLUE, 0.08), ("squat", ORANGE, -0.08)]:
        ys = [ymap[k] + off for k in kinds[name]]
        ax_c.plot(thetas, ys, ".", color=col, ms=5, label="%s block (w/h = %.1f)" % (name, {"tall": 0.4, "squat": 2.5}[name]))
    ax_c.axvline(np.degrees(np.arctan(0.4)), color=INK2, lw=0.8, ls=":")
    ax_c.axvline(43, color=INK2, lw=0.8, ls=":")
    ax_c.text(np.degrees(np.arctan(0.4)) + 0.8, 0.45, "arctan(w/h)\n= 21.8°", fontsize=7.5, color=INK2)
    ax_c.text(43.8, 0.45, "friction\nangle 43°", fontsize=7.5, color=INK2)
    ax_c.set_yticks([0, 1, 2]); ax_c.set_yticklabels(["fully stable", "sliding", "hinging"])
    ax_c.set_xlabel("ground tilt θ (deg)")
    ax_c.set_ylim(-0.5, 2.5)
    ax_c.legend(loc="center left", bbox_to_anchor=(0.0, 0.64), fontsize=7.5)
    ax_c.set_title("Solver verdict vs tilt (1° steps)", fontsize=9.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BLUE, lw=2, label="relative acceleration a_k (Eq. 15), 4 units per g"),
               Line2D([], [], color=ORANGE, lw=2, label="normal force α_k (units of W)"),
               Line2D([], [], color=INK, lw=2, label="weight W")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.36, -0.02), fontsize=8)
    fig.suptitle("The same solver separates sliding from hinging and returns where each part moves",
                 fontsize=10.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save(fig, "yao2017-decorative-joinery-3.png")


# ----------------------------------------------------------------------------
# Figure 4: benchmark table (paper's Fig. 8): critical tilt angles
# ----------------------------------------------------------------------------
def benchmark_chart():
    cases = ["Feasible arch\n(t/r = 0.15, 36 blocks)", "3-legged π\n(4 blocks)"]
    exp = [4.7, 14.3]; exp_err = [0.2, 0.2]
    var = [5.3, 16.4]
    em = [8.2, 19.1]
    analytic = [8.2, None]
    x = np.arange(len(cases)); wd = 0.24
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    clean(ax)
    ax.bar(x - wd, exp, wd, color=BLUE, label="experiment (±0.2°)", yerr=exp_err, error_kw={"lw": 1, "ecolor": INK2, "capsize": 3})
    ax.bar(x, var, wd, color=ORANGE, label="variational solver")
    ax.bar(x + wd, em, wd, color=AQUA, label="equilibrium method (EM)")
    ax.plot([x[0] - 1.6 * wd, x[0] + 1.6 * wd], [analytic[0]] * 2, ls="--", color=INK, lw=1.2)
    ax.text(x[0] + 1.65 * wd, analytic[0] - 0.7, "analytic (no sliding) 8.2°", fontsize=8, color=INK, va="center")
    for xi, vals in zip(x, zip(exp, var, em)):
        for off, v in zip([-wd, 0, wd], vals):
            ax.text(xi + off, v + 0.35, "%.1f°" % v, ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(cases)
    ax.set_ylabel("critical ground tilt angle (deg)")
    ax.set_ylim(0, 22)
    ax.legend(loc="upper left")
    ax.set_title("The variational solver's critical tilt lands between experiment and EM", fontsize=10)
    fig.tight_layout()
    save(fig, "yao2017-decorative-joinery-4.png")


if __name__ == "__main__":
    block_between_walls()
    slot_sweep()
    slope_figure()
    benchmark_chart()
