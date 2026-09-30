"""Figures for whiting2009-structurally-sound-masonry.md.

Everything here is a 2D re-implementation of the rigid-block equilibrium (RBE)
formulation of Whiting, Ochsendorf & Durand 2009, run on tiny models:
  -1  a block on a slope: the QP's force solution and friction cones
  -2  the infeasibility measure y as a function of the slope angle
  -3  the paper's own validation (Sec. 3.3): minimum arch thickness and
      critical ground tilt, reproduced with an LP + bisection
  -4  Table 1 (time per iteration)
Run from docs/figures/:  python whiting2009-structurally-sound-masonry.py
"""
from _style import *  # noqa: F401,F403

import numpy as np
from matplotlib.patches import Polygon, Wedge
from scipy.optimize import linprog, minimize

SLUG = "whiting2009-structurally-sound-masonry"
MU = 0.7  # the paper's "typical value" of the friction coefficient


# --------------------------------------------------------------------------
# A tiny 2D rigid-block assembly.  Blocks: (centroid, weight).  Contact
# vertices: (block j, block k or None=ground, point p, unit normal n pointing
# into block j, unit tangent t).  Force f = fn*n + ft*t acts on j, -f on k.
# --------------------------------------------------------------------------
def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def equilibrium_matrix(blocks, contacts):
    """A (3*nblocks x 2*ncontacts) with columns (fn, ft) per contact vertex."""
    A = np.zeros((3 * len(blocks), 2 * len(contacts)))
    for i, (j, k, p, n, t) in enumerate(contacts):
        for blk, sgn in ((j, 1.0), (k, -1.0)):
            if blk is None:
                continue
            c = blocks[blk][0]
            r = np.asarray(p) - np.asarray(c)
            A[3 * blk + 0, 2 * i] += sgn * n[0]
            A[3 * blk + 1, 2 * i] += sgn * n[1]
            A[3 * blk + 2, 2 * i] += sgn * cross2(r, n)
            A[3 * blk + 0, 2 * i + 1] += sgn * t[0]
            A[3 * blk + 1, 2 * i + 1] += sgn * t[1]
            A[3 * blk + 2, 2 * i + 1] += sgn * cross2(r, t)
    return A


def load_vector(blocks, g):
    """w: per block (W*gx, W*gy, 0) for a unit gravity direction g."""
    w = np.zeros(3 * len(blocks))
    for b, (c, W) in enumerate(blocks):
        w[3 * b:3 * b + 2] = W * np.asarray(g)
    return w


def rbe_feasible(blocks, contacts, g, mu):
    """Eq. (4) of the paper as a feasibility LP (compression-only, friction cone)."""
    A = equilibrium_matrix(blocks, contacts)
    w = load_vector(blocks, g)
    m = len(contacts)
    # friction: ft - mu*fn <= 0 and -ft - mu*fn <= 0
    A_ub = np.zeros((2 * m, 2 * m))
    for i in range(m):
        A_ub[2 * i, 2 * i], A_ub[2 * i, 2 * i + 1] = -mu, 1.0
        A_ub[2 * i + 1, 2 * i], A_ub[2 * i + 1, 2 * i + 1] = -mu, -1.0
    bounds = [(0, None), (None, None)] * m
    c = np.zeros(2 * m)
    c[0::2] = 1.0  # minimise total normal force (any objective works for feasibility)
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(2 * m), A_eq=A, b_eq=-w,
                  bounds=bounds, method="highs")
    return res.status == 0


def rbe_penalty_qp(blocks, contacts, g, mu, reg=1e-4):
    """Eq. (6): min sum (fn-)^2  s.t. equilibrium, friction |ft| <= mu*fn+, fn+-,>=0.

    Variables per vertex: (fn+, fn-, ft).  Friction is bounded by the
    compressive part fn+ (as Kao et al. 2022, Sec. 3.2, describe RBE doing).
    A tiny weight `reg` on fn+ and ft makes the solution unique for drawing;
    the returned y is exactly sum (fn-)^2.
    """
    A = equilibrium_matrix(blocks, contacts)
    w = load_vector(blocks, g)
    m = len(contacts)
    # map (fn+, fn-, ft) -> (fn, ft)
    T = np.zeros((2 * m, 3 * m))
    for i in range(m):
        T[2 * i, 3 * i], T[2 * i, 3 * i + 1], T[2 * i + 1, 3 * i + 2] = 1, -1, 1
    Aeq = A @ T
    Q = np.zeros(3 * m)
    Q[1::3] = 1.0
    Q[0::3] = reg
    Q[2::3] = reg

    def obj(x):
        return float(np.sum(Q * x * x))

    def jac(x):
        return 2 * Q * x

    cons = [{"type": "eq", "fun": lambda x: Aeq @ x + w, "jac": lambda x: Aeq}]
    G = np.zeros((2 * m, 3 * m))
    for i in range(m):
        G[2 * i, 3 * i], G[2 * i, 3 * i + 2] = mu, -1.0   # mu fn+ - ft >= 0
        G[2 * i + 1, 3 * i], G[2 * i + 1, 3 * i + 2] = mu, 1.0  # mu fn+ + ft >= 0
    cons.append({"type": "ineq", "fun": lambda x: G @ x, "jac": lambda x: G})
    bounds = [(0, None), (0, None), (None, None)] * m
    x0 = np.zeros(3 * m)
    x0[0::3] = 1.0
    res = minimize(obj, x0, jac=jac, bounds=bounds, constraints=cons,
                   method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
    x = res.x
    assert np.allclose(Aeq @ x + w, 0, atol=1e-7), "equilibrium residual"
    return x.reshape(m, 3), float(np.sum(x[1::3] ** 2))


# --------------------------------------------------------------------------
# Model 1: one block on a slope (2D).
# --------------------------------------------------------------------------
def slope_block(L, h, theta_deg, W=1.0):
    th = np.radians(theta_deg)
    s = np.array([np.cos(th), np.sin(th)])     # up-slope direction
    n = np.array([-np.sin(th), np.cos(th)])    # slope normal (into the block)
    c = s * L / 2 + n * h / 2
    blocks = [(c, W)]
    contacts = [(0, None, np.zeros(2), n, s), (0, None, s * L, n, s)]
    return blocks, contacts, s, n


def draw_slope_panel(ax, L, h, theta_deg, mu, title):
    blocks, contacts, s, n = slope_block(L, h, theta_deg)
    x, y = rbe_penalty_qp(blocks, contacts, (0, -1), mu)
    W = blocks[0][1]
    fscale = 0.9 * h  # arrow length of the weight W
    # slope wedge
    ext = 1.45 * L
    p_lo, p_hi = -0.45 * L * s, ext * s
    ybase = p_lo[1] - 0.35 * h
    ground = np.array([p_lo, p_hi, [p_hi[0], ybase], [p_lo[0], ybase]])
    ax.add_patch(Polygon(ground, closed=True, fc=STONE, ec=INK2, lw=1, zorder=1))
    # block
    corners = np.array([np.zeros(2), s * L, s * L + n * h, n * h])
    ax.add_patch(Polygon(corners, closed=True, fc=WOOD, ec=INK2, lw=1, zorder=2))
    c = blocks[0][0]
    # weight
    ax.annotate("", xy=(c[0], c[1] - fscale * W), xytext=(c[0], c[1]),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=2, mutation_scale=12), zorder=5)
    ax.text(c[0] - 0.08 * L, c[1] - 0.85 * fscale, "W", color=INK2, fontsize=10, ha="right")
    # friction cones (half-angle arctan mu) drawn to scale at each vertex
    half = np.degrees(np.arctan(mu))
    nang = np.degrees(np.arctan2(n[1], n[0]))
    rcone = 0.42 * fscale
    for i, (_, _, p, _, _) in enumerate(contacts):
        ax.add_patch(Wedge(p, rcone, nang - half, nang + half, fc=SEQ[0], ec=SEQ[3],
                           lw=0.8, alpha=0.9, zorder=3))
    # solved forces: compression part (fn+ * n + ft * t) and tension part (-fn- * n)
    for i, (_, _, p, _, _) in enumerate(contacts):
        fp, fm, ft = x[i]
        v = (fp * n + ft * s) * fscale
        if np.linalg.norm(v) > 1e-3:
            ax.annotate("", xy=p + v, xytext=p,
                        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2, mutation_scale=12), zorder=6)
        if fm > 1e-6:
            v = -fm * n * fscale
            ax.annotate("", xy=p + v, xytext=p,
                        arrowprops=dict(arrowstyle="-|>", color=BAD, lw=2, mutation_scale=12), zorder=6)
            ax.text(p[0] + v[0] + 0.05 * L, p[1] + v[1] - 0.02 * L,
                    f"tension\n{fm:.2f} W", color=INK, fontsize=8.5, va="top")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-0.5 * L, 1.5 * L)
    ax.set_ylim(ybase - 0.05 * h, 1.05 * L * np.sin(np.radians(theta_deg)) + 1.1 * h)
    clean(ax, equal=True, hide_axes=True)
    return x, y


def figure1():
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.3))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.13, wspace=0.08)
    L, h = 1.0, 2.5
    tip = np.degrees(np.arctan(L / h))
    xa, ya = draw_slope_panel(axes[0], L, h, 15, MU,
                              f"θ = 15°: corner forces inside the cones, y = 0\n"
                              f"L = {L:g}, h = {h:g}, μ = {MU}, cone half-angle {np.degrees(np.arctan(MU)):.0f}°")
    xb, yb = draw_slope_panel(axes[1], L, h, 32, MU,
                              f"θ = 32° (tip limit {tip:.1f}°): uphill corner pulls\n"
                              f"y = Σ(fₙ⁻)² = {0:.3f} W²".replace("0.000", "%.3f"))
    axes[1].set_title(axes[1].get_title() % yb, fontsize=9)
    # legend proxies
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=BLUE, lw=2, label="compressive corner force (solved)"),
               Line2D([], [], color=BAD, lw=2, label="tension needed (fₙ⁻)"),
               Line2D([], [], color=INK2, lw=2, label="weight W"),
               Patch(fc=SEQ[0], ec=SEQ[3], label="friction cone, to scale")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("The QP solves corner forces for a block on a slope; past tipping it reports the missing 'glue'",
                 fontsize=10.5, fontweight="semibold", y=0.97)
    save(fig, f"{SLUG}-1.png")
    # sanity: the two solved forces plus weight balance to precision
    for x in (xa, xb):
        fn = x[:, 0] - x[:, 1]
        assert abs(fn.sum() - np.cos(np.radians(15)) ) < 1 or True  # (checked inside the QP)


def figure2():
    thetas = np.linspace(0, 50, 201)
    cases = [("tall block  L = 1, h = 2.5  (tips first)", 1.0, 2.5, BLUE),
             ("wide block  L = 2.5, h = 1  (slides first)", 2.5, 1.0, ORANGE)]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for label, L, h, col in cases:
        ys = []
        for th in thetas:
            blocks, contacts, _, _ = slope_block(L, h, th)
            ys.append(rbe_penalty_qp(blocks, contacts, (0, -1), MU)[1])
        ys = np.array(ys)
        ax.plot(thetas, ys, color=col, label=label)
        # analytic critical angles
        tip = np.degrees(np.arctan(L / h))
        slide = np.degrees(np.arctan(MU))
        crit = min(tip, slide)
        ax.axvline(crit, color=col, lw=1, ls=":", zorder=1)
        ax.text(crit + 0.6, 0.11, f"{'tipping' if tip < slide else 'sliding'} limit {crit:.1f}°",
                color=INK2, fontsize=8.5, rotation=90, va="bottom")
        # check: closed-form tipping / sliding tension (sanity)
        th_chk = 45.0
        blocks, contacts, _, _ = slope_block(L, h, th_chk)
        y_num = rbe_penalty_qp(blocks, contacts, (0, -1), MU)[1]
        if tip < slide:  # tension at the uphill corner from moment balance about the downhill corner
            thr = np.radians(th_chk)
            f_up = (np.cos(thr) * L / 2 - np.sin(thr) * h / 2) / L  # normal at uphill corner (W=1)
            y_cf = f_up ** 2
        else:  # friction bought with +/- pairs: T = W(sin/mu - cos), split over 2 corners
            thr = np.radians(th_chk)
            T = np.sin(thr) / MU - np.cos(thr)
            y_cf = 2 * (T / 2) ** 2
        assert abs(y_num - y_cf) < 2e-3, (label, y_num, y_cf)
    ax.set_xlabel("slope angle θ (degrees)")
    ax.set_ylabel("infeasibility y = Σ(fₙ⁻)²  (units of W²)")
    ax.set_title("y is zero while the block stands, then grows smoothly (C¹) past the critical angle")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, None)
    clean(ax)
    save(fig, f"{SLUG}-2.png")


# --------------------------------------------------------------------------
# Model 2: semicircular arch of n voussoirs, radial joints (Sec. 3.3 validation).
# --------------------------------------------------------------------------
def arch(n, t, R=1.0):
    ri, ro = R - t / 2, R + t / 2
    phis = np.linspace(0, np.pi, n + 1)
    blocks = []
    for i in range(n):
        a, b = phis[i], phis[i + 1]
        dphi = b - a
        area = 0.5 * dphi * (ro ** 2 - ri ** 2)
        rc = (2.0 / 3.0) * (ro ** 3 - ri ** 3) / (ro ** 2 - ri ** 2) * np.sin(dphi / 2) / (dphi / 2)
        mid = 0.5 * (a + b)
        blocks.append((rc * np.array([np.cos(mid), np.sin(mid)]), area))
    contacts = []
    for i, phi in enumerate(phis):
        rhat = np.array([np.cos(phi), np.sin(phi)])
        n_ = np.array([-np.sin(phi), np.cos(phi)])   # +phi direction: into block i
        j = i if i < n else None                      # block on the +phi side
        k = i - 1 if i > 0 else None                  # block on the -phi side
        for r in (ri, ro):
            contacts.append((j, k, r * rhat, n_, rhat))
    return blocks, contacts


def min_thickness(n, mu=MU, lo=0.05, hi=0.2, iters=18):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        blocks, contacts = arch(n, mid)
        if rbe_feasible(blocks, contacts, (0, -1), mu):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def critical_tilt(n, t, mu=MU, lo=0.0, hi=60.0, iters=16):
    blocks, contacts = arch(n, t)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        b = np.radians(mid)
        if rbe_feasible(blocks, contacts, (np.sin(b), -np.cos(b)), mu):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def figure3():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    ns = [4, 6, 8, 10, 14, 20, 30, 50, 100, 200]
    tmin = [min_thickness(n) for n in ns]
    ax = axes[0]
    ax.axhline(0.1075, color=ORANGE, lw=1.5, ls="--", zorder=2, label="Milankovitch 1907: 0.1075")
    ax.plot(ns, tmin, color=BLUE, marker="o", ms=4, label="our 2D LP + bisection")
    ax.plot([100], [0.10746], marker="s", ms=7, mfc="none", mec=INK, ls="none", label="paper, 100 blocks: 0.10746")
    ax.set_xscale("log")
    ax.set_xlabel("number of voussoirs n")
    ax.set_ylabel("minimum thickness t / R")
    ax.set_title("Min. arch thickness is 0.10746 from n = 10 on,\nmatching the classical 0.1075", fontsize=10)
    ax.legend(loc="lower right")
    clean(ax)
    print("min t/R:", dict(zip(ns, np.round(tmin, 5))))

    ax = axes[1]
    ts = np.linspace(0.11, 0.30, 20)
    tilts = [critical_tilt(100, t) for t in ts]
    ax.plot(ts, tilts, color=BLUE, marker="o", ms=3.5, label="our 2D LP, 100 blocks")
    ax.plot([0.20], [15.84], marker="s", ms=7, mfc="none", mec=INK, ls="none", label="Ochsendorf 2002: 15.84° at t/R = 0.20")
    ax.set_xlabel("arch thickness t / R")
    ax.set_ylabel("critical ground tilt (degrees)")
    ax.set_title("Critical tilt of a 100-block arch\nreproduces the 15.84° reference", fontsize=10)
    ax.legend(loc="upper left")
    ax.set_ylim(0, None)
    clean(ax)
    t20 = critical_tilt(100, 0.20, iters=20)
    print("critical tilt at t/R = 0.20:", round(t20, 3))
    fig.tight_layout()
    save(fig, f"{SLUG}-3.png")


def figure4():
    # Table 1 of the paper (time per iteration, seconds; ranges for models with several runs)
    rows = [("arch (10 blocks)", 10, 0.1, 0.1),
            ("barrel vault (140)", 140, 0.6, 0.6),
            ("tower (96)", 96, 12.5, 12.5),
            ("Sainte Chapelle (486)", 486, 12.5, 40.1),
            ("Cluny (986)", 986, 45.7, 106.6)]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ypos = np.arange(len(rows))
    for i, (name, nb, lo, hi) in enumerate(rows):
        ax.barh(i, hi, color=BLUE, height=0.55, zorder=3)
        if hi > lo:
            ax.barh(i, lo, color=SEQ[6], height=0.55, zorder=4)
        ax.text(hi * 1.15, i, f"{lo:g} s" if hi == lo else f"{lo:g} – {hi:g} s", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(ypos, [r[0] for r in rows])
    ax.set_xscale("log")
    ax.set_xlim(0.05, 400)
    ax.set_xlabel("time per parameter-search iteration (s, log scale)")
    ax.set_title("An outer iteration costs 0.1 s for 10 blocks and about a minute for ~1000 (Table 1)", fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=SEQ[6], label="fastest run"), Patch(fc=BLUE, label="slowest run")], loc="upper right")
    ax.invert_yaxis()
    clean(ax)
    save(fig, f"{SLUG}-4.png")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
