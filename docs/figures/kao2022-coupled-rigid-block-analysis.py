"""Figures for kao2022-coupled-rigid-block-analysis.md.

2D re-implementation of RBE (Kao et al. Eq. 7: force-only equilibrium) and of
CRA (Eq. 11: equilibrium + virtual rigid-body motion, contact complementarity
with overlap eps, friction opposed to the virtual slip).  CRA is solved
*exactly* for these tiny models by enumerating every pattern (which contact
vertices are closed, which way each closed vertex slips): each pattern is one
linear program, and the assembly is CRA-feasible iff some pattern is feasible.
This replaces the paper's IPOPT local solve, so there are no local-optimum
issues on these toys.

  -1  model H (block between two walls): the RBE force solution vs the two
      virtual motions CRA can try
  -2  2D keystone wedge tilt test: critical tilt vs mu for RBE and CRA, for a
      sharp and a blunt wedge (2D analogue of Table 2 / model A)
  -3  when CRA *does* certify model H: the rotational "drawer jam", computed
      threshold vs the analytic line mu = 2h/w   (our observation)
  -4  Table 2 of the paper (3-block wedges, 3DEC vs CRA vs RBE)
Run from docs/figures/:  python kao2022-coupled-rigid-block-analysis.py
"""
from _style import *  # noqa: F401,F403

import itertools

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Wedge
from scipy.optimize import linprog, minimize

SLUG = "kao2022-coupled-rigid-block-analysis"
EPS, ETA = 1e-4, 1e-2   # paper: eps 1e-5..1e-4, eta 1e-3..1e-2 of the block size (block size ~ 1 here)


# --------------------------------------------------------------------------
# 2D assembly: blocks = [(centroid, weight)] (all free); contacts =
# (j, k|None, p, n, t): force f = fn*n + ft*t acts on j at p, -f on k;
# n points from k into j.  Kinematics is A^T (virtual-work duality, Eq. 8):
# (dn_i, dt_i) = A^T dq, dn > 0 = opening.
# --------------------------------------------------------------------------
def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def equilibrium_matrix(blocks, contacts):
    A = np.zeros((3 * len(blocks), 2 * len(contacts)))
    for i, (j, k, p, n, t) in enumerate(contacts):
        for blk, sgn in ((j, 1.0), (k, -1.0)):
            if blk is None:
                continue
            r = np.asarray(p) - np.asarray(blocks[blk][0])
            A[3 * blk:3 * blk + 3, 2 * i] += sgn * np.array([n[0], n[1], cross2(r, n)])
            A[3 * blk:3 * blk + 3, 2 * i + 1] += sgn * np.array([t[0], t[1], cross2(r, t)])
    return A


def load_vector(blocks, g):
    w = np.zeros(3 * len(blocks))
    for b, (c, W) in enumerate(blocks):
        w[3 * b:3 * b + 2] = W * np.asarray(g)
    return w


def friction_rows(m, mu):
    G = np.zeros((2 * m, 2 * m))
    for i in range(m):
        G[2 * i, 2 * i], G[2 * i, 2 * i + 1] = -mu, 1.0
        G[2 * i + 1, 2 * i], G[2 * i + 1, 2 * i + 1] = -mu, -1.0
    return G


def rbe_feasible(blocks, contacts, g, mu):
    """Eq. (7) without objective: compression-only, friction-cone equilibrium exists?"""
    A = equilibrium_matrix(blocks, contacts)
    w = load_vector(blocks, g)
    m = len(contacts)
    res = linprog(np.zeros(2 * m), A_ub=friction_rows(m, mu), b_ub=np.zeros(2 * m),
                  A_eq=A, b_eq=-w, bounds=[(0, None), (None, None)] * m, method="highs")
    return res.status == 0


def rbe_forces(blocks, contacts, g, mu):
    """Eq. (7): min ||fn||^2 (plus a tiny ft term for uniqueness) -> (m, 2) forces."""
    A = equilibrium_matrix(blocks, contacts)
    w = load_vector(blocks, g)
    m = len(contacts)
    G = friction_rows(m, mu)
    Q = np.tile([1.0, 1e-3], m)
    cons = [{"type": "eq", "fun": lambda x: A @ x + w, "jac": lambda x: A},
            {"type": "ineq", "fun": lambda x: -G @ x, "jac": lambda x: -G}]
    x0 = np.tile([1.0, 0.0], m)
    res = minimize(lambda x: float(np.sum(Q * x * x)), x0, jac=lambda x: 2 * Q * x,
                   bounds=[(0, None), (None, None)] * m, constraints=cons, method="SLSQP",
                   options={"ftol": 1e-15, "maxiter": 500})
    assert res.success and np.allclose(A @ res.x + w, 0, atol=1e-8)
    return res.x.reshape(m, 2)


def cra_feasible(blocks, contacts, g, mu, eps=EPS, eta=ETA, margin=1e-3, return_solution=False):
    """Eq. (11) decided exactly by pattern enumeration.

    Pattern = (S, sigma): S = closed vertices (dn = -eps, may carry force),
    sigma_i in {-1, 0, +1} = sign of the virtual slip dt_i at a closed vertex;
    friction then has sign -sigma_i (ft = -alpha*dt, alpha >= 0), and is zero
    when sigma_i = 0 (dt_i = 0).  Open vertices carry no force and dn >= -eps.
    Every dn, dt is bounded by eta.  Feasible iff some pattern's LP is feasible.
    """
    A = equilibrium_matrix(blocks, contacts)
    w = load_vector(blocks, g)
    m, nq = len(contacts), 3 * len(blocks)
    K = A.T                       # (2m x nq): row 2i = dn_i, row 2i+1 = dt_i
    nv = 2 * m + nq               # variables: f (2m) then dq (nq)
    G = friction_rows(m, mu)
    for size in range(1, m + 1):
        for S in itertools.combinations(range(m), size):
            for sigma in itertools.product((-1, 0, 1), repeat=size):
                A_eq, b_eq, A_ub, b_ub = [], [], [], []
                bounds = [(0, None), (None, None)] * m + [(None, None)] * nq
                # equilibrium
                for r in range(A.shape[0]):
                    A_eq.append(np.concatenate([A[r], np.zeros(nq)])); b_eq.append(-w[r])
                for i in range(m):
                    rn = np.concatenate([np.zeros(2 * m), K[2 * i]])
                    rt = np.concatenate([np.zeros(2 * m), K[2 * i + 1]])
                    # |dn|, |dt| <= eta
                    A_ub += [rn, -rn, rt, -rt]; b_ub += [eta, eta, eta, eta]
                    if i in S:
                        A_eq.append(rn); b_eq.append(-eps)                   # closed by exactly eps
                        A_ub.append(np.concatenate([G[2 * i], np.zeros(nq)])); b_ub.append(0.0)
                        A_ub.append(np.concatenate([G[2 * i + 1], np.zeros(nq)])); b_ub.append(0.0)
                        sg = sigma[S.index(i)]
                        if sg == 0:
                            A_eq.append(rt); b_eq.append(0.0)
                            bounds[2 * i + 1] = (0.0, 0.0)
                        else:
                            A_ub.append(-sg * rt); b_ub.append(-margin * eta)  # sg*dt >= margin*eta
                            bounds[2 * i + 1] = (None, 0.0) if sg > 0 else (0.0, None)
                    else:
                        A_ub.append(-rn); b_ub.append(eps)                   # dn >= -eps
                        bounds[2 * i] = (0.0, 0.0)
                        bounds[2 * i + 1] = (0.0, 0.0)
                res = linprog(np.zeros(nv), A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                              A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, method="highs")
                if res.status == 0:
                    if return_solution:
                        return True, (S, sigma, res.x[:2 * m].reshape(m, 2), res.x[2 * m:])
                    return True
    return (False, None) if return_solution else False


def critical_tilt(feasible, blocks, contacts, mu, iters=9):
    """Largest gravity rotation (deg) for which `feasible` holds; 180 if it holds upside down."""
    def ok(phi):
        b = np.radians(phi)
        return feasible(blocks, contacts, (np.sin(b), -np.cos(b)), mu)
    if not ok(0.0):
        return 0.0
    if ok(180.0):
        return 180.0
    lo, hi = 0.0, 180.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def model_h(w=1.0, h=0.8, W=1.0):
    """Block of width w, height h between two fixed vertical walls (paper Fig. 3a)."""
    a, b = w / 2, h / 2
    blocks = [((0.0, 0.0), W)]
    contacts = [(0, None, (-a, b), (1.0, 0.0), (0.0, 1.0)),     # left wall, top
                (0, None, (-a, -b), (1.0, 0.0), (0.0, 1.0)),    # left wall, bottom
                (0, None, (a, b), (-1.0, 0.0), (0.0, 1.0)),     # right wall, top
                (0, None, (a, -b), (-1.0, 0.0), (0.0, 1.0))]    # right wall, bottom
    return blocks, contacts


def flat_block(w=1.0, h=1.0, W=1.0):
    blocks = [((0.0, h / 2), W)]
    contacts = [(0, None, (-w / 2, 0.0), (0.0, 1.0), (1.0, 0.0)),
                (0, None, (w / 2, 0.0), (0.0, 1.0), (1.0, 0.0))]
    return blocks, contacts


def keystone(beta_deg, W=1.0):
    """Triangular 2D wedge, apex at the origin, flanks at angle beta from the vertical,
    resting in a matching fixed V-notch (2D analogue of the paper's wedge models)."""
    b = np.radians(beta_deg)
    tipL, tipR = (-np.sin(b), np.cos(b)), (np.sin(b), np.cos(b))
    blocks = [((0.0, 2 * np.cos(b) / 3), W)]
    nL, tL = (np.cos(b), np.sin(b)), (-np.sin(b), np.cos(b))
    nR, tR = (-np.cos(b), np.sin(b)), (np.sin(b), np.cos(b))
    contacts = [(0, None, (0.0, 0.0), nL, tL), (0, None, tipL, nL, tL),
                (0, None, (0.0, 0.0), nR, tR), (0, None, tipR, nR, tR)]
    return blocks, contacts


# --------------------------------------------------------------------------
# Sanity checks: RBE == CRA on a plain planar contact; model H verdicts.
# --------------------------------------------------------------------------
def sanity():
    for mu, expect in ((0.5, np.degrees(np.arctan(0.5))), (2.0, 45.0)):
        blocks, contacts = flat_block()
        r = critical_tilt(rbe_feasible, blocks, contacts, mu, iters=12)
        c = critical_tilt(cra_feasible, blocks, contacts, mu, iters=12)
        assert abs(r - expect) < 0.2 and abs(c - expect) < 0.2, (mu, r, c, expect)
    bl, ct = model_h(1.0, 0.8)
    assert rbe_feasible(bl, ct, (0, -1), 0.7) and not cra_feasible(bl, ct, (0, -1), 0.7)
    assert rbe_feasible(bl, ct, (0, -1), 0.05)  # RBE: any mu > 0
    print("sanity checks passed")


# --------------------------------------------------------------------------
# Figure 1: model H
# --------------------------------------------------------------------------
def draw_h_scene(ax, w, h, dx=0.0, dth=0.0, closed=()):
    a, b = w / 2, h / 2
    gap = 0.0
    for sx in (-1, 1):  # walls
        x0 = sx * (a + gap)
        ax.add_patch(Polygon([[x0, -1.0 * h], [x0 + sx * 0.35 * w, -1.0 * h],
                              [x0 + sx * 0.35 * w, 1.0 * h], [x0, 1.0 * h]],
                             closed=True, fc=STONE, ec=INK2, lw=1, zorder=1))
    c, s_ = np.cos(dth), np.sin(dth)
    R = np.array([[c, -s_], [s_, c]])
    corners = np.array([[-a, -b], [a, -b], [a, b], [-a, b]]) @ R.T + np.array([dx, 0.0])
    ax.add_patch(Polygon(corners, closed=True, fc=WOOD, ec=INK2, lw=1, zorder=3))
    verts = {"LT": (-a, b), "LB": (-a, -b), "RT": (a, b), "RB": (a, -b)}
    for name, p in verts.items():
        q = R @ np.array(p) + np.array([dx, 0.0])
        ax.plot(q[0], q[1], marker="o", ms=6, mfc=(INK if name in closed else SURFACE), mec=INK, zorder=8)
    ax.set_xlim(-a - 0.4 * w, a + 0.4 * w)
    ax.set_ylim(-1.45 * h, 1.05 * h)
    clean(ax, equal=True, hide_axes=True)
    return corners


def arrow(ax, p, v, color, zorder=7):
    if np.linalg.norm(v) < 1e-6:
        return
    ax.annotate("", xy=(p[0] + v[0], p[1] + v[1]), xytext=(p[0], p[1]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=12), zorder=zorder)


def figure1():
    w, h, mu = 1.0, 0.8, 0.7
    blocks, contacts = model_h(w, h)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.9))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.14, wspace=0.05)
    fscale = 0.55 * h  # length of W
    half = np.degrees(np.arctan(mu))
    # (a) RBE
    ax = axes[0]
    draw_h_scene(ax, w, h, closed=("LT", "LB", "RT", "RB"))
    f = rbe_forces(blocks, contacts, (0, -1), mu)
    for (_, _, p, n, t), (fn, ft) in zip(contacts, f):
        nang = np.degrees(np.arctan2(n[1], n[0]))
        ax.add_patch(Wedge(p, 0.5 * fscale, nang - half, nang + half, fc=SEQ[0], ec=SEQ[3], lw=0.8, zorder=4))
        arrow(ax, p, (fn * np.asarray(n) + ft * np.asarray(t)) * fscale, BLUE)
    arrow(ax, (0, 0), (0, -fscale), INK2)
    ax.text(0.05 * w, -0.5 * fscale, "W", color=INK2, fontsize=10)
    ax.text(0, -1.12 * h, f"per wall: N = {f[:2, 0].sum():.2f} W = W/(2μ)\nfriction = {f[:2, 1].sum():.2f} W = μN",
            ha="center", va="top", fontsize=8, color=INK, zorder=9)
    ax.set_title("(a) RBE: feasible for any μ > 0\nsqueeze W/(2μ) buys the friction", fontsize=8.5)
    # (b) CRA, translation
    ax = axes[1]
    d = 0.06 * w
    draw_h_scene(ax, w, h, dx=-d, closed=("LT", "LB"))
    arrow(ax, (0, 0), (-1.3 * d, 0), INK2)
    ax.text(0.03 * w, 0.06 * h, "δx = −ε", color=INK2, fontsize=8.5)
    ax.text(-w / 2 - 0.04 * w, 0, "closed\n(δdₙ = −ε)", ha="right", va="center", fontsize=8, color=INK)
    ax.text(w / 2 + 0.02 * w, 0, "open\n(no force)", ha="left", va="center", fontsize=8, color=INK)
    ax.text(0, -1.12 * h, "ΣFₓ = 0 ⇒ N_left = 0\n⇒ no friction, W unbalanced", ha="center", va="top", fontsize=8, color=INK)
    ax.set_title("(b) CRA, try 1: translate onto a wall\nthe other wall opens", fontsize=8.5)
    # (c) CRA, rotation (drawer jam attempt)
    ax = axes[2]
    th = 0.09
    corners = draw_h_scene(ax, w, h, dth=th, closed=("LT", "RB"))
    a, b = w / 2, h / 2
    N = a / (2 * b)                    # largest N that keeps the LT friction >= 0
    F_rb, F_lt = 1.0, 0.0              # best attempt: all the weight carried at RB
    c_, s_ = np.cos(th), np.sin(th)
    R = np.array([[c_, -s_], [s_, c_]])
    pRB = R @ np.array([a, -b])
    pLT = R @ np.array([-a, b])
    for p, n, F in ((pRB, (-1.0, 0.0), F_rb), (pLT, (1.0, 0.0), F_lt)):
        nang = np.degrees(np.arctan2(n[1], n[0]))
        ax.add_patch(Wedge(p, 0.5 * fscale, nang - half, nang + half, fc=SEQ[0], ec=SEQ[3], lw=0.8, zorder=4))
        arrow(ax, p, (N * np.asarray(n) + F * np.array([0.0, 1.0])) * fscale, BAD)
    arrow(ax, (0, 0), (0, -fscale), INK2)
    ax.text(0.05 * w, -0.5 * fscale, "W", color=INK2, fontsize=10)
    ax.text(0, -1.12 * h, f"best attempt leans {np.degrees(np.arctan(2 * b / a)):.0f}° off the normal\n"
                          f"> cone {half:.0f}°: needs μ ≥ 2h/w = {2 * h / w:.1f}", ha="center", va="top", fontsize=8, color=INK)
    ax.set_title("(c) CRA, try 2: rotate onto two corners\nneeded friction is outside the cone", fontsize=8.5)
    handles = [Line2D([], [], color=BLUE, lw=2, label="RBE corner force (solved)"),
               Line2D([], [], color=BAD, lw=2, label="force CRA would need"),
               Line2D([], [], color=INK2, lw=2, label="weight W or virtual motion"),
               Line2D([], [], marker="o", ms=6, mfc=INK, mec=INK, ls="none", label="closed vertex"),
               Line2D([], [], marker="o", ms=6, mfc=SURFACE, mec=INK, ls="none", label="open vertex"),
               Patch(fc=SEQ[0], ec=SEQ[3], label=f"friction cone, μ = {mu}")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Model H (w = {w:g}, h = {h:g}, μ = {mu}): RBE finds forces, CRA finds no motion that could cause them",
                 fontsize=10, fontweight="semibold", y=0.97)
    save(fig, f"{SLUG}-1.png")


# --------------------------------------------------------------------------
# Figure 3: the rotational jam threshold  (our observation)
# --------------------------------------------------------------------------
def figure_jam():
    ratios = np.array([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    thresholds = []
    for r in ratios:
        blocks, contacts = model_h(1.0, r)
        lo, hi = 0.05, 3.0
        assert not cra_feasible(blocks, contacts, (0, -1), lo)
        if not cra_feasible(blocks, contacts, (0, -1), hi):
            thresholds.append(np.nan)
            continue
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if cra_feasible(blocks, contacts, (0, -1), mid):
                hi = mid
            else:
                lo = mid
        thresholds.append(0.5 * (lo + hi))
    thresholds = np.array(thresholds)
    print("CRA jam threshold mu:", dict(zip(ratios, np.round(thresholds, 3))))
    ok = ~np.isnan(thresholds)
    assert np.allclose(thresholds[ok], 2 * ratios[ok], atol=0.01)

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    rr = np.linspace(0.1, 1.05, 50)
    ax.fill_between(rr, 2 * rr, 3.0, color=GOOD, alpha=0.10, lw=0)
    ax.fill_between(rr, 0, 2 * rr, color=BAD, alpha=0.10, lw=0)
    ax.plot(rr, 2 * rr, color=ORANGE, lw=2, ls="--", label="analytic jam limit μ = 2h/w")
    ax.plot(ratios[ok], thresholds[ok], color=BLUE, marker="o", ms=5, ls="none", label="CRA threshold (pattern enumeration)")
    ax.axhline(0.84, color=MUTED, lw=1, ls=":")
    ax.text(1.03, 0.84 + 0.03, "μ = 0.84 (paper's benchmarks)", color=INK2, fontsize=8, ha="right")
    ax.text(0.14, 1.75, "CRA feasible: rotational 'drawer jam'", fontsize=9, color=INK, va="center")
    ax.text(0.58, 0.45, "CRA infeasible: block falls", fontsize=9, color=INK, va="center")
    ax.set_xlabel("block height / wall spacing  h / w")
    ax.set_ylabel("friction coefficient μ")
    ax.set_xlim(0.1, 1.05)
    ax.set_ylim(0, 3.0)
    ax.set_title("CRA certifies model H only when μ ≥ 2h/w: a jam by rotation, not a squeeze")
    ax.legend(loc="upper left")
    clean(ax)
    fig.tight_layout()
    save(fig, f"{SLUG}-3.png")


# --------------------------------------------------------------------------
# Figure 2: keystone wedge tilt tests
# --------------------------------------------------------------------------
def figure_wedge():
    mus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.84, 1.0, 1.2, 1.5, 2.0]
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.7), gridspec_kw={"width_ratios": [1, 1.6]})
    # geometry inset
    ax = axes[0]
    for beta, col, dx in ((15, BLUE, -0.75), (60, ORANGE, 0.75)):
        b = np.radians(beta)
        tri = np.array([[0, 0], [-np.sin(b), np.cos(b)], [np.sin(b), np.cos(b)]]) + np.array([dx, 0])
        notch = np.array([[dx - 1.3, -0.5], [dx + 1.3, -0.5], [dx + 1.3, np.cos(b)],
                          [dx + np.sin(b), np.cos(b)], [dx, 0], [dx - np.sin(b), np.cos(b)], [dx - 1.3, np.cos(b)]])
        ax.add_patch(Polygon(notch, closed=True, fc=STONE, ec=INK2, lw=1, zorder=1))
        ax.add_patch(Polygon(tri, closed=True, fc=WOOD, ec=col, lw=2, zorder=3))
        ax.text(dx, 1.12, f"β = {beta}°\n{'sharp' if beta < 30 else 'blunt'}", ha="center", fontsize=9, color=INK)
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-0.6, 1.5)
    ax.set_title("2D keystone in a V-notch\n(flank angle β from vertical)", fontsize=10)
    clean(ax, equal=True, hide_axes=True)
    # tilt curves
    ax = axes[1]
    results = {}
    for beta, col, lab in ((15, BLUE, "sharp β = 15°"), (60, ORANGE, "blunt β = 60°")):
        blocks, contacts = keystone(beta)
        r = [critical_tilt(rbe_feasible, blocks, contacts, mu) for mu in mus]
        c = [critical_tilt(cra_feasible, blocks, contacts, mu) for mu in mus]
        results[beta] = (r, c)
        ax.plot(mus, r, color=col, ls="--", marker="s", ms=4, mfc="none", label=f"RBE, {lab}")
        ax.plot(mus, c, color=col, ls="-", marker="o", ms=4, label=f"CRA, {lab}")
    print("tilt results (mu: RBE, CRA):")
    for beta in results:
        print(beta, [f"{m}:{r:.1f}/{c:.1f}" for m, r, c in zip(mus, *results[beta])])
    ax.axhline(180, color=MUTED, lw=1, ls=":")
    ax.text(2.0, 183, "upside down", ha="right", fontsize=8, color=INK2)
    ax.axvline(np.tan(np.radians(15)), color=MUTED, lw=1, ls=":")
    ax.text(np.tan(np.radians(15)) + 0.03, 100, "μ = tan 15°", fontsize=8, color=INK2, rotation=90, va="center")
    ax.set_xlabel("friction coefficient μ")
    ax.set_ylabel("critical tilt of gravity (degrees)")
    ax.set_ylim(0, 200)
    ax.set_xlim(0, 2.1)
    ax.set_title("Sharp wedge: RBE says stable upside down once μ > tan β;\nCRA stays finite and μ-dependent", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    clean(ax)
    fig.tight_layout()
    save(fig, f"{SLUG}-2.png")


# --------------------------------------------------------------------------
# Figure 4: Table 2 of the paper
# --------------------------------------------------------------------------
def figure4():
    rows = [("type-a", "x", 21.8, 21.8, 25.0, 59.2, 59.2, 180.0),
            ("type-a", "y", 71.3, 71.3, 71.3, 100.0, 100.0, 180.0),
            ("type-a", "xy30", 24.8, 24.7, 28.0, 62.7, 62.9, 180.0),
            ("type-b", "x", 21.8, 21.8, 25.0, 58.7, 58.5, 180.0),
            ("type-b", "y", 67.4, 67.4, 67.4, 109.1, 109.1, 180.0),
            ("type-b", "xy30", 24.8, 24.7, 28.0, 63.1, 62.0, 180.0),
            ("type-c", "x", 13.0, 13.0, 13.0, 44.1, 44.1, 47.0),
            ("type-c", "y", 41.3, 41.3, 41.3, 64.5, 64.4, 65.7),
            ("type-c", "xy30", 14.9, 14.9, 15.0, 48.1, 40.2, 51.0),
            ("type-d", "x", 13.0, 13.0, 13.0, 44.1, 44.1, 50.0),
            ("type-d", "y", 41.3, 41.3, 41.3, 70.0, 70.0, 70.0),
            ("type-d", "xy30", 14.9, 14.9, 15.0, 49.0, 48.5, 54.0)]
    labels = [f"{t}, rotate-{ax_}" for t, ax_, *_ in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), sharey=True)
    for ax, (mu, off) in zip(axes, ((0.20, 2), (0.84, 5))):
        d3 = [r[off] for r in rows]
        cra = [r[off + 1] for r in rows]
        rbe = [r[off + 2] for r in rows]
        for i in range(len(rows)):
            ax.plot([min(d3[i], cra[i], rbe[i]), max(d3[i], cra[i], rbe[i])], [i, i], color=GRID, lw=3, zorder=1)
        ax.plot(rbe, y, ls="none", marker="s", ms=6, mfc="none", mec=ORANGE, mew=1.6, label="RBE (force only)", zorder=3)
        ax.plot(d3, y, ls="none", marker="o", ms=7, mfc=AQUA, mec=AQUA, label="3DEC (dynamic)", zorder=4)
        ax.plot(cra, y, ls="none", marker="D", ms=4.5, mfc=BLUE, mec=BLUE, label="CRA", zorder=5)
        ax.set_xlabel(f"critical tilt angle (degrees), μ = {mu}")
        ax.set_xlim(0, 190)
        ax.axvline(180, color=MUTED, lw=1, ls=":")
        clean(ax)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_title("μ = 0.20: all three agree to within 4°", fontsize=10)
    axes[1].set_title("μ = 0.84: RBE gives 180° for a and b", fontsize=10)
    axes[0].legend(loc="center right", fontsize=8)
    fig.suptitle("3-block wedge tilt tests (paper Table 2): CRA tracks 3DEC, RBE fails on sharp wedges at high μ",
                 fontsize=10, fontweight="semibold")
    fig.tight_layout()
    save(fig, f"{SLUG}-4.png")


if __name__ == "__main__":
    sanity()
    figure1()
    figure_wedge()
    figure_jam()
    figure4()
