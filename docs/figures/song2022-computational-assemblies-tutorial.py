"""Figures for docs/song2022-computational-assemblies-tutorial.md (run from docs/figures/).

1. The tutorial's "stability spectrum" read off a computed quantity: the kinematic (motion-based)
   infeasibility energy  E = max_v  w.v - 1/2 v.v  s.t. v in the motion cone, evaluated while the
   ground of a 2-block stack is tilted.  E = 0 means equilibrium; the width of the E = 0 range is
   lateral stability; E > 0 at zero tilt is non-equilibrium.
2. Static-kinematic duality on the same toy: the force-based energy (Whiting-style squared tension)
   and the kinematic energy vanish at exactly the same tilt.
"""
from _style import *  # noqa: F401,F403

import numpy as np
from matplotlib.patches import Polygon, FancyArrowPatch
from scipy.optimize import nnls, minimize

SLUG = "song2022-computational-assemblies-tutorial"
np.random.seed(0)

MU = 0.5  # friction coefficient used throughout


# ----------------------------------------------------------------------------------------
# Toy: block 1 on the ground, block 2 on block 1 (both w x h, top block shifted by `offset`).
# Planar rigid motion of block i: (vx_i, vz_i, omega_i).  Ground is fixed.
# ----------------------------------------------------------------------------------------
def stack_geometry(w, h, offset):
    b1 = dict(c=np.array([0.0, h / 2]), corners=[(-w / 2, 0), (w / 2, 0)])
    # the block-to-block interface is the overlap of the two footprints; its end points carry the contacts
    lo, hi = max(-w / 2, offset - w / 2), min(w / 2, offset + w / 2)
    assert hi > lo
    b2 = dict(c=np.array([offset, 1.5 * h]), corners=[(lo, h), (hi, h)])
    return b1, b2


def contact_rows(w, h, offset, mu):
    """Rows of B (6 columns: block 1 then block 2) such that B v >= 0 is the motion cone.
    With friction the admissible force at a contact lies in the cone spanned by the two rays
    n +- mu t; by polarity the admissible *relative* velocity must have v_rel . (n +- mu t) >= 0.
    Rows are written for the relative velocity of the upper body with respect to the lower one."""
    b1, b2 = stack_geometry(w, h, offset)
    rays = [np.array([mu, 1.0]), np.array([-mu, 1.0])]  # n = (0,1), t = (1,0): n + mu t, n - mu t
    rows = []
    # ground -> block 1 (lower body fixed)
    for p in b1["corners"]:
        r = np.array(p) - b1["c"]
        for d in rays:
            # velocity of the point on block 1: v1 + omega1 x r = (vx - w rz, vz + w rx)
            rows.append([d[0], d[1], -r[1] * d[0] + r[0] * d[1], 0, 0, 0])
    # block 1 -> block 2: relative velocity = point velocity on block 2 minus point velocity on block 1
    for p in b2["corners"]:
        r2 = np.array(p) - b2["c"]; r1 = np.array(p) - b1["c"]
        for d in rays:
            rows.append([-d[0], -d[1], -(-r1[1] * d[0] + r1[0] * d[1]),
                         d[0], d[1], -r2[1] * d[0] + r2[0] * d[1]])
    return np.array(rows)


def load_vector(phi, Wt=1.0):
    """Generalised gravity (force on each block's centre, no torque) in the frame of the tilted ground."""
    g = np.array([np.sin(phi), -np.cos(phi)]) * Wt
    return np.array([g[0], g[1], 0.0, g[0], g[1], 0.0])


def kinematic_energy(B, w):
    """E = max_{B v >= 0} w.v - 1/2 v.v = 1/2 |P_C(w)|^2 with C = {v : B v >= 0}.
    By Moreau's decomposition P_C(w) = w - P_{C°}(w), and P_{C°}(w) = -B^T lam* with
    lam* = argmin_{lam >= 0} |B^T lam + w|  (a non-negative least squares problem).
    lam* are the contact force magnitudes along the cone rays, so the same number is the
    least-squares equilibrium residual  1/2 min_{f in cones} |A f + w|^2  (static-kinematic duality)."""
    lam, res = nnls(B.T, -w)
    v = w + B.T @ lam
    assert np.all(B @ v >= -1e-9)  # the projection lies in the motion cone
    return 0.5 * float(v @ v), v, lam


def force_energy(B, w):
    """Whiting-style force-based infeasibility: split each cone-ray force into f+ - f- (both >= 0),
    enforce exact equilibrium  A (f+ - f-) + w = 0  with A = B^T, and minimise sum |f-|^2."""
    m = B.shape[0]
    A = B.T
    x0 = np.zeros(2 * m)
    def obj(x):
        return float(x[m:] @ x[m:])
    cons = {"type": "eq", "fun": lambda x: A @ (x[:m] - x[m:]) + w}
    res = minimize(obj, x0, method="SLSQP", constraints=[cons], bounds=[(0, None)] * (2 * m),
                   options={"ftol": 1e-14, "maxiter": 500})
    assert res.success and np.linalg.norm(A @ (res.x[:m] - res.x[m:]) + w) < 1e-6
    return float(res.fun)


def draw_stack(ax, w, h, offset, x0, label):
    """Draw one 2-block stack (untilted) with its base centred at x0 in a shared axes."""
    ax.add_patch(Polygon([(x0 - 1.0, 0), (x0 + 1.0, 0), (x0 + 1.0, -0.15), (x0 - 1.0, -0.15)], closed=True, fc=WOOD, ec=INK2, lw=0.7))
    for (cx, z0) in [(x0, 0.0), (x0 + offset, h)]:
        ax.add_patch(Polygon([(cx - w / 2, z0), (cx + w / 2, z0), (cx + w / 2, z0 + h), (cx - w / 2, z0 + h)], closed=True,
                             fc=WOOD2, ec=INK2, lw=0.7))
    ax.text(x0, -0.35, label, ha="center", va="top", fontsize=8, color=INK2)


def fig1():
    w, h = 1.0, 1.0
    designs = [(0.70, BLUE, "overhang 0.70 w"), (0.35, ORANGE, "overhang 0.35 w"), (0.0, AQUA, "centred")]
    phis = np.radians(np.linspace(0, 40, 161))
    fig, (axg, ax) = plt.subplots(1, 2, figsize=(8, 3.6), gridspec_kw={"width_ratios": [1, 1.7]})
    crit = {}
    for i, (off, colr, lab) in enumerate(designs):
        B = contact_rows(w, h, off, MU)
        E = np.array([kinematic_energy(B, load_vector(p))[0] for p in phis])
        ax.plot(np.degrees(phis), E, color=colr, lw=2, label=lab)
        idx = np.argmax(E > 1e-10)  # first tilt with E > 0
        crit[off] = np.degrees(phis[idx]) if E[idx] > 1e-10 else np.nan
        draw_stack(axg, w, h, off, 3.0 * i, lab.replace("overhang ", "overhang\n"))
    axg.set_xlim(-1.3, 7.3); axg.set_ylim(-1.3, 2.6)
    axg.set_aspect("equal")
    axg.set_title("Three designs, $w$ = $h$ = 1, at zero tilt", fontsize=9.5)
    clean(axg, hide_axes=True)
    # sanity checks against closed forms for the centred stack: a 1 x 2 column of two blocks tips at
    # tan(phi) = (w/2)/h = 0.5 (both blocks together about the ground corner) or slides at tan(phi) = mu = 0.5
    assert abs(crit[0.0] - np.degrees(np.arctan(0.5))) < 0.3, crit
    # overhang 0.70 > w/2: the top block's centre lies outside the lower block -> unstable at zero tilt
    assert kinematic_energy(contact_rows(w, h, 0.70, MU), load_vector(0.0))[0] > 1e-6
    ax.set_xlabel("ground tilt $\\phi$ (degrees)")
    ax.set_ylabel("kinematic infeasibility energy $E$ (units of $W^2$)")
    ax.set_xlim(0, 40); ax.set_ylim(-0.005, 0.135)
    for off, colr, lab in designs[1:]:
        ax.axvline(crit[off], color=colr, lw=1, ls=":")
        ax.text(crit[off] + 0.4, 0.126, f"{crit[off]:.1f}$^\\circ$", color=INK2, fontsize=8)
    ax.text(0.6, 0.037, "non-equilibrium\nalready at $\\phi$ = 0", color=INK2, fontsize=8, va="bottom")
    ax.text(1.5, 0.004, "flat stretch at $E$ = 0: equilibrium; its width is the lateral-stability margin",
            color=INK2, fontsize=7.5, va="bottom")
    ax.legend(loc="upper left", title="top block", title_fontsize=8)
    ax.set_title("Stability as a number: how fast the stack could start to fall", fontsize=9.5)
    clean(ax)
    fig.suptitle(f"Two stacked blocks, $\\mu$ = {MU}: the motion-based energy is zero exactly while equilibrium holds",
                 fontsize=10, y=1.0)
    fig.tight_layout()
    save(fig, f"{SLUG}-1.png")


def fig2():
    w, h, off = 1.0, 1.0, 0.35
    B = contact_rows(w, h, off, MU)
    phis = np.radians(np.linspace(0, 40, 81))
    Ek = np.array([kinematic_energy(B, load_vector(p))[0] for p in phis])
    Ef = np.array([force_energy(B, load_vector(p)) for p in phis])
    # same zero set (up to solver tolerance)
    zk, zf = Ek > 1e-8, Ef > 1e-8
    assert np.array_equal(zk, zf), (np.degrees(phis[zk != zf]))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(np.degrees(phis), Ek, color=BLUE, lw=2, label="kinematic: $\\max_{v\\in\\mathrm{cone}}\\; w\\cdot v - \\frac{1}{2} v\\cdot v$")
    ax.plot(np.degrees(phis), Ef, color=ORANGE, lw=2, ls="--", label="force-based: $\\min \\sum \\|F^-\\|^2$ s.t. $A(F^+ - F^-) + w = 0$")
    c = np.degrees(phis[np.argmax(zk)])
    ax.axvline(c, color=MUTED, lw=1, ls=":")
    ax.text(c + 0.5, ax.get_ylim()[1] * 0.55, f"both leave zero at {c:.1f}$^\\circ$", color=INK2, fontsize=8)
    ax.set_xlabel("ground tilt $\\phi$ (degrees)")
    ax.set_ylabel("infeasibility energy (units of $W^2$)")
    ax.set_title("Force view and motion view agree on when equilibrium is lost, not on the size of the violation", fontsize=10)
    ax.legend(loc="upper left")
    clean(ax)
    fig.tight_layout()
    save(fig, f"{SLUG}-2.png")


if __name__ == "__main__":
    fig1()
    fig2()
