"""Figures for docs/umetani2012-guided-exploration-furniture.md (run from docs/figures/).

Toy plank model (ours): a base plank C on the ground, a vertical post A nailed to its end, and a
shelf B nailed to the post that cantilevers over the base by a length L; a load W hangs at the tip.
1. Toppling = a ground contact force going negative (Eq. 8), shown along a drag of L and in the
   force space of the two ground contacts (the paper's Fig. 11 idea: first quadrant = stable).
2. Nail-joint validity: the bending-moment box |h^r_y| <= Lambda_max (Eq. 9) and the (L, W) map of
   valid designs with the toppling and pull-out boundaries; paper's Fig. 18 effect of thickness.
3. Table 1 timings.
"""
from _style import *  # noqa: F401,F403

import numpy as np
from matplotlib.patches import Polygon, FancyArrowPatch

SLUG = "umetani2012-guided-exploration-furniture"
np.random.seed(0)

G = 9.81
RHO = 750.0          # kg/m^3, MDF (our assumption; the paper gives no density)
T = 0.012            # plank thickness l_z, 12 mm (paper, Sec. 5.1)
DEPTH = 0.25         # plank depth into the page, m (ours)
H_POST = 0.60        # post height, m
B_BASE = 0.30        # base plank length, m (from the post to the toe)
H_SHELF = 0.45       # shelf height above ground, m
N_NAIL = 3           # nails at the shelf joint (as in the paper's supplement experiment)
F_PULL_MAX = 35000.0 * (0.032 - T)   # N per nail: 35 kN/m x 20 mm embedded (paper + supplement)
F_SHEAR_MAX = 190.0  # N per nail (paper, Sec. 7)
LAMBDA_MAX = 0.5 * T * F_PULL_MAX * N_NAIL   # Eq. 9 with h^t_n = 0 (no horizontal load)
LAMBDA_SUPP = 1.86   # N m, the supplement's stated value for the same joint (its arithmetic differs)


def plank_weight(length, t=T):
    return RHO * length * DEPTH * t * G


def ground_forces(L, W):
    """Two ground contacts: heel at x = -T (outer bottom corner of the post) and toe at x = B_BASE.
    Vertical balance and moment balance about the toe give the two normal forces uniquely (2D)."""
    xh, xt = -T, B_BASE
    Wc, Wa, Wb = plank_weight(B_BASE), plank_weight(H_POST), plank_weight(L)
    loads = [(Wc, B_BASE / 2), (Wa, -T / 2), (Wb, L / 2), (W, L)]  # (magnitude, x of line of action)
    total = sum(m for m, _ in loads)
    moment_about_toe = sum(m * (xt - x) for m, x in loads)  # positive = restoring (weight left of toe)
    f_heel = moment_about_toe / (xt - xh)
    f_toe = total - f_heel
    return f_heel, f_toe


def joint_demand(L, W, t=T):
    """Bending moment at the shelf joint (about the joint point on the post face) and the
    per-nail pull and shear forces of Eqs. 4-5 (h^t_n = 0 because nothing pushes horizontally)."""
    Wb = plank_weight(L, t)
    M = W * L + Wb * L / 2
    f_pull = (2 * M / t - 0.0) / N_NAIL
    f_shear = (W + Wb) / N_NAIL
    return M, f_pull, f_shear


def L_topple(W):
    Ls = np.linspace(0.01, 2.0, 4000)
    fh = np.array([ground_forces(L, W)[0] for L in Ls])
    i = np.argmax(fh < 0)
    return Ls[i] if fh[i] < 0 else np.nan


def L_pull(W, lam=LAMBDA_MAX, t=T):
    Ls = np.linspace(0.01, 2.0, 4000)
    M = np.array([joint_demand(L, W, t)[0] for L in Ls])
    i = np.argmax(M > lam)
    return Ls[i] if M[i] > lam else np.nan


def draw_model(ax, L, W, fh, ft):
    ax.add_patch(Polygon([(0, 0), (B_BASE, 0), (B_BASE, T), (0, T)], fc=WOOD, ec=INK2, lw=0.7))            # base C
    ax.add_patch(Polygon([(-T, 0), (0, 0), (0, H_POST), (-T, H_POST)], fc=WOOD2, ec=INK2, lw=0.7))         # post A
    ax.add_patch(Polygon([(0, H_SHELF - T), (L, H_SHELF - T), (L, H_SHELF), (0, H_SHELF)], fc=WOOD, ec=INK2, lw=0.7))  # shelf B
    ax.plot([0], [H_SHELF - T / 2], "o", ms=4, color=INK, zorder=5)
    ax.text(0.03, H_SHELF - T - 0.03, "nail joint", fontsize=8, color=INK2, va="top")
    ax.text(-0.08, H_POST / 2, "A", fontsize=9, color=INK); ax.text(B_BASE / 2, T + 0.02, "C", fontsize=9, color=INK, ha="center")
    ax.text(L / 2, H_SHELF + 0.02, "B", fontsize=9, color=INK, ha="center")
    sc = 0.004  # m of arrow per N
    ax.add_patch(FancyArrowPatch((L, H_SHELF + W * sc), (L, H_SHELF), arrowstyle="-|>", mutation_scale=9, color=INK, lw=1.6))
    ax.text(L + 0.02, H_SHELF + W * sc / 2, f"W = {W:.0f} N", fontsize=8, color=INK, va="center")
    for x, f, name in [(-T, fh, "heel"), (B_BASE, ft, "toe")]:
        colr = GOOD if f >= 0 else BAD
        if f >= 0:  # the ground pushes up
            ax.add_patch(FancyArrowPatch((x, -0.03 - f * sc), (x, -0.03), arrowstyle="-|>", mutation_scale=9, color=colr, lw=1.6))
        else:       # the ground would have to pull down
            ax.add_patch(FancyArrowPatch((x, -0.03), (x, -0.03 + f * sc), arrowstyle="-|>", mutation_scale=9, color=colr, lw=1.6))
        ax.text(x, -0.06 - abs(f) * sc, f"{name}\n{f:+.1f} N", ha="center", va="top", fontsize=8, color=colr)
    ax.plot([-0.12, 0.75], [0, 0], color=INK2, lw=0.8)
    ax.set_xlim(-0.14, 0.82); ax.set_ylim(-0.33, H_POST + 0.06)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    clean(ax)


def fig1():
    W = 10.0
    Ls = np.linspace(0.05, 0.9, 400)
    F = np.array([ground_forces(L, W) for L in Ls])
    Lc = L_topple(W)
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.4), gridspec_kw={"width_ratios": [1.1, 1.25, 1]})
    ax = axes[0]
    fh, ft = ground_forces(0.70, W)
    draw_model(ax, 0.70, W, fh, ft)
    ax.set_title("Model at L = 0.70 m", fontsize=9.5)
    ax = axes[1]
    ax.plot(Ls, F[:, 0], color=BLUE, lw=2, label="heel contact (x = −12 mm)")
    ax.plot(Ls, F[:, 1], color=ORANGE, lw=2, label="toe contact (x = 0.30 m)")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvspan(Lc, Ls[-1], color=BAD, alpha=0.08, lw=0)
    ax.axvline(Lc, color=BAD, lw=1, ls=":")
    ax.text(Lc - 0.02, 22, f"topples at\nL = {Lc:.2f} m", ha="right", va="top", fontsize=8, color=INK2)
    ax.set_xlabel("shelf length L (m)"); ax.set_ylabel("ground normal force (N)")
    ax.set_title(f"Drag L with W = {W:.0f} N", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_xlim(Ls[0], Ls[-1]); ax.set_ylim(-10, 62)
    clean(ax)
    ax = axes[2]
    ax.fill_betweenx([0, 62], 0, 30, color=GOOD, alpha=0.12, lw=0)
    ax.plot(F[:, 0], F[:, 1], color=BLUE, lw=2)
    ax.plot(F[0, 0], F[0, 1], "o", color=BLUE, ms=5); ax.plot(F[-1, 0], F[-1, 1], "s", color=BLUE, ms=5)
    ax.text(F[0, 0] - 1.0, F[0, 1] + 2.5, "L = 0.05 m", fontsize=8, color=INK2, ha="right")
    ax.text(-13.5, 52, "L = 0.90 m", fontsize=8, color=INK2, va="top")
    ax.axvline(0, color=INK2, lw=0.8); ax.axhline(0, color=INK2, lw=0.8)
    ax.text(4, 17, "stable quadrant:\nboth forces ≥ 0", fontsize=8, color=INK2, ha="center")
    ax.set_xlabel("heel force (N)"); ax.set_ylabel("toe force (N)")
    ax.set_xlim(-14, 30); ax.set_ylim(0, 62)
    ax.set_title("Same drag in force space", fontsize=9.5)
    clean(ax)
    fig.suptitle("Toppling shows up as a ground contact force that would have to become negative (Eq. 8)", fontsize=10, y=1.0)
    fig.tight_layout(w_pad=1.5)
    save(fig, f"{SLUG}-1.png")


def fig2():
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 3.5), gridspec_kw={"width_ratios": [1, 1.15]})
    # --- left: demand vs box along a drag of L, with the linearised prediction (sensitivity) vs the exact crossing
    W = 10.0
    Ls = np.linspace(0.0, 0.9, 400)
    M = np.array([joint_demand(L, W)[0] for L in Ls])
    ax0.plot(Ls, M, color=BLUE, lw=2, label="bending moment $|h^r_y|$ at the joint")
    ax0.axhspan(-LAMBDA_MAX, LAMBDA_MAX, color=GOOD, alpha=0.10, lw=0)
    ax0.axhline(LAMBDA_MAX, color=INK2, lw=0.8)
    ax0.text(0.02, LAMBDA_MAX + 0.4, f"$\\Lambda_{{max}}$ = {LAMBDA_MAX:.1f} N·m: box edge (Eq. 9), {N_NAIL} nails, $l_z$ = 12 mm", fontsize=8, color=INK2)
    ax0.axhline(LAMBDA_SUPP, color=INK2, lw=0.8, ls="--")
    ax0.text(0.88, LAMBDA_SUPP + 0.4, f"{LAMBDA_SUPP} N·m as stated in the supplement", fontsize=8, color=INK2, ha="right")
    # sensitivity at the current design L0: K0 = dM/dL, ray to the box edge
    L0 = 0.30
    M0 = joint_demand(L0, W)[0]
    K0 = (joint_demand(L0 + 1e-6, W)[0] - joint_demand(L0 - 1e-6, W)[0]) / 2e-6
    t_lin = (LAMBDA_MAX - M0) / K0
    L_exact = L_pull(W)
    ax0.plot([L0, L0 + t_lin], [M0, LAMBDA_MAX], color=ORANGE, lw=1.6, ls="--", label="linearised ray from L = 0.30 m")
    ax0.plot(L0, M0, "o", color=ORANGE, ms=5)
    ax0.plot(L0 + t_lin, LAMBDA_MAX, "v", color=ORANGE, ms=6)
    ax0.plot(L_exact, LAMBDA_MAX, "o", color=BLUE, ms=6, mfc="white", mew=1.5)
    ax0.text(0.02, 18.6, f"box edge reached at\nL = {L0 + t_lin:.2f} m (linear ray)\nL = {L_exact:.2f} m (exact)", ha="left", va="top", fontsize=8, color=INK2)
    ax0.set_xlabel("shelf length L (m)"); ax0.set_ylabel("joint bending moment (N·m)")
    ax0.set_xlim(0, 0.9); ax0.set_ylim(0, 26)
    ax0.set_title(f"Durability box along a drag, W = {W:.0f} N", fontsize=9.5)
    ax0.legend(loc="upper left", fontsize=7.5)
    clean(ax0)
    # --- right: (L, W) validity map
    Ws = np.linspace(0, 40, 161)
    Lt = np.array([L_topple(w) for w in Ws])
    Lp = np.array([L_pull(w) for w in Ws])
    # thicker plank (paper Fig. 18): Lambda_max scales with l_z and the plank weight scales too
    t18 = 0.018
    Lp18 = np.array([L_pull(w, 0.5 * t18 * F_PULL_MAX * N_NAIL, t18) for w in Ws])
    valid = np.minimum(Lt, Lp)
    ax1.fill_betweenx(Ws, 0, valid, color=GOOD, alpha=0.12, lw=0)
    ax1.plot(Lt, Ws, color=BLUE, lw=2, label="toppling: heel force = 0")
    ax1.plot(Lp, Ws, color=ORANGE, lw=2, label="nail pull-out: $|h^r_y| = \\Lambda_{max}$, $l_z$ = 12 mm")
    ax1.plot(Lp18, Ws, color=ORANGE, lw=1.6, ls="--", label="nail pull-out, $l_z$ = 18 mm (Fig. 18 effect)")
    ax1.text(0.15, 24, "valid", fontsize=9, color=INK2)
    ax1.text(0.72, 22, "joint fails\nor topples", fontsize=8, color=INK2, ha="center")
    # shear never binds here: (W + W_B)/N <= 190 N needs W + W_B > 570 N
    assert joint_demand(0.9, 40)[2] < F_SHEAR_MAX
    ax1.set_xlabel("shelf length L (m)"); ax1.set_ylabel("tip load W (N)")
    ax1.set_xlim(0, 0.9); ax1.set_ylim(0, 40)
    ax1.set_title("Valid (L, W) region = inside both boundaries", fontsize=9.5)
    ax1.legend(loc="lower left", fontsize=7.5)
    clean(ax1)
    fig.suptitle("Nail-joint validity: a box in moment space, whose edge maps to a curve in the design space", fontsize=10, y=1.0)
    fig.tight_layout(w_pad=2)
    save(fig, f"{SLUG}-2.png")


def fig3():
    labels = ["Fig. 17 right\n9 planks, 13 joints", "Fig. 16 right\n10 planks, 13 joints",
              "Fig. 16 middle\n20 planks, 33 joints", "Fig. 1 right\n28 planks, 49 joints"]
    cand = np.array([13.2, 22.3, 160, 758])
    line = np.array([92.3, 83.1, 670, 1512])
    disc = np.array([3.8, 5.6, 48, 52])
    total = np.array([110, 123, 880, 2420])
    other = total - (cand + line + disc)  # the table's totals exceed the itemised rows by 0-98 ms
    assert np.all(other >= -1.0)
    fig, ax = plt.subplots(figsize=(8, 3.3))
    x = np.arange(4)
    ax.bar(x, cand, color=BLUE, width=0.6, label="candidate generation")
    ax.bar(x, line, bottom=cand, color=ORANGE, width=0.6, label="line search (bisection with the full model)")
    ax.bar(x, disc, bottom=cand + line, color=AQUA, width=0.6, label="discrete suggestion")
    ax.bar(x, np.maximum(other, 0), bottom=cand + line + disc, color=STONE, width=0.6, label="not itemised in Table 1")
    for i in range(4):
        ax.text(x[i], total[i] + 40, f"{total[i]:,} ms  ({100 * line[i] / total[i]:.0f}% line search)", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("time per suggestion round (ms)")
    ax.set_ylim(0, 2800)
    ax.set_title("Suggestion time grows superlinearly with model size and is dominated by the exact line search (Table 1)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    clean(ax)
    fig.tight_layout()
    save(fig, f"{SLUG}-3.png")


if __name__ == "__main__":
    print(f"F_PULL_MAX per nail = {F_PULL_MAX:.0f} N, Lambda_max = {LAMBDA_MAX:.2f} N m")
    fig1()
    fig2()
    fig3()
