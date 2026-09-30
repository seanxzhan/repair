"""Figures for liu2022-worst-case-rigidity.md.

Toy re-implementation of the paper's worst-case rigidity analysis on a 2D
bar-and-hinge portal frame (the generic version of the paper's Fig. 12(a)):

* parts are thin rectangles meshed with constant-strain triangles (plane
  stress, E = 1, nu = 0.3, unit thickness) -> block-diagonal K;
* every hinge is built with the paper's Sec. 4.1 recipe: rows of A_allow are
  the rigid motion of both parts, the hinge rotation of one part, and each
  part's own deformation of its two selected nodes; A_k = null space of
  A_allow; J_k = A_k^T S_k A_k with S_k = s * I;
* fixed ground blocks are handled by deleting their rows/columns;
* lambda_1, u_1 = smallest eigenpair of K + J (numpy.linalg.eigh).

Everything is dimensionless; only ratios between configurations matter.
"""
from _style import *  # noqa: F401,F403
import numpy as np
from scipy.linalg import null_space
from matplotlib.patches import Polygon

E_MOD, NU, THICK = 1.0, 0.3, 1.0
W = 0.08          # bar width
H_SEG = 0.1       # target element length along a bar
S_DEFAULT = 100.0  # joint stiffness (dimensionless)


# ----------------------------------------------------------------------------
# FEM pieces
# ----------------------------------------------------------------------------
def cst_stiffness(xy):
    """Plane-stress constant-strain-triangle stiffness (6x6) for nodes xy (3x2)."""
    (x1, y1), (x2, y2), (x3, y3) = xy
    a2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)  # 2*signed area
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    B = np.zeros((3, 6))
    B[0, 0::2] = b
    B[1, 1::2] = c
    B[2, 0::2] = c
    B[2, 1::2] = b
    B /= a2
    D = E_MOD / (1 - NU**2) * np.array([[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU) / 2]])
    return THICK * abs(a2) / 2 * B.T @ D @ B


class Assembly:
    def __init__(self):
        self.X = []          # node coordinates
        self.parts = []      # dict(nodes=[...], tris=[...], outline=[...])
        self.fixed = []      # fixed node ids
        self.joints = []     # (part_i, part_j, hinge point)

    def add_bar(self, p, q, seg=H_SEG):
        p, q = np.asarray(p, float), np.asarray(q, float)
        L = np.linalg.norm(q - p)
        n = max(2, int(round(L / seg)))
        e = (q - p) / L
        nrm = np.array([-e[1], e[0]])
        base = len(self.X)
        nodes = []
        for i in range(n + 1):
            c = p + e * (L * i / n)
            self.X.append(c - nrm * W / 2)
            self.X.append(c + nrm * W / 2)
            nodes += [base + 2 * i, base + 2 * i + 1]
        tris = []
        for i in range(n):
            a, b, c, d = base + 2 * i, base + 2 * i + 1, base + 2 * i + 2, base + 2 * i + 3
            tris += [(a, c, d), (a, d, b)]
        outline = [base + 2 * i for i in range(n + 1)] + [base + 2 * i + 1 for i in range(n, -1, -1)]
        self.parts.append(dict(nodes=nodes, tris=tris, outline=outline))
        return len(self.parts) - 1

    def add_ground(self, p):
        """A fixed block: two fixed nodes straddling the pin point p."""
        p = np.asarray(p, float)
        base = len(self.X)
        self.X += [p + (-W / 2, 0.0), p + (W / 2, 0.0)]
        self.fixed += [base, base + 1]
        self.parts.append(dict(nodes=[base, base + 1], tris=[], outline=[]))
        return len(self.parts) - 1

    def hinge(self, i, j, p):
        self.joints.append((i, j, np.asarray(p, float)))

    # -- the paper's joint recipe --------------------------------------------
    def joint_matrix(self, i, j, p, s):
        X = np.asarray(self.X)
        sel = []
        for part in (i, j):
            nodes = np.array(self.parts[part]["nodes"])
            d = np.linalg.norm(X[nodes] - p, axis=1)
            sel += list(nodes[np.argsort(d, kind="stable")[:2]])   # two closest nodes
        P = X[sel]                                                  # 4 x 2
        rows = []
        # (i) rigid motion of both parts: 2 translations + 1 rotation about origin
        rows.append(np.tile([1.0, 0.0], 4))
        rows.append(np.tile([0.0, 1.0], 4))
        rows.append(np.column_stack([-P[:, 1], P[:, 0]]).ravel())
        # (ii) motion allowed by the hinge: part i rotates about p, part j fixed
        h = np.zeros(8)
        h[0:4] = np.column_stack([-(P[:2, 1] - p[1]), P[:2, 0] - p[0]]).ravel()
        rows.append(h)
        # (iii) each part's own deformation = complement of its rigid motions
        for k in (0, 1):
            Pk = P[2 * k: 2 * k + 2]
            rig = np.array([np.tile([1.0, 0.0], 2), np.tile([0.0, 1.0], 2),
                            np.column_stack([-Pk[:, 1], Pk[:, 0]]).ravel()])
            for v in null_space(rig).T:
                r = np.zeros(8)
                r[4 * k: 4 * k + 4] = v
                rows.append(r)
        A_allow = np.array(rows)
        assert np.linalg.matrix_rank(A_allow) == 6
        A_k = null_space(A_allow).T                                 # 2 x 8 forbidden directions
        assert A_k.shape[0] == 2
        dofs = np.array([[2 * n, 2 * n + 1] for n in sel]).ravel()
        return dofs, s * A_k.T @ A_k

    def solve(self, s=S_DEFAULT, nmodes=2):
        X = np.asarray(self.X)
        N = len(X)
        K = np.zeros((2 * N, 2 * N))
        for part in self.parts:
            for tri in part["tris"]:
                dofs = np.array([[2 * n, 2 * n + 1] for n in tri]).ravel()
                K[np.ix_(dofs, dofs)] += cst_stiffness(X[list(tri)])
        J = np.zeros_like(K)
        for (i, j, p) in self.joints:
            dofs, Jk = self.joint_matrix(i, j, p, s)
            J[np.ix_(dofs, dofs)] += Jk
        free = np.array(sorted(set(range(2 * N)) - {2 * n + d for n in self.fixed for d in (0, 1)}))
        M = (K + J)[np.ix_(free, free)]
        lam, vec = np.linalg.eigh(M)
        U = np.zeros((2 * N, nmodes))
        U[free] = vec[:, :nmodes]
        return lam[:nmodes], U.T.reshape(nmodes, N, 2)


# ----------------------------------------------------------------------------
# The four frames of the paper's Fig. 12(a), generic coordinates
# ----------------------------------------------------------------------------
def portal(braces=()):
    """Unit portal frame, pinned to fixed ground blocks; braces = [(beam_x, col_side, col_y)]."""
    A = Assembly()
    L = A.add_bar((0, 0), (0, 1))
    R = A.add_bar((1, 0), (1, 1))
    B = A.add_bar((0, 1), (1, 1))
    gL, gR = A.add_ground((0, 0)), A.add_ground((1, 0))
    A.hinge(L, gL, (0, 0))
    A.hinge(R, gR, (1, 0))
    A.hinge(L, B, (0, 1))
    A.hinge(R, B, (1, 1))
    for (bx, side, cy) in braces:
        col = L if side == 0 else R
        br = A.add_bar((bx, 1), (side, cy))
        A.hinge(br, B, (bx, 1))
        A.hinge(br, col, (side, cy))
    return A


FRAMES = [
    ("(a1) four-bar linkage", []),
    ("(a2) one short brace", [(0.8, 1, 0.8)]),
    ("(a3) two short braces", [(0.8, 1, 0.8), (0.2, 0, 0.8)]),
    ("(a4) two long braces", [(0.6, 1, 0.4), (0.4, 0, 0.4)]),
]


def draw_frame(ax, A, u, lam, title, scale=0.08):
    X = np.asarray(A.X)
    amp = scale / np.abs(u).max()
    Xd = X + amp * u
    for part in A.parts:
        if not part["outline"]:
            continue
        ax.add_patch(Polygon(X[part["outline"]], closed=True, fc=STONE, ec=INK2, lw=0.6, zorder=2))
    mech = lam < 1e-9
    col = BAD if mech else BLUE
    for part in A.parts:
        if not part["outline"]:
            continue
        ax.add_patch(Polygon(Xd[part["outline"]], closed=True, fc="none", ec=col, lw=2, zorder=3))
    for n in A.fixed[::2]:
        p = X[n] + (W / 2, 0)
        ax.plot([p[0] - 0.12, p[0] + 0.12], [p[1] - 0.03] * 2, color=INK2, lw=1.2, zorder=1)
        ax.plot(p[0], p[1], "o", ms=4, color=INK, zorder=4)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.1, 1.2)
    ax.set_aspect("equal", adjustable="box")
    tag = "λ₁ = 0  (mechanism)" if mech else f"λ₁ = {lam:.2e}"
    ax.set_title(f"{title}\n{tag}", fontsize=9)
    clean(ax, hide_axes=True)


# ============================================================================
# Figure 1: softest mode of the four frames
# ============================================================================
fig, axes = plt.subplots(1, 4, figsize=(8, 2.9))
lams = []
for ax, (title, braces) in zip(axes, FRAMES):
    A = portal(braces)
    lam, U = A.solve()
    lams.append(lam[0])
    draw_frame(ax, A, U[0], lam[0], title)
    print(title, "lambda_1,2 =", lam)
assert lams[0] < 1e-9 and lams[1] > 1e-6 and lams[2] > lams[1] and lams[3] > lams[2]
# legend via proxy artists
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=INK2, lw=0.6, label="rest shape"),
           Line2D([], [], color=BAD, lw=2, label="softest mode u₁, zero energy (mechanism)"),
           Line2D([], [], color=BLUE, lw=2, label="softest mode u₁, costs energy (rigid)")]
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Bracing turns the zero eigenvalue positive, and larger triangles raise it further",
             fontsize=11, fontweight="semibold", y=1.0)
fig.subplots_adjust(top=0.9, bottom=0.12, wspace=0.05)
save(fig, "liu2022-worst-case-rigidity-1.png")


# ============================================================================
# Figure 2: (a) brace-candidate ranking from u1 of the mechanism (Sec. 5.2),
#           (b) the score against the lambda_1 each candidate actually gives,
#           (c) lambda_1 versus joint stiffness (joint "jitter")
# ============================================================================
A0 = portal([])
lam0, U0 = A0.solve()
u = U0[0]
X0 = np.asarray(A0.X)
grid = np.round(np.arange(0.1, 0.95, 0.1), 1)


def nearest(A, p):
    return int(np.argmin(np.linalg.norm(np.asarray(A.X) - p, axis=1)))


score = np.zeros((len(grid), len(grid)))     # rows: column height yc, cols: beam x
lam_b = np.zeros_like(score)
for a, yc in enumerate(grid):
    for b, bx in enumerate(grid):
        na = nearest(A0, np.array([bx, 1 - W / 2]))      # beam node on its lower edge
        nb = nearest(A0, np.array([1 - W / 2, yc]))      # column node on its inner edge
        d = X0[nb] - X0[na]
        score[a, b] = abs((u[nb] - u[na]) @ d) / np.linalg.norm(d)
        lam_b[a, b] = portal([(bx, 1, yc)]).solve()[0][0]
score /= score.max()
ia, ib = np.unravel_index(np.argmax(score), score.shape)
ja, jb = np.unravel_index(np.argmax(lam_b), lam_b.shape)
print("best by score: beam x=%.1f, column y=%.1f; best by lambda_1: beam x=%.1f, column y=%.1f"
      % (grid[ib], grid[ia], grid[jb], grid[ja]))

s_sweep = np.logspace(-3, 4, 29)
lam_a2 = [portal(FRAMES[1][1]).solve(s=s)[0][0] for s in s_sweep]
lam_a4 = [portal(FRAMES[3][1]).solve(s=s)[0][0] for s in s_sweep]

fig, axes = plt.subplots(1, 3, figsize=(8, 3.4), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
ax = axes[0]
im = ax.imshow(score, origin="lower", cmap="Blues", vmin=0, vmax=1,
               extent=[0.05, 0.95, 0.05, 0.95], aspect="equal")
ax.plot(grid[ib], grid[ia], "o", ms=9, mfc="none", mec=ORANGE, mew=2, label="top-ranked pair")
ax.set_xlabel("beam end of brace, x (frame widths)", fontsize=9)
ax.set_ylabel("column end of brace, y (heights)", fontsize=9)
ax.set_title("(a) Brace candidates scored\nfrom u₁ of the mechanism", fontsize=10)
ax.set_xticks(grid[::2])
ax.set_yticks(grid[::2])
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("score (normalised)")
cb.outline.set_visible(False)
ax.legend(loc="lower left", fontsize=8)

ax = axes[1]
ax.plot(score.ravel(), lam_b.ravel() * 1e3, "o", ms=4, color=BLUE, mec="white", mew=0.5, alpha=0.9)
ax.plot(score[ia, ib], lam_b[ia, ib] * 1e3, "o", ms=9, mfc="none", mec=ORANGE, mew=2)
ax.set_xlabel("candidate score (normalised)")
ax.set_ylabel("λ₁ with the brace added  (×10⁻³)", fontsize=9)
ax.set_title("(b) Score tracks the λ₁\neach brace actually gives", fontsize=10)
ax.set_xlim(0, 1.05)
ax.set_ylim(0, None)
clean(ax)

ax = axes[2]
ax.loglog(s_sweep, lam_a2, color=BLUE, label="(a2) one short brace")
ax.loglog(s_sweep, lam_a4, color=ORANGE, label="(a4) two long braces")
ax.axvline(S_DEFAULT, color=MUTED, lw=1, ls=":")
ax.text(S_DEFAULT * 1.6, max(lam_a4) * 0.35, "s used\nin Fig. 1", color=INK2, fontsize=8, ha="left")
ax.set_xlabel("joint stiffness s (dimensionless)")
ax.set_ylabel("λ₁ (dimensionless)")
ax.set_title("(c) Looser joints lower λ₁\nsmoothly, stiff ones saturate", fontsize=10)
ax.set_ylim(3e-8, 2e-4)
ax.legend(loc="lower right", fontsize=8)
clean(ax)
fig.tight_layout(w_pad=1.5)
save(fig, "liu2022-worst-case-rigidity-2.png")


# ============================================================================
# Figure 3: the paper's own numbers (Fig. 12 values; Table 1 before/after)
# ============================================================================
fig12a = [("(a1)", 0.0), ("(a2)", 6.17e-6), ("(a3)", 1.35e-5), ("(a4)", 4.84e-5)]
fig12b = {  # overlap: (with jitter, without jitter)
    "short\noverlap": (1.31e-5, 3.38e-5),
    "long\noverlap": (2.96e-5, 4.20e-5),
}
table1 = [  # name, before, after  (x 1e-7), Table 1
    ("ROLLING CHASSIS", 8.656, 26.91),
    ("BRIDGE FRAME", 63000, 113513),
    ("Fig. 9 linkage", 16.2, 44.3),
    ("BOOMERANG", 249.62, 1097.59),
    ("BUNNY", 80.95, 216.31),
    ("DRYING RACK", 5.11, 64.6),
    ("TECHNIC BIRD", 65.31, 97.87),
]

fig, axes = plt.subplots(1, 3, figsize=(8, 3.0), gridspec_kw=dict(width_ratios=[1, 1, 1.5]))
ax = axes[0]
names = [n for n, _ in fig12a]
vals = np.array([v for _, v in fig12a]) * 1e5
bars = ax.bar(names, vals, color=BLUE, width=0.6, zorder=3)
for b_, v in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width() / 2, v + 0.1, "0" if v == 0 else f"{v:.2f}",
            ha="center", va="bottom", fontsize=8, color=INK2)
ax.set_ylabel("λ₁  (×10⁻⁵)")
ax.set_title("(a) Fig. 12(a): each brace\nadds rigidity", fontsize=10)
ax.set_ylim(0, 5.6)
clean(ax)

ax = axes[1]
x = np.arange(2)
wdt = 0.36
with_j = np.array([v[0] for v in fig12b.values()]) * 1e5
wo_j = np.array([v[1] for v in fig12b.values()]) * 1e5
ax.bar(x - wdt / 2, with_j, wdt, color=BLUE, label="with joint jitter", zorder=3)
ax.bar(x + wdt / 2, wo_j, wdt, color=ORANGE, label="without joint jitter", zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(list(fig12b.keys()))
ax.set_ylabel("λ₁  (×10⁻⁵)")
ax.set_title("(b) Fig. 12(b): collinear bars,\noverlap and joint jitter", fontsize=10)
ax.set_ylim(0, 5.6)
ax.legend(loc="upper left", fontsize=8)
clean(ax)

ax = axes[2]
y = np.arange(len(table1))[::-1]
for yi, (name, b0, b1) in zip(y, table1):
    ax.plot([b0, b1], [yi, yi], color=GRID, lw=2, zorder=2)
    ax.annotate("", xy=(b1, yi), xytext=(b0, yi),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1, shrinkA=4, shrinkB=4))
ax.plot([t[1] for t in table1], y, "o", color=BLUE, ms=6, label="before optimisation", zorder=3)
ax.plot([t[2] for t in table1], y, "o", color=ORANGE, ms=6, label="after optimisation", zorder=3)
for yi, (name, b0, b1) in zip(y, table1):
    ax.text(b1 * 1.5, yi, f"×{b1 / b0:.1f}", va="center", fontsize=8, color=INK2)
ax.set_xscale("log")
ax.set_xlim(2, 1e6)
ax.set_yticks(y)
ax.set_yticklabels([t[0].title() if t[0].isupper() else t[0] for t in table1], fontsize=8)
ax.set_xlabel("worst-case rigidity λ₁  (×10⁻⁷, log scale)")
ax.set_title("(c) Table 1: geometry optimisation\nraises λ₁ by 1.5–13×", fontsize=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=8)
clean(ax)
fig.tight_layout(w_pad=1.2)
save(fig, "liu2022-worst-case-rigidity-3.png")
