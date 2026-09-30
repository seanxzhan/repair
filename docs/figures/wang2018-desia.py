"""Figures for docs/wang2018-desia.md (DESIA, Wang, Song, Pauly 2018).

Everything below is computed: contacts are extracted from the geometry, the
directional blocking graphs (DBGs) are built from the contact normals, strongly
connected components come from Tarjan-style SCC (scipy), and the movability test
is the paper's LP (Eq. 2) solved with scipy.optimize.linprog.
Run from docs/figures/:  python wang2018-desia.py
"""
from _style import *  # noqa: F401,F403

import numpy as np
from matplotlib.patches import Polygon, Circle, FancyArrowPatch
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# --------------------------------------------------------------------------- toolkit
def dbg(cons, d, N):
    """DESIA directional blocking graph for translation d: edge i->j iff Pj blocks Pi.
    cons: (i, j, n) with n the contact normal pointing from Pi into Pj."""
    A = np.zeros((N, N), dtype=int)
    for i, j, n in cons:
        s = float(np.dot(d, n))
        if s > 1e-9:
            A[i, j] = 1      # Pi moving along d closes the gap -> blocked by Pj
        elif s < -1e-9:
            A[j, i] = 1      # Pj moving along d closes the gap -> blocked by Pi
    return A


def sccs(A):
    return connected_components(csr_matrix(A), directed=True, connection="strong")


def desia_lp(cons, N, fixed):
    """Paper Eq. 2:  max sum t_ij  s.t. (v_j - v_i).n_ij >= t_ij, 0 <= t_ij <= 1, v_fixed = 0."""
    M, nv = len(cons), 2 * N
    c = np.zeros(nv + M); c[nv:] = -1.0
    A = np.zeros((M, nv + M)); b = np.zeros(M)
    for k, (i, j, n) in enumerate(cons):
        A[k, 2 * j:2 * j + 2] -= n
        A[k, 2 * i:2 * i + 2] += n
        A[k, nv + k] = 1.0
    bounds = [(0, 0) if (p in fixed) else (None, None) for p in range(N) for _ in range(2)]
    bounds += [(0, 1)] * M
    r = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    return -r.fun, r.x[:nv].reshape(N, 2), r.x[nv:]


def pixel_contacts(grid):
    """Unit contacts between 4-neighbouring pixels of different parts; also returns the segment."""
    H, W = grid.shape
    out = []
    for y in range(H):
        for x in range(W):
            a = grid[y, x]
            if x + 1 < W and grid[y, x + 1] != a:
                out.append((a, grid[y, x + 1], np.array([1.0, 0.0]), (x + 1, y), (x + 1, y + 1)))
            if y + 1 < H and grid[y + 1, x] != a:
                out.append((a, grid[y + 1, x], np.array([0.0, 1.0]), (x, y + 1), (x + 1, y + 1)))
    return out


def poly_contacts(polys, tol=1e-7):
    """Contacts between CCW polygons: (i, j, n_ij, p, q) for collinear, overlapping, opposite edges."""
    out = []

    def edges(P):
        P = np.asarray(P, float)
        return [(P[k], P[(k + 1) % len(P)]) for k in range(len(P))]

    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            for a0, a1 in edges(polys[i]):
                ta = a1 - a0; La = np.linalg.norm(ta); ta = ta / La
                n = np.array([ta[1], -ta[0]])                   # outward normal of Pi
                for b0, b1 in edges(polys[j]):
                    tb = b1 - b0; tb = tb / np.linalg.norm(tb)
                    if abs(np.dot(ta, tb) + 1) > 1e-6 or abs(np.dot(b0 - a0, n)) > tol:
                        continue
                    s0, s1 = np.dot(b0 - a0, ta), np.dot(b1 - a0, ta)
                    lo, hi = max(0.0, min(s0, s1)), min(La, max(s0, s1))
                    if hi - lo > tol:
                        out.append((i, j, n.copy(), a0 + lo * ta, a0 + hi * ta))
    return out


def base_directions(cons):
    """Diameters parallel to every contact line; endpoints on the half circle [0, pi)."""
    angs = sorted({round(np.arctan2(-c[2][0], c[2][1]) % np.pi, 9) for c in cons})
    return [np.array([np.cos(a), np.sin(a)]) for a in angs], angs


def draw_graph(ax, A, pos, labels, key=None, title=""):
    """Directed graph with curved arrows so that opposite edges do not overlap."""
    N = len(pos)
    for i in range(N):
        for j in range(N):
            if A[i, j]:
                ax.add_patch(FancyArrowPatch(pos[i], pos[j], arrowstyle="-|>", mutation_scale=11,
                                             color=INK2, lw=1.2, connectionstyle="arc3,rad=0.22",
                                             shrinkA=11, shrinkB=11, zorder=2))
    for i in range(N):
        ec = ORANGE if i == key else INK2
        ax.add_patch(Circle(pos[i], 0.17, fc=SURFACE, ec=ec, lw=2.0 if i == key else 1.2, zorder=3))
        ax.text(*pos[i], labels[i], ha="center", va="center", fontsize=9, color=INK, zorder=4)
    ax.set_xlim(-0.7, 1.7); ax.set_ylim(-0.6, 1.7)
    ax.set_title(title, fontsize=9, color=INK, fontweight="normal")
    clean(ax, equal=True, hide_axes=True)


FILLS = [WOOD2, WOOD, STONE, "#f2f0ea"]

# --------------------------------------------------------------------------- example 1: 4-part pixel puzzle
rows = """4 4 1 1 1 4 4
4 2 1 1 1 3 4
4 2 2 2 3 3 4
4 2 3 3 3 3 4
4 2 2 2 2 3 4
4 4 4 4 4 4 4"""
grid = np.array([[int(c) - 1 for c in r.split()] for r in rows.splitlines()])[::-1]  # row 0 = y 0
N = 4
labels = [f"P{k + 1}" for k in range(N)]
pc = pixel_contacts(grid)
cons = [(i, j, n) for i, j, n, p, q in pc]
dirs = {"+x": np.array([1.0, 0.0]), "+y": np.array([0.0, 1.0])}
G = {name: dbg(cons, d, N) for name, d in dirs.items()}
verdict = {}
for name, A in G.items():
    k, lab = sccs(A)
    verdict[name] = (k, lab)
    print(name, "SCCs:", k, lab)

# LP, round 1 (one part at a time) and round 2 (key + reference fixed)
round1 = []
for p in range(N):
    opt, V, t = desia_lp(cons, N, {q for q in range(N) if q != p})
    round1.append(opt)
    print(f"round 1, only {labels[p]} free: max sum t = {opt:.2f}, v = {V[p]}")
key = int(np.argmax(round1))
opt2, V2, t2 = desia_lp(cons, N, {key, 3})
print("round 2 (key + P4 fixed):", opt2)
opt_key, V_key, t_key = desia_lp(cons, N, {q for q in range(N) if q != key})


def draw_pixels(ax, grid, t=None, cons_geom=None):
    H, W = grid.shape
    for y in range(H):
        for x in range(W):
            ax.add_patch(Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)], closed=True,
                                 fc=FILLS[grid[y, x]], ec="none", zorder=1))
    # part outlines: draw every pixel edge that separates two parts or borders the outside
    for y in range(H):
        for x in range(W):
            a = grid[y, x]
            for dx, dy, seg in ((1, 0, ((x + 1, y), (x + 1, y + 1))), (-1, 0, ((x, y), (x, y + 1))),
                                (0, 1, ((x, y + 1), (x + 1, y + 1))), (0, -1, ((x, y), (x + 1, y)))):
                xx, yy = x + dx, y + dy
                if not (0 <= xx < W and 0 <= yy < H) or grid[yy, xx] != a:
                    ax.plot(*zip(*seg), color=INK2, lw=1.0, zorder=2, solid_capstyle="round")
    if t is not None:
        for (i, j, n, p, q), tk in zip(cons_geom, t):
            if tk > 1e-6:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=ORANGE, lw=3.2, zorder=3, solid_capstyle="butt")
    # part labels at centroids
    for k in range(N):
        ys, xs = np.where(grid == k)
        cx, cy = xs.mean() + 0.5, ys.mean() + 0.5
        if k == 3:
            cx, cy = 3.5, 0.5
        ax.text(cx, cy, labels[k], ha="center", va="center", fontsize=9, color=INK, zorder=4)
    ax.set_xlim(-0.3, W + 0.3); ax.set_ylim(-0.3, H + 0.9)
    clean(ax, equal=True, hide_axes=True)


# ---- Figure 1: geometry + two base DBGs
fig, axs = plt.subplots(1, 3, figsize=(8, 2.9), gridspec_kw={"width_ratios": [1.5, 1, 1]})
ax = axs[0]
draw_pixels(ax, grid)
ax.annotate("", xy=(3.5, 6.85), xytext=(3.5, 6.1), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
ax.text(3.75, 6.45, "+y", color=INK, fontsize=9, va="center")
ax.set_title("(a) 4-part 2D assembly\nP4 is a lipped frame", fontsize=9, fontweight="normal")
pos = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
for ax, name in zip(axs[1:], ["+x", "+y"]):
    k, lab = verdict[name]
    kk = None
    if k == 2:
        sizes = np.bincount(lab)
        kk = int(np.where(lab == np.argmin(sizes))[0][0])
    sub = "1 SCC: nothing moves along ±x" if k == 1 else f"2 SCCs: only {labels[kk]} splits off"
    tag = "(b)" if name == "+x" else "(c)"
    draw_graph(ax, G[name], pos, labels, key=kk, title=f"{tag} G({name})\n{sub}")
fig.suptitle("Two base blocking graphs decide interlocking: P1 is the only key, free along +y",
             fontsize=11, fontweight="semibold", color=INK, y=1.13)
save(fig, "wang2018-desia-1.png")

# ---- Figure 2: the LP on the same puzzle
fig, axs = plt.subplots(1, 2, figsize=(8, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axs[0]
draw_pixels(ax, grid, t=t_key, cons_geom=pc)
ax.plot([], [], color=ORANGE, lw=3.2, label="contact that opens (t = 1)")
ax.plot([], [], color=INK2, lw=1.0, label="contact that stays closed (t = 0)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=1, fontsize=8.5)
ax.set_title("(a) Round 1, only P1 free: v = (0, 1)\nopens the 3 floor contacts", fontsize=9, fontweight="normal")
ax = axs[1]
names = [f"only {l} free" for l in labels] + ["P1 + P4 fixed\n(round 2)"]
vals = round1 + [opt2]
cols = [BLUE] * N + [ORANGE]
ax.barh(np.arange(len(vals))[::-1], vals, color=cols, height=0.6, zorder=2)
ax.set_yticks(np.arange(len(vals))[::-1]); ax.set_yticklabels(names, fontsize=8.5)
for k, v in enumerate(vals):
    ax.text(v + 0.06, len(vals) - 1 - k, f"{(v if v > 1e-9 else 0):.0f}", va="center", fontsize=8.5, color=INK2)
ax.set_xlim(0, 3.8)
ax.set_xlabel("LP optimum Σ t_ij (contacts that can open)")
ax.set_title("(b) Exactly one movable part,\nthen deadlock", fontsize=9, fontweight="normal")
clean(ax)
fig.suptitle("The LP (Eq. 2) gives the same verdict: only P1 can move; with P1 held, nothing can",
             fontsize=11, fontweight="semibold", color=INK, y=1.13)
save(fig, "wang2018-desia-2.png")

# --------------------------------------------------------------------------- example 2: paper's Fig. 4 escape
L, R, T = np.array([-1.0, 0]), np.array([1.0, 0]), np.array([0, 1.6])
A_, B_, V_ = np.array([-0.5, 0.8]), np.array([0.5, 0.8]), np.array([0.0, 0.4])


def tooth(p, q, s, w, h):
    """Rectangular tooth on segment p->q centred at fraction s, width w, height h to the left of p->q."""
    t = (q - p) / np.linalg.norm(q - p); n = np.array([-t[1], t[0]])
    c = p + s * (q - p)
    return [c - w / 2 * t, c - w / 2 * t + h * n, c + w / 2 * t + h * n, c + w / 2 * t]


tl = tooth(A_, V_, 0.5, 0.12, 0.10)                       # P1's tooth into P3 (left slanted face)
tr = tooth(V_, B_, 0.5, 0.12, 0.10)                       # P2's tooth into P3 (right slanted face)
tv = tooth(np.array([0, 0.0]), V_, 0.45, 0.10, -0.10)     # P1's tooth into P2 (vertical face)
P1 = [L, np.array([0, 0.0]), *tv, V_, *tl[::-1], A_]
P2 = [np.array([0, 0.0]), R, B_, *tr[::-1], V_, *tv[::-1]]
P3 = [T, A_, *tl, V_, *tr, B_]
polys = [P1, P2, P3]
pcs = poly_contacts(polys)
cons3 = [(i, j, n) for i, j, n, p, q in pcs]
bdirs, bangs = base_directions(cons3)
scc_counts = [sccs(dbg(cons3, d, 3))[0] for d in bdirs]
print("base directions (deg):", np.round(np.degrees(bangs), 1), "SCC counts:", scc_counts)
single = [desia_lp(cons3, 3, {q for q in range(3) if q != p})[0] for p in range(3)]
opt3, V3, t3 = desia_lp(cons3, 3, {2})
print("each part alone:", single, " | P3 fixed, P1+P2 free:", opt3, V3[:2])

fig, axs = plt.subplots(1, 3, figsize=(8, 3.1), gridspec_kw={"width_ratios": [1.2, 1.0, 1.2]})
# (a) geometry with contact normals and the base-direction rose
ax = axs[0]
for k, P in enumerate(polys):
    ax.add_patch(Polygon(np.array(P), closed=True, fc=FILLS[k], ec=INK2, lw=1.0, zorder=1))
for i, j, n, p, q in pcs:
    m = (p + q) / 2
    ax.annotate("", xy=m + 0.12 * n, xytext=m, arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8,
                                                                 mutation_scale=7), zorder=3)
for k, (cx, cy) in enumerate([(-0.6, 0.3), (0.6, 0.3), (0.0, 0.98)]):
    ax.text(cx, cy, labels[k] + (" (fixed)" if k == 2 else ""), ha="center", va="center", fontsize=8, color=INK)
ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.15, 1.75)
ax.set_title("(a) Non-orthogonal contacts\n(after the paper's Fig. 4)", fontsize=9, fontweight="normal")
clean(ax, equal=True, hide_axes=True)
# (b) the six base directions and their SCC counts
ax = axs[1]
th = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(th), np.sin(th), color=GRID, lw=1.5, zorder=1)
for a, k in zip(bangs, scc_counts):
    d = np.array([np.cos(a), np.sin(a)])
    ax.plot([-d[0], d[0]], [-d[1], d[1]], color=INK2, lw=1.0, zorder=2)
    ax.plot([d[0]], [d[1]], "o", ms=5, color=BLUE, zorder=3)
    ax.text(1.22 * d[0], 1.22 * d[1], f"{np.degrees(a):.0f}°", ha="center", va="center", fontsize=8, color=INK2)
ax.text(0, -1.45, f"all {len(bangs)} base DBGs: 1 SCC\n→ graph test says deadlocked", ha="center",
        va="top", fontsize=8.5, color=INK)
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-2.05, 1.5)
ax.set_title("(b) Base directions and\nSCC count of each DBG", fontsize=9, fontweight="normal")
clean(ax, equal=True, hide_axes=True)
# (c) LP motion
ax = axs[2]
for k, P in enumerate(polys):
    ax.add_patch(Polygon(np.array(P), closed=True, fc=FILLS[k], ec=INK2, lw=1.0, zorder=1))
for (i, j, n, p, q), tk in zip(pcs, t3):
    if tk > 1e-6:
        ax.plot([p[0], q[0]], [p[1], q[1]], color=ORANGE, lw=3.0, zorder=3, solid_capstyle="butt")
cents = [(-0.55, 0.3), (0.55, 0.3)]
for k in range(2):
    v = V3[k] / np.linalg.norm(V3[k]) * 0.45
    c = np.array(cents[k])
    ax.annotate("", xy=c + v, xytext=c, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8, mutation_scale=12), zorder=4)
ax.text(0, 0.98, "P3 (fixed)", ha="center", va="center", fontsize=8, color=INK)
ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.75, 1.75)
ax.text(0, -0.3, f"P1 alone: Σt = 0,  P2 alone: Σt = 0\nboth free: Σt = {opt3:.0f}, v_P1 = ({V3[0][0]:+.2f}, {V3[0][1]:+.2f}),\n"
        f"v_P2 = ({V3[1][0]:+.2f}, {V3[1][1]:+.2f})  →  movable",
        ha="center", va="top", fontsize=8, color=INK)
ax.set_title("(c) LP with P3 fixed finds the\ntwo-direction escape", fontsize=9, fontweight="normal")
clean(ax, equal=True, hide_axes=True)
fig.suptitle("Why the graphs are not enough: every base DBG is strongly connected, yet the LP finds a motion",
             fontsize=11, fontweight="semibold", color=INK, y=1.13)
save(fig, "wang2018-desia-3.png")

# --------------------------------------------------------------------------- Table 1 timings
# (label, N parts, M base directions, generation time in seconds), read from Table 1 of the paper.
table1 = [
    ("Voxelized", "Cartoon Dog", 14, 3, 1.06 * 3600), ("Voxelized", "4×4×4 Cube", 9, 3, 1.13 * 3600),
    ("Voxelized", "Bunny", 80, 3, 0.74 * 3600), ("Voxelized", "35³ Cube", 1500, 3, 3.33 * 3600),
    ("Plate", "Lizard", 33, 45, 2.73), ("Plate", "Bookshelf", 12, 3, 0.30), ("Plate", "Cabinet", 7, 5, 0.09),
    ("Plate", "Table", 6, 3, 0.02), ("Plate", "Chair", 8, 3, 0.22),
    ("Frame", "Flower", 23, 3, 4.10), ("Frame", "Frame Cube", 12, 3, 0.53), ("Frame", "Frame Chair", 11, 3, 15.66),
    ("Frame", "Scaffold", 92, 3, 26.41),
]
cube = [(7, 0.3), (8, 12.0), (9, 1.13 * 3600)]   # Section 5.4: 4x4x4 Cube with 7, 8, 9 parts

fig, axs = plt.subplots(1, 2, figsize=(8, 3.6), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.45})
ax = axs[0]
ypos = np.arange(len(table1))[::-1]
for y, (grp, name, n, m, t) in zip(ypos, table1):
    col = {"Voxelized": BLUE, "Plate": ORANGE, "Frame": AQUA}[grp]
    ax.barh(y, t, color=col, height=0.62, zorder=2)
    ax.text(t * 1.35, y, f"N = {n}" + (f", M = {m}" if m != 3 else ""), va="center", fontsize=7.5, color=INK2)
ax.set_yticks(ypos); ax.set_yticklabels([r[1] for r in table1], fontsize=8.5)
ax.set_xscale("log"); ax.set_xlim(0.01, 3e5)
ax.set_xticks([0.01, 1, 100, 1e4]); ax.set_xticklabels(["0.01 s", "1 s", "100 s", "10⁴ s"])
ax.set_xlabel("time to generate the interlocking design (s, log scale)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=g) for g, c in [("Voxelized", BLUE), ("Plate", ORANGE), ("Frame", AQUA)]],
          loc="lower right", fontsize=8.5, title="structure type", title_fontsize=8.5)
ax.set_title("(a) Table 1: time to generate each design", fontsize=9, fontweight="normal")
clean(ax)
ax = axs[1]
xs = [c[0] for c in cube]; ys = [c[1] for c in cube]
ax.plot(xs, ys, "o-", color=BLUE, lw=2, ms=6, zorder=3)
for x, y, lab in zip(xs, ys, ["0.3 s", "12 s", "1.13 h"]):
    ax.text(x + 0.1, y * 0.8, lab, ha="left", va="top", fontsize=8.5, color=INK2)
ax.set_yscale("log"); ax.set_ylim(0.1, 3e4)
ax.set_xticks([7, 8, 9]); ax.set_xlim(6.5, 9.5)
ax.set_xlabel("number of parts N  (4×4×4 Cube)")
ax.set_ylabel("generation time (s, log scale)")
ax.set_title("(b) 4×4×4 Cube: ×40, then ×340\nper extra part (Sec. 5.4)", fontsize=9, fontweight="normal")
clean(ax)
fig.suptitle("The interlocking test is milliseconds; the design search is what grows sharply with N",
             fontsize=11, fontweight="semibold", color=INK, y=1.13)
save(fig, "wang2018-desia-4.png")
