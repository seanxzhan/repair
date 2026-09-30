"""Figures for song2012-recursive-interlocking-puzzles.md.

Figure 1: a 4-piece 3D voxel puzzle we designed by the paper's rules, and the
          axis-aligned blocking test run on every subset of pieces.
Figure 2: the recursive disassembly, re-running the test after each removal.
Figure 3: Table 1 timings.
"""
from itertools import combinations

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401

from _style import *  # noqa: F401,F403

# ---------------------------------------------------------------- the toy puzzle
# Grid 6 x 3 x 4 (x, y, z), z up. y = 0 and y = 2 are solid side walls of R.
# Middle slice y = 1 (rows top to bottom are z = 3 .. 0, columns x = 0 .. 5):
#   z=3:  R R . . K R      (5,1,3) is the lip that stops K sideways
#   z=2:  R Q Q Q K .      (5,1,2) is empty: Q's exit slot
#   z=1:  R R P P R R
#   z=0:  R R R R R R
NX, NY, NZ = 6, 3, 4
K = {(4, 1, 2), (4, 1, 3)}
Q = {(1, 1, 2), (2, 1, 2), (3, 1, 2)}
P = {(2, 1, 1), (3, 1, 1)}
EMPTY = {(5, 1, 2), (2, 1, 3), (3, 1, 3)}
ALL = {(x, y, z) for x in range(NX) for y in range(NY) for z in range(NZ)}
R = ALL - K - Q - P - EMPTY
PIECES = {"K": K, "Q": Q, "P3": P, "R": R}
FILL = {"K": BLUE, "Q": WOOD, "P3": WOOD2, "R": STONE}
DIRS = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0),
        "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


def occupancy(pieces):
    occ = {}
    for name, vox in pieces.items():
        for v in vox:
            occ[v] = name
    return occ


def can_step(occ, names, d):
    """Blocking test: the union of `names` can translate one voxel along d
    iff no voxel of the set lands on a voxel of another piece."""
    for v in (v for n in names for v in PIECES[n]):
        w = (v[0] + d[0], v[1] + d[1], v[2] + d[2])
        if w in occ and occ[w] not in names:
            return False
    return True


def removable(pieces, names, d):
    """Slide the set along d step by step until it has left the bounding box,
    or until it collides.  Returns True if it leaves."""
    others = {v for n, vox in pieces.items() if n not in names for v in vox}
    moving = {v for n in names for v in pieces[n]}
    for _ in range(max(NX, NY, NZ) + 1):
        moving = {(v[0] + d[0], v[1] + d[1], v[2] + d[2]) for v in moving}
        if moving & others:
            return False
        inside = [v for v in moving if 0 <= v[0] < NX and 0 <= v[1] < NY and 0 <= v[2] < NZ]
        if not inside:
            return True
    return False


def mobility(pieces):
    """For every proper non-empty subset of the pieces present, which of the
    six directions is free (one step). Also checks that a free step means
    the subset is fully removable, which the paper's model assumes."""
    occ = occupancy(pieces)
    names = list(pieces)
    out = {}
    for r in range(1, len(names)):
        for sub in combinations(names, r):
            free = [dn for dn, d in DIRS.items() if can_step(occ, set(sub), d)]
            for dn in free:
                assert removable(pieces, set(sub), DIRS[dn]), (sub, dn)
            out[sub] = free
    return out


# ------------------------------------------------------------------ drawing
def draw_slice(ax, pieces, y=1, title=None, arrows=(), dim=()):
    """Draw slice y of the voxel grid as a 2D x-z map."""
    for name, vox in pieces.items():
        for (x, yy, z) in vox:
            if yy != y:
                continue
            alpha = 0.35 if name in dim else 1.0
            ax.add_patch(plt.Rectangle((x, z), 1, 1, facecolor=FILL[name], alpha=alpha,
                                       edgecolor=INK2, linewidth=0.8, zorder=2))
    for name, vox in pieces.items():
        sl = [(x, z) for (x, yy, z) in vox if yy == y]
        if sl:
            cx = np.mean([x for x, _ in sl]) + 0.5
            cz = np.mean([z for _, z in sl]) + 0.5
            if name == "R":
                cx, cz = 0.5, 0.5
            ax.text(cx, cz, name, ha="center", va="center", fontsize=9,
                    color=INK, zorder=4, fontweight="semibold")
    for (x, z, dx, dz) in arrows:
        ax.annotate("", xy=(x + dx, z + dz), xytext=(x, z),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=2, mutation_scale=14),
                    zorder=5)
    ax.set_xlim(-0.2, NX + 0.2)
    ax.set_ylim(-0.2, NZ + 0.2)
    ax.set_xticks(range(NX + 1))
    ax.set_yticks(range(NZ + 1))
    ax.set_xlabel("x (voxels)")
    ax.set_ylabel("z (voxels)")
    ax.set_aspect("equal")
    ax.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=10)


def draw_cutaway(ax, pieces):
    """3D voxel render with the front wall (y = 0) removed so the reader can
    see the middle slice."""
    filled = np.zeros((NX, NY, NZ), dtype=bool)
    colors = np.empty((NX, NY, NZ), dtype=object)
    for name, vox in pieces.items():
        for (x, y, z) in vox:
            if y == 0:
                continue  # cut away
            filled[x, y, z] = True
            colors[x, y, z] = FILL[name]
    ax.voxels(filled, facecolors=colors, edgecolors=INK2, linewidth=0.5)
    ax.set_xlim(0, NX)
    ax.set_ylim(0, NY)
    ax.set_zlim(0, NZ)
    ax.set_box_aspect((NX, NY, NZ))
    ax.view_init(elev=22, azim=-58)
    ax.set_xlabel("x", labelpad=-10)
    ax.set_ylabel("y", labelpad=-10)
    ax.set_zlabel("")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(GRID); ax.yaxis.pane.set_edgecolor(GRID); ax.zaxis.pane.set_edgecolor(GRID)
    ax.grid(False)


def draw_matrix(ax, mob, title):
    subs = list(mob)
    labels = ["{" + ", ".join(s) + "}" if len(s) > 1 else s[0] for s in subs]
    dnames = list(DIRS)
    for i, sub in enumerate(subs):
        for j, dn in enumerate(dnames):
            free = dn in mob[sub]
            ax.add_patch(plt.Rectangle((j, len(subs) - 1 - i), 1, 1,
                                       facecolor=GOOD if free else SURFACE,
                                       edgecolor=GRID, linewidth=0.8))
            ax.text(j + 0.5, len(subs) - 1 - i + 0.5, "free" if free else "–",
                    ha="center", va="center", fontsize=7.5,
                    color=SURFACE if free else MUTED, fontweight="semibold" if free else "normal")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, len(subs))
    ax.set_xticks(np.arange(6) + 0.5)
    ax.set_xticklabels(dnames, fontsize=8.5)
    ax.set_yticks(np.arange(len(subs)) + 0.5)
    ax.set_yticklabels(labels[::-1], fontsize=8.5)
    ax.set_xlabel("translation direction")
    ax.set_ylabel("subset of pieces (R = the rest)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=10)


# ------------------------------------------------------------------ figure 1
mob0 = mobility(PIECES)
# report to stdout for the caption
for sub, free in mob0.items():
    print(sub, free)
free_sets = {s: f for s, f in mob0.items() if f}
assert set(free_sets) == {("K",), ("Q", "P3", "R")}, free_sets
assert free_sets[("K",)] == ["+z"]

fig = plt.figure(figsize=(8, 3.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.05], wspace=0.55,
                      left=0.02, right=0.98, top=0.8, bottom=0.14)
ax3 = fig.add_subplot(gs[0], projection="3d")
draw_cutaway(ax3, PIECES)
ax3.set_box_aspect((NX, NY, NZ), zoom=1.25)
ax3.set_title("6×3×4 puzzle, front wall (y = 0) cut away", fontsize=10, y=1.02)
ax_s = fig.add_subplot(gs[1])
draw_slice(ax_s, PIECES, title="middle slice y = 1")
ax_m = fig.add_subplot(gs[2])
# only subsets not containing R; the rest follow by Lemma 2 (relativity)
mob_noR = {s: f for s, f in mob0.items() if "R" not in s}
draw_matrix(ax_m, mob_noR, "blocking test on every subset")
ax_m.set_ylabel("")
fig.suptitle("Only the key K is free, and only along +z: every other subset is blocked "
             "on all six axes", fontsize=11, fontweight="semibold", y=0.97)
save(fig, "song2012-recursive-interlocking-puzzles-1.png")

# ------------------------------------------------------------------ figure 2
stages = []
pieces = dict(PIECES)
order = []
while len(pieces) > 1:
    mob = mobility(pieces)
    singles = {s[0]: f for s, f in mob.items() if len(s) == 1 and f and s[0] != "R"}
    assert len(singles) == 1, singles  # exactly one mobile piece (besides R's relative motion)
    name, free = next(iter(singles.items()))
    assert len(free) == 1, free
    stages.append((dict(pieces), name, free[0], mob))
    order.append((name, free[0]))
    del pieces[name]
print("disassembly order:", order)

fig, axes = plt.subplots(1, 3, figsize=(8, 3.5))
for step, (ax, (pcs, name, dn, mob)) in enumerate(zip(axes, stages), start=1):
    d = DIRS[dn]
    vox = [(x, z) for (x, y, z) in pcs[name] if y == 1]
    cx = np.mean([x for x, _ in vox]) + 0.5
    cz = max(z for _, z in vox) + 1 if d[2] > 0 else np.mean([z for _, z in vox]) + 0.5
    if d[0] > 0:
        cx = max(x for x, _ in vox) + 1
    arrows = [(cx, cz, 1.1 * d[0], 1.1 * d[2])]
    dim = [n for n in pcs if n != name]
    n_blocked = sum(1 for s, f in mob.items() if not f)
    others = [s for s, f in mob.items() if f and s != (name,)]
    draw_slice(ax, pcs, arrows=arrows, dim=dim,
               title=f"step {step}: {name} exits {dn}")
    ax.set_ylim(-0.2, NZ + 1.4)
    ax.text(0.5 * NX, -2.3, f"{n_blocked} of {len(mob)} subsets blocked;\n"
            f"free: {name} and its complement", ha="center", va="top", fontsize=8, color=INK2)
fig.suptitle("Re-running the test after each removal: one piece is free at a time, "
             "and successive exits are perpendicular", fontsize=10.5, fontweight="semibold")
fig.subplots_adjust(top=0.84, bottom=0.36, wspace=0.35)
save(fig, "song2012-recursive-interlocking-puzzles-2.png")

# ------------------------------------------------------------------ figure 3
# Table 1 of the paper (minutes), read from the rendered page image.
# each point: (K, minutes, label offset in points, horizontal alignment)
L, R_, C = "left", "right", "center"
cube = {"5³ (N=125)": [(10, 0.70, (7, -4), L)],
        "15³ (N=3,375)": [(10, 0.08, (7, -11), L), (100, 3.03, (7, -11), L)],
        "25³ (N=15,625)": [(10, 8.26, (0, 7), C), (100, 16.17, (0, 7), C), (500, 110.70, (-7, 4), R_)]}
bunny = {"10×9×7": [(10, 0.03, (7, -4), L)],
         "20×19×15": [(10, 3.02, (7, -11), L), (75, 319.85, (-7, 4), R_)],
         "30×29×23": [(10, 6.31, (0, 7), C), (75, 78.71, (7, -11), L), (150, 534.43, (-7, 4), R_)]}
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
for ax, data, ttl in zip(axes, (cube, bunny), ("CUBE", "BUNNY")):
    for (label, pts), col, mk in zip(data.items(), (BLUE, ORANGE, AQUA), ("o", "s", "^")):
        ks = [p[0] for p in pts]
        ts = [p[1] for p in pts]
        ax.plot(ks, ts, marker=mk, color=col, label=label, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1)
        for k, t, off, ha in pts:
            ax.annotate(f"{t:g}", (k, t), textcoords="offset points", xytext=off,
                        fontsize=7.5, color=INK2, ha=ha)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K, number of pieces")
    ax.set_title(ttl, fontsize=10)
    ax.legend(title="voxel grid", title_fontsize=8, loc="upper left")
    clean(ax)
axes[0].set_ylabel("generation time (min)")
axes[0].set_ylim(0.01, 3000)
axes[0].set_xlim(7, 800)
axes[1].set_xlim(7, 250)
fig.suptitle("Time grows with K and N but not monotonically: the 20×19×15 bunny "
             "at K = 75 took 4× longer than the larger one", fontsize=10.5, fontweight="semibold")
fig.subplots_adjust(top=0.82, wspace=0.12)
save(fig, "song2012-recursive-interlocking-puzzles-3.png")
