"""Figures for fu2015-interlocking-furniture-assembly.md.

Figure 1: three voxel joints, and the six-axis blocking test run on each to
          show they leave exactly one free direction.
Figure 2: the paper's Figure 4 four-part cycle: joint directions, the 6-bit
          mobility codes after each joint, and the subset test on all 14
          proper subsets (computed here).
Figure 3: Tables 1 and 2: the planner against the randomized baseline.
"""
from itertools import combinations

import numpy as np
from mpl_toolkits.mplot3d import proj3d

from _style import *  # noqa: F401,F403

DIRS = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0),
        "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}
BITS = list(DIRS)  # bit order of the paper's 6-bit code


# ----------------------------------------------------------- figure 1: joints
def box(x0, x1, y0, y1, z0, z1):
    return {(x, y, z) for x in range(x0, x1) for y in range(y0, y1) for z in range(z0, z1)}


def free_dirs(A, B, limit=12):
    """Directions along which A can translate away from B, voxel by voxel,
    until it no longer overlaps B's bounding box (i.e. it is removable)."""
    out = []
    bx = [min(v[i] for v in B) for i in range(3)], [max(v[i] for v in B) for i in range(3)]
    for name, d in DIRS.items():
        cur = set(A)
        ok = True
        for _ in range(limit):
            cur = {(v[0] + d[0], v[1] + d[1], v[2] + d[2]) for v in cur}
            if cur & B:
                ok = False
                break
            inside = any(all(bx[0][i] <= v[i] <= bx[1][i] for i in range(3)) for v in cur)
            if not inside:
                break
        if ok:
            out.append(name)
    return out


# (a) halved (cross-lap) joint: two orthogonal bars, each notched half depth
lapA = box(0, 5, 2, 3, 0, 2) - {(2, 2, 1)}
lapB = box(2, 3, 0, 5, 0, 2) - {(2, 2, 0)}
# (b) mortise and tenon: a block with a 1-voxel mortise, a bar whose end fills it
mtB = box(0, 2, 0, 3, 0, 3) - {(1, 1, 1)}
mtA = {(1, 1, 1), (2, 1, 1), (3, 1, 1), (4, 1, 1)}
# (c) blind sliding dovetail, voxel stand-in: a T-slot open at one end only
channel = {(x, y, 1) for x in range(3) for y in range(1, 5)}
neck = {(1, y, 2) for y in range(1, 5)}
dtB = box(0, 3, 0, 5, 0, 3) - channel - neck
dtA = ({(x, y, 1) for x in range(3) for y in range(1, 4)}
       | {(1, y, 2) for y in range(1, 4)}
       | {(x, y, 3) for x in range(3) for y in range(1, 4)})

joints = [("halved (cross-lap) joint", lapA, lapB, (24, -55)),
          ("mortise and tenon", mtA, mtB, (22, -28)),
          ("blind sliding dovetail (T-slot stand-in)", dtA, dtB, (24, 32))]

fig = plt.figure(figsize=(8, 3.5))
for i, (name, A, B, view) in enumerate(joints):
    fr = free_dirs(A, B)
    assert len(fr) == 1, (name, fr)
    code = "".join("1" if b in fr else "0" for b in BITS)
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    allv = A | B
    nx_, ny_, nz_ = (max(v[k] for v in allv) + 1 for k in range(3))
    filled = np.zeros((nx_, ny_, nz_), dtype=bool)
    colors = np.empty((nx_, ny_, nz_), dtype=object)
    for v in A:
        filled[v] = True
        colors[v] = BLUE
    for v in B:
        filled[v] = True
        colors[v] = WOOD
    ax.voxels(filled, facecolors=colors, edgecolors=INK2, linewidth=0.4)
    d = DIRS[fr[0]]
    axis = [k for k in range(3) if d[k] != 0][0]
    ext = max(v[axis] for v in A) if d[axis] > 0 else min(v[axis] for v in A)
    face = [v for v in A if v[axis] == ext]
    c = np.mean(face, axis=0) + 0.5
    c[axis] = ext + (1.05 if d[axis] > 0 else -0.05)
    ax.set_box_aspect((nx_, ny_, nz_), zoom=1.15)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlim(0, nx_); ax.set_ylim(0, ny_); ax.set_zlim(0, nz_)
    e = c + 2.2 * np.array(d)
    x0, y0, _ = proj3d.proj_transform(c[0], c[1], c[2], ax.get_proj())
    x1, y1, _ = proj3d.proj_transform(e[0], e[1], e[2], ax.get_proj())
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords="data", textcoords="data",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.5, mutation_scale=16),
                zorder=100, annotation_clip=False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel("x", labelpad=-10); ax.set_ylabel("y", labelpad=-10); ax.set_zlabel("z", labelpad=-10)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(GRID)
    ax.grid(False)
    ax.set_title(name, fontsize=9.5, y=0.98)
    ax.text2D(0.5, -0.13, f"part A (blue) free: {fr[0]} only\ncode {code}",
              transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=INK2)
fig.suptitle("Each joint in the library leaves its part exactly one free axis "
             "(six-direction blocking test on voxel models)", fontsize=10.5, fontweight="semibold")
fig.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.2, wspace=0.0)
save(fig, "fu2015-interlocking-furniture-assembly-1.png")


# ------------------------------------------------- figure 2: the 4-part cycle
PARTS = ["P1", "P2", "P3", "P4"]
# paper's Figure 4, in assignment order: (A, B, direction A may leave B)
JOINTS = [("P1", "P4", "+y"), ("P1", "P2", "+y"), ("P2", "P3", "-x"), ("P3", "P4", "-z")]
NEG = {"+x": "-x", "-x": "+x", "+y": "-y", "-y": "+y", "+z": "-z", "-z": "+z"}


def dir_of(joints, A, B):
    for a, b, d in joints:
        if (a, b) == (A, B):
            return d
        if (a, b) == (B, A):
            return NEG[d]
    return None


def codes_after(joints):
    """6-bit code per part: bit ON = part still free along that axis."""
    code = {p: set(BITS) for p in PARTS}
    for a, b, d in joints:
        code[a] &= {d}
        code[b] &= {NEG[d]}
    return {p: "".join("1" if bit in code[p] else "0" for bit in BITS) for p in PARTS}


def mobile_subsets(joints):
    """Subset S can translate along d iff every joint crossing S's boundary
    lets the S-side part leave along d (the paper's 'agreeing directions')."""
    res = {}
    for r in range(1, len(PARTS)):
        for S in combinations(PARTS, r):
            free = []
            for d in BITS:
                ok = True
                for a, b, _ in joints:
                    ina, inb = a in S, b in S
                    if ina != inb:
                        side = a if ina else b
                        other = b if ina else a
                        if dir_of(joints, side, other) != d:
                            ok = False
                            break
                if ok:
                    free.append(d)
            res[S] = free
    return res


stages = [codes_after(JOINTS[:k]) for k in range(len(JOINTS) + 1)]
mob = mobile_subsets(JOINTS)
print("mobile subsets:", {s: f for s, f in mob.items() if f})
assert {s for s, f in mob.items() if f} == {("P1",), ("P2", "P3", "P4")}

fig = plt.figure(figsize=(8, 5.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 2.1],
                      left=0.05, right=0.98, top=0.88, bottom=0.08, wspace=0.3, hspace=0.4)

# (a) the frame, front view: x right, z up, y into the page
ax = fig.add_subplot(gs[:, 0])
rects = {"P1": (0, 5, 6, 1), "P4": (0, 0, 1, 5), "P2": (5, 0, 1, 5), "P3": (1, 1, 4, 1)}
fills = {"P1": BLUE, "P2": WOOD, "P3": WOOD2, "P4": WOOD}
for p, (x, z, w, h) in rects.items():
    ax.add_patch(plt.Rectangle((x, z), w, h, facecolor=fills[p], edgecolor=INK2, lw=0.9, zorder=2))
    ax.text(x + w / 2, z + h / 2, p, ha="center", va="center", fontsize=9, fontweight="semibold",
            color=INK, zorder=4)
# joint marks: (x, z) of the contact, label lines
jpos = {("P1", "P4"): (0.5, 5.0), ("P1", "P2"): (5.5, 5.0), ("P2", "P3"): (5.0, 1.5), ("P3", "P4"): (1.0, 1.5)}
glyph = {"+y": "⊗ +y", "-y": "⊙ −y", "+x": "→ +x",
         "-x": "← −x", "+z": "↑ +z", "-z": "↓ −z"}
offs = {("P1", "P4"): (-0.9, 1.3, "right"), ("P1", "P2"): (0.9, 1.3, "left"),
        ("P2", "P3"): (1.4, -1.2, "left"), ("P3", "P4"): (-1.4, -1.2, "right")}
for (a, b, d) in JOINTS:
    x, z = jpos[(a, b)]
    ax.plot(x, z, "o", color=INK, markersize=4, zorder=5)
    dx, dz, ha = offs[(a, b)]
    ax.annotate(f"J{a[1]}{b[1]}: {a} {glyph[d]}\n{b} {glyph[NEG[d]]}", (x, z),
                xytext=(x + dx, z + dz), fontsize=7.2, color=INK2, ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6), zorder=6)
ax.set_xlim(-4.2, 10.2)
ax.set_ylim(-2.6, 7.6)
ax.text(3, -2.2, "\u2297 = +y, into the page;  \u2299 = \u2212y, out of the page",
        ha="center", va="center", fontsize=7.5, color=INK2)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_xticks(range(0, 7)); ax.set_yticks(range(0, 7))
clean(ax)
ax.set_title("four parts, four joints (paper Fig. 4)", fontsize=9.5)

# (b) code table: rows = stages, columns = parts
ax = fig.add_subplot(gs[0, 1])
labels = ["start"] + [f"+J{a[1]}{b[1]}" for a, b, _ in JOINTS]
for i, (lab, st) in enumerate(zip(labels, stages)):
    ax.text(-0.15, len(stages) - 1 - i + 0.5, lab, ha="right", va="center", fontsize=8, color=INK2)
    for j, p in enumerate(PARTS):
        code = st[p]
        n_on = code.count("1")
        col = GOOD if n_on == 1 and i == len(stages) - 1 and p == "P1" else (SURFACE if n_on else STONE)
        ax.add_patch(plt.Rectangle((j, len(stages) - 1 - i), 1, 1, facecolor=col, edgecolor=GRID, lw=0.8))
        ax.text(j + 0.5, len(stages) - 1 - i + 0.5, code, ha="center", va="center", fontsize=8,
                family="DejaVu Sans Mono", color=SURFACE if col == GOOD else INK)
for j, p in enumerate(PARTS):
    ax.text(j + 0.5, len(stages) + 0.15, p, ha="center", va="bottom", fontsize=8.5, color=INK)
ax.set_xlim(-1.3, 4)
ax.set_ylim(0, len(stages) + 0.7)
ax.set_axis_off()
ax.set_title("6-bit code (+x −x +y −y +z −z), 1 = still free", fontsize=9, y=1.0)

# (c) subset test
ax = fig.add_subplot(gs[1, 1])
subs = list(mob)
for i, S in enumerate(subs):
    for j, d in enumerate(BITS):
        free = d in mob[S]
        ax.add_patch(plt.Rectangle((j, len(subs) - 1 - i), 1, 1, facecolor=GOOD if free else SURFACE,
                                   edgecolor=GRID, lw=0.6))
        if free:
            ax.text(j + 0.5, len(subs) - 1 - i + 0.5, "free", ha="center", va="center",
                    fontsize=7.5, color=SURFACE, fontweight="semibold")
    ax.text(-0.1, len(subs) - 1 - i + 0.5, "{" + ",".join(s[1] for s in S) + "}", ha="right",
            va="center", fontsize=7.5, color=INK2)
for j, d in enumerate(BITS):
    ax.text(j + 0.5, len(subs) + 0.1, d, ha="center", va="bottom", fontsize=8, color=INK)
ax.set_xlim(-1.6, 6)
ax.set_ylim(0, len(subs) + 0.8)
ax.set_axis_off()
ax.set_title("subset test: only {P1} and {P2,P3,P4} can move", fontsize=9, y=1.0)

fig.suptitle("Each joint switches bits off; after four joints only the key P1 keeps one, "
             "and no subset of the others can move", fontsize=10.5, fontweight="semibold")
save(fig, "fu2015-interlocking-furniture-assembly-2.png")


# ------------------------------------------------------ figure 3: Tables 1-2
# Table 1 (planner): model -> (parts, joints, LIGs, valid configurations, seconds)
t1 = {"CONSOLE TABLE": (6, 8, 3, 87, 0.02), "BEDSTAND": (6, 9, 3, 46, 0.07),
      "BENCH": (9, 12, 4, 112, 0.38), "SHOE RACK": (9, 16, 5, 4488, 18.89),
      "MULTI-FUNCTION TABLE": (10, 18, 5, 124, 2.29), "CHAIR": (12, 18, 6, 320, 13.16),
      "SOFA": (13, 20, 4, 8766, 5.9), "BOOKSHELF": (13, 22, 6, 97808, 62.3),
      "BABY BED": (23, 42, 6, 43160, 48.98), "CHILD BED": (32, 66, 8, 15492, 157.68)}
# Table 2 (randomized baseline): model -> (trials as printed, non-interlocking,
# non-assemblable, valid, seconds)
t2 = {"CONSOLE TABLE": (1e6, 68184, 30523, 1293, 6.18), "BENCH": (1e6, 72049, 27728, 223, 47.8),
      "SHOE RACK": (1e6, 72573, 27424, 3, 73.83),
      "MULTI-FUNCTION TABLE": (1e6, 68481, 31519, 0, 135.24),
      "BOOKSHELF": (1e6, 82853, 17147, 0, 1359.74), "BABY BED": (1e3, 7937, 2063, 0, 12906.2)}
models = list(t2)
x = np.arange(len(models))
short = ["CONSOLE\nTABLE", "BENCH", "SHOE\nRACK", "MULTI-FN\nTABLE", "BOOK-\nSHELF", "BABY\nBED"]
short = [s + f"\n{t1[m][0]} parts" for s, m in zip(short, models)]

fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
ax = axes[0]
ax.plot(x, [t1[m][4] for m in models], "o", color=BLUE, label="planner (Table 1)", markersize=7)
ax.plot(x, [t2[m][4] for m in models], "s", color=ORANGE, label="randomized baseline (Table 2)", markersize=6)
for xi, m in zip(x, models):
    ax.plot([xi, xi], [t1[m][4], t2[m][4]], color=GRID, lw=1, zorder=0)
ax.set_yscale("log")
ax.set_ylabel("time (s)")
ax.set_xticks(x)
ax.set_xticklabels(short, fontsize=6.5)
ax.set_xlabel("furniture model")
ax.set_title("time taken", fontsize=10)
clean(ax)

ax = axes[1]
floor = 0.4
v1 = [t1[m][3] for m in models]
v2 = [t2[m][3] for m in models]
ax.plot(x, v1, "o", color=BLUE, label="planner (Table 1)", markersize=7)
ax.plot(x, [v if v > 0 else floor for v in v2], "s", color=ORANGE, label="randomized baseline (Table 2)", markersize=6)
for xi, v in zip(x, v2):
    if v == 0:
        ax.text(xi, floor, "0", ha="center", va="bottom", fontsize=8, color=INK2)
ax.set_yscale("log")
ax.set_ylim(0.2, 5e5)
ax.set_ylabel("valid configurations found (count)")
ax.set_xticks(x)
ax.set_xticklabels(short, fontsize=6.5)
ax.set_xlabel("furniture model")
ax.set_title("global interlocking configurations", fontsize=10)
clean(ax)
handles, labels_ = ax.get_legend_handles_labels()
fig.legend(handles, labels_, loc="upper center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, 0.95))
fig.suptitle("Chaining local groups finds thousands of valid joint networks in seconds; "
             "random joint choice finds few or none in hours", fontsize=10.5, fontweight="semibold")
fig.subplots_adjust(top=0.76, bottom=0.25, wspace=0.3)
save(fig, "fu2015-interlocking-furniture-assembly-3.png")
