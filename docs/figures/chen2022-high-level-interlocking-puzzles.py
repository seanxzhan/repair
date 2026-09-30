"""Figures for chen2022-high-level-interlocking-puzzles.md.

The kernel disassembly graph of Section 4 (Algorithms 1 and 2) is
re-implemented here and run on two small voxel puzzles of our own, so the
levels, graphs and disassembly moves drawn below are computed, not sketched.

Figure 1: the kernel disassembly plan of a 4-piece level-2 puzzle of ours: one
          piece must step aside before the first piece can be pulled out.
Figure 2: the kernel disassembly graphs of a level-1 and a level-2 puzzle, with
          the shortest root-to-target path that defines the exact level.
Figure 3: Tables 2 and 3 of the paper.
"""
from collections import deque
from itertools import combinations

import numpy as np

from _style import *  # noqa: F401,F403

# --------------------------------------------------------------- the planner
DIRS = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0),
        "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


def add(c, d, h=1):
    return tuple(ci + di * h for ci, di in zip(c, d))


def build_graph(pieces, max_nodes=20000):
    """Breadth-first enumeration of every configuration the pieces can reach
    by whole-voxel axis translations, as in the paper's Algorithm 1.  A node is
    the tuple of per-piece integer displacements; an edge is one move of one
    connected subassembly of at most ceil(K/2) pieces (Algorithm 2).  A
    subassembly whose travel is unbounded is *removed*: that gives a target."""
    K = len(pieces)
    allc = [c for p in pieces for c in p]
    span = max(max(c[i] for c in allc) - min(c[i] for c in allc)
               for i in range(3)) + 2
    root = tuple((0, 0, 0) for _ in range(K))

    def placed(state, i):
        return {add(c, state[i]) for c in pieces[i]}

    def subassemblies(state):
        occ = {}
        for i in range(K):
            for c in placed(state, i):
                occ[c] = i
        adj = {i: set() for i in range(K)}
        for c, i in occ.items():
            for d in DIRS.values():
                n = add(c, d)
                if n in occ and occ[n] != i:
                    adj[i].add(occ[n])
                    adj[occ[n]].add(i)
        out = []
        for r in range(1, (K + 1) // 2 + 1):      # relativity: only half of them
            for S in combinations(range(K), r):
                Ss, st, seen = set(S), [S[0]], set()
                while st:
                    u = st.pop()
                    if u in seen:
                        continue
                    seen.add(u)
                    st.extend((adj[u] & Ss) - seen)
                if seen == Ss:                     # must touch to move together
                    out.append(S)
        return out

    def neighbors(state):
        res = []
        for S in subassemblies(state):
            Ss = set(S)
            moving = set().union(*(placed(state, i) for i in S))
            rest = set().union(*(placed(state, i) for i in range(K) if i not in Ss))
            for dn, d in DIRS.items():
                steps, removable = 0, False
                for h in range(1, span + 3):
                    if {add(c, d, h) for c in moving} & rest:
                        break
                    steps = h
                    if h >= span + 2:
                        removable = True
                if steps == 0:
                    continue
                if removable:
                    res.append(("remove", S, dn, None))
                else:
                    for h in range(1, steps + 1):
                        ns = list(state)
                        for i in S:
                            ns[i] = add(state[i], d, h)
                        res.append(("move", S, dn, (tuple(ns), h)))
        return res

    nodes, order = {root: 0}, [root]
    edges, targets = {}, {}
    dist, parent = {root: 0}, {root: None}
    q = deque([root])
    while q:
        st = q.popleft()
        for kind, S, dn, payload in neighbors(st):
            if kind == "remove":
                targets.setdefault(nodes[st], []).append((S, dn))
                continue
            ns, h = payload
            if ns not in nodes:
                nodes[ns] = len(order)
                order.append(ns)
                dist[ns] = dist[st] + 1
                parent[ns] = (st, S, dn, h)
                if len(nodes) > max_nodes:
                    raise RuntimeError("graph too large")
                q.append(ns)
            edges.setdefault(frozenset((nodes[st], nodes[ns])), (S, dn, h))
    level = (min(dist[order[i]] for i in targets) + 1) if targets else 0
    return dict(nodes=nodes, order=order, edges=edges, targets=targets,
                dist=dist, parent=parent, level=level, pieces=pieces)


def shortest_kernel_plan(g):
    """The moves along a shortest root-to-target path: the paper's L_exact."""
    src = min(g["targets"], key=lambda i: g["dist"][g["order"][i]])
    node, chain = g["order"][src], []
    while g["parent"][node] is not None:
        prev, S, dn, h = g["parent"][node]
        chain.append((S, dn, h, g["nodes"][node]))
        node = prev
    chain.reverse()
    return chain, g["targets"][src][0], src


def disassemblable(pieces, depth=0):
    """Algorithm 3: take out a subassembly and recurse on both halves, so the
    puzzle really does come apart (the paper's design requirement 4)."""
    if len(pieces) == 1:
        return True
    if depth > 6:
        return False
    try:
        g = build_graph(pieces, max_nodes=3000)
    except RuntimeError:       # a loose sub-assembly can wander very far
        return False
    for src, rs in sorted(g["targets"].items()):
        for S, dn in rs:
            out = [pieces[i] for i in S]
            rest = [pieces[i] for i in range(len(pieces)) if i not in set(S)]
            if disassemblable(out, depth + 1) and disassemblable(rest, depth + 1):
                return True
    return False


# ------------------------------------------------------------- the puzzles
# Both were found by hill-climbing single-voxel reassignments between pieces
# (the paper's Algorithm 4 in miniature), maximising: nothing removable in the
# assembled state, then the exact level.  They are listed literally so this
# script is deterministic.
HIGH = [  # 4 pieces, 4x4x4 Cube with one hole voxel at (1,1,1); level 2
    [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(0,1,2),(0,2,0),(0,2,1),(0,2,2),(0,3,0),
     (0,3,1),(0,3,2),(1,2,1),(1,3,2),(1,3,3)],
    [(2,0,0),(2,0,1),(2,0,2),(2,1,0),(2,1,2),(3,0,0),(3,0,1),(3,1,0),(3,1,1),
     (3,1,2),(3,2,0),(3,2,2),(3,2,3),(3,3,2)],
    [(1,1,2),(1,2,0),(1,2,2),(1,3,0),(1,3,1),(2,1,1),(2,2,0),(2,2,1),(2,2,2),
     (2,3,0),(2,3,1),(3,2,1),(3,3,0),(3,3,1)],
    [(0,0,2),(0,0,3),(0,1,3),(0,2,3),(0,3,3),(1,0,0),(1,0,1),(1,0,2),(1,0,3),
     (1,1,0),(1,1,3),(1,2,3),(2,0,3),(2,1,3),(2,2,3),(2,3,2),(2,3,3),(3,0,2),
     (3,0,3),(3,1,3),(3,3,3)],
]
LOW = [  # 4 pieces, 3x3x3 Cube with one hole voxel at (1,1,1); level 1
    [(0,0,0),(0,1,0),(0,2,0),(1,0,0),(1,1,0),(2,0,0)],
    [(1,1,2),(1,2,2),(2,1,1),(2,1,2),(2,2,2)],
    [(0,1,1),(0,2,1),(1,2,0),(1,2,1),(2,1,0),(2,2,0),(2,2,1)],
    [(0,0,1),(0,0,2),(0,1,2),(0,2,2),(1,0,1),(1,0,2),(2,0,1),(2,0,2)],
]
HIGH = [frozenset(p) for p in HIGH]
LOW = [frozenset(p) for p in LOW]

PNAME = ["P1", "P2", "P3", "P4"]

gh = build_graph(HIGH)
gl = build_graph(LOW)
print("high-level puzzle: level", gh["level"], "nodes", len(gh["order"]),
      "edges", len(gh["edges"]), "target nodes", len(gh["targets"]))
print("level-1 puzzle:    level", gl["level"], "nodes", len(gl["order"]),
      "edges", len(gl["edges"]), "target nodes", len(gl["targets"]))
assert gh["level"] > 1 and 0 not in gh["targets"]
assert gl["level"] == 1 and 0 in gl["targets"]
assert disassemblable(HIGH) and disassemblable(LOW)   # both really come apart
chain_h, remove_h, src_h = shortest_kernel_plan(gh)
print("kernel plan:", [([PNAME[i] for i in S], dn, h) for S, dn, h, _ in chain_h],
      "then remove", [PNAME[i] for i in remove_h[0]], remove_h[1])


# ------------------------------------------------------------------ drawing
def draw_config(ax, pieces, state, key=None, mover=None, pad=0, elev=22, azim=-58,
                title=None):
    lo = [min(c[i] + s[i] for p, s in zip(pieces, state) for c in p) - pad
          for i in range(3)]
    hi = [max(c[i] + s[i] for p, s in zip(pieces, state) for c in p) + pad
          for i in range(3)]
    shape = [hi[i] - lo[i] + 1 for i in range(3)]
    filled = np.zeros(shape, dtype=bool)
    colors = np.empty(shape, dtype=object)
    fills = [WOOD, WOOD2, STONE, WOOD]
    for i, (p, s) in enumerate(zip(pieces, state)):
        col = BLUE if i == key else (ORANGE if i == mover else fills[i % len(fills)])
        for c in p:
            v = tuple(c[k] + s[k] - lo[k] for k in range(3))
            filled[v] = True
            colors[v] = col
    ax.voxels(filled, facecolors=colors, edgecolors=INK2, linewidth=0.3)
    ax.set_box_aspect(shape, zoom=1.35)
    ax.view_init(elev=elev, azim=azim)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.fill = False
        a.pane.set_edgecolor(GRID)
        a.line.set_color(GRID)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    if title:
        ax.text2D(0.5, -0.02, title, transform=ax.transAxes, ha="center",
                  va="top", fontsize=8.2, color=INK2)


def draw_graph(ax, g, title, path_nodes=None):
    """x = number of moves from the assembled state, y spread inside a layer.
    A node is orange when some subassembly can be removed from it."""
    layers = {}
    for st, i in g["nodes"].items():
        layers.setdefault(g["dist"][st], []).append(i)
    pos = {}
    for d, ids in layers.items():
        for k, i in enumerate(sorted(ids)):
            pos[i] = (d, k - (len(ids) - 1) / 2.0)
    path_nodes = path_nodes or []
    pe = {frozenset(e) for e in zip(path_nodes, path_nodes[1:])}
    for e in g["edges"]:
        a, b = tuple(e)
        on = frozenset((a, b)) in pe
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color=BLUE if on else GRID, lw=2.4 if on else 1.0,
                zorder=3 if on else 1, solid_capstyle="round")
    for i, (x, y) in pos.items():
        tgt = i in g["targets"]
        ax.plot([x], [y], marker="o", markersize=8 if tgt else 6,
                color=ORANGE if tgt else SURFACE, markeredgecolor=INK2,
                markeredgewidth=0.9, zorder=5)
    ax.plot([], [], "o", color=SURFACE, markeredgecolor=INK2,
            label="configuration, still locked")
    ax.plot([], [], "o", color=ORANGE, markeredgecolor=INK2,
            label="a subassembly can be removed from here")
    ax.plot([], [], "-", color=BLUE, lw=2.4, label="shortest path to one")
    ax.set_xlabel("moves from the assembled puzzle (count)")
    ax.set_xticks(sorted(layers))
    ax.set_yticks([])
    ax.set_ylabel("")
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=9.5)


# ------------------------------------------------------------------ figure 1
# the kernel disassembly plan: one panel per configuration along the shortest
# path, then the removal itself.
path_h = [0] + [n for _, _, _, n in chain_h]
key = remove_h[0][0]                      # the piece that leaves first
movers = sorted({i for S, _, _, _ in chain_h for i in S} - {key})
mover = movers[0] if movers else None
steps = [("assembled: no subassembly of any size\n"
          "can be removed along any of the six axes", gh["order"][0])]
for S, dn, h, node in chain_h:
    who = "+".join(PNAME[i] for i in S)
    steps.append((f"move {len(steps)}: {who} slides {dn} by {h} step,\n"
                  f"then it is stopped — it cannot leave", gh["order"][node]))
last = list(gh["order"][path_h[-1]])
last[key] = add(last[key], DIRS[remove_h[1]], 3)
steps.append((f"move {len(steps)}: {PNAME[key]} is now free and\n"
              f"pulls out {remove_h[1]} — level {gh['level']} reached", tuple(last)))

fig = plt.figure(figsize=(8, 3.4))
gs = fig.add_gridspec(1, len(steps), wspace=0.02, left=0.01, right=0.99,
                      top=0.88, bottom=0.24)
for k, (ttl, state) in enumerate(steps):
    ax = fig.add_subplot(gs[k], projection="3d")
    draw_config(ax, HIGH, state, key=key, mover=mover, title=ttl)
fig.suptitle(f"Our 4-piece 4×4×4 puzzle is level {gh['level']}: {PNAME[mover]} (orange) has to "
             f"step aside before {PNAME[key]} (blue) can leave",
             fontsize=10.5, fontweight="semibold", y=0.97)
save(fig, "chen2022-high-level-interlocking-puzzles-1.png")

# ------------------------------------------------------------------ figure 2
fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
draw_graph(axes[0], gl, f"level-1 puzzle (3×3×3): {len(gl['order'])} configurations, "
                        f"but\na piece comes out of the assembled one", [0])
nmv = gh["level"] - 1
draw_graph(axes[1], gh, f"level-{gh['level']} puzzle (4×4×4): only {len(gh['order'])} "
                        f"configurations,\nand the nearest way out is "
                        f"{nmv} move{'s' if nmv != 1 else ''} away", path_h)
axes[0].legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(1.06, -0.2), ncol=3)
spread = max(len([1 for st, i in g["nodes"].items() if g["dist"][st] == d])
             for g in (gl, gh) for d in set(g["dist"].values()))
for ax in axes:
    ax.set_ylim(-spread / 2 - 0.6, spread / 2 + 0.6)
fig.suptitle("The level is the distance to the nearest exit, not the size of the motion "
             "space", fontsize=10.5, fontweight="semibold")
fig.subplots_adjust(top=0.78, bottom=0.27, wspace=0.12, left=0.04, right=0.98)
save(fig, "chen2022-high-level-interlocking-puzzles-2.png")

# ------------------------------------------------------- figure 3: Tables 2-3
# Table 2: highest level reached in 12 h, this paper vs the baseline
# [Gontier 2020], on Cubes with a single centre hole voxel.
T2_SETTINGS = [("4³", 3), ("4³", 4), ("4³", 5), ("4³", 6),
               ("5³", 3), ("5³", 4), ("5³", 5), ("5³", 6),
               ("6³", 3), ("6³", 4), ("6³", 5), ("6³", 6)]
T2_GONTIER = [8, 5, 5, 1, 9, 8, 8, 1, 3, 1, 8, 1]
T2_OURS = [8, 6, 8, 5, 11, 16, 14, 13, 15, 20, 27, 17]
# Table 3: minutes to generate a K-piece level-L 5x5x5 Cube with one hole voxel.
T3 = {4: {4: 0.028, 8: 4.540, 12: 68.272, 16: 70.757},
      6: {4: 0.057, 8: 3.909, 12: 45.643},
      8: {4: 1.463, 8: 62.011, 12: 365.675}}

fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), gridspec_kw={"width_ratios": [1.45, 1]})

ax = axes[0]
x = np.arange(len(T2_SETTINGS))
w = 0.38
ax.bar(x - w / 2, T2_OURS, w, color=BLUE, label="Chen et al. 2022", zorder=2)
ax.bar(x + w / 2, T2_GONTIER, w, color=ORANGE, label="baseline [Gontier 2020]", zorder=2)
for xi, v in zip(x, T2_OURS):
    ax.text(xi - w / 2, v + 0.6, str(v), ha="center", fontsize=7, color=INK)
ax.set_xticks(x)
ax.set_xticklabels([f"{r}\nK={k}" for r, k in T2_SETTINGS], fontsize=7)
ax.set_xlabel("Cube resolution and piece count K")
ax.set_ylabel("highest level reached in 12 h (moves)")
ax.set_ylim(0, 31)
ax.legend(fontsize=7.5, loc="upper left")
ax.set_title("Table 2: same 12 h budget, both planners", fontsize=9.5)
clean(ax)

ax = axes[1]
for (K, row), col, mk in zip(sorted(T3.items()), (BLUE, ORANGE, AQUA), ("o", "s", "^")):
    Ls = sorted(row)
    ax.plot(Ls, [row[L] for L in Ls], marker=mk, color=col, label=f"K = {K} pieces",
            markersize=6, markeredgecolor=SURFACE, markeredgewidth=1)
# label the cheapest and the dearest run of each series, offset per series so
# the three curves' labels never collide
LAB = {4: ((0, -14), "center"), 6: ((13, -3), "left"), 8: ((0, 8), "center")}
for K, row in T3.items():
    off, ha = LAB[K]
    for L in (min(row), max(row)):
        ax.annotate(f"{row[L]:g}", (L, row[L]), textcoords="offset points",
                    xytext=off, ha=ha, fontsize=7, color=INK2)
ax.set_yscale("log")
ax.set_xticks([4, 8, 12, 16])
ax.set_xlabel("target level L (moves)")
ax.set_ylabel("generation time (min)")
ax.set_ylim(0.015, 3000)
ax.legend(fontsize=7.5, loc="lower right")
ax.set_title("Table 3: 5×5×5 Cube, one hole voxel", fontsize=9.5)
clean(ax)

fig.suptitle("Levels far beyond the genetic-algorithm baseline, at a cost that climbs "
             "steeply with level and piece count", fontsize=10.5, fontweight="semibold")
fig.subplots_adjust(top=0.82, bottom=0.2, wspace=0.28, left=0.08, right=0.98)
save(fig, "chen2022-high-level-interlocking-puzzles-3.png")
