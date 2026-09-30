"""Figures for song2017-reconfigurable-interlocking-furniture.md.

A toy reconfigurable set: two rails (R1, R2) and two rungs (S1, S2), each with
two half joints, assembled into two forms.  Form A is a ladder-like frame.
Form B is a table-like frame in which R1 and R2 are turned so that R2's two
half joints swap partners.  Each half joint has one free direction fixed in
its part's frame; the paper's compatibility constraint T_a d_a = -T_b d_b
must hold for every mate in every form.

Figure 1: the two forms with one direction assignment that interlocks both,
          the half-joint graph, and the functional-stability check.
Figure 2: the enumeration funnel: how many of the 4^8 assignments survive
          each constraint.
Figure 3: Tables 1 and 2 of the paper.
"""
from itertools import combinations, product

import numpy as np

from _style import *  # noqa: F401,F403

# ----------------------------------------------------------------- the parts
# 2D, x right, z up.  Local frames: rails span local z in [0, 4], x in [0, 1];
# half joint "a" at local z in [3, 4], "b" at local z in [0, 1].
# Rungs span local x in [0, 4], z in [0, 1]; "l" at local x in [0, 1], "r" at [3, 4].
PARTS = ["R1", "R2", "S1", "S2"]
HALF = [("R1", "a"), ("R1", "b"), ("R2", "a"), ("R2", "b"),
        ("S1", "l"), ("S1", "r"), ("S2", "l"), ("S2", "r")]
LOCAL_RECT = {"R1": (1, 4), "R2": (1, 4), "S1": (4, 1), "S2": (4, 1)}  # (w, h) in local frame
LOCAL_HJ = {("R1", "a"): (0.5, 3.5), ("R1", "b"): (0.5, 0.5), ("R2", "a"): (0.5, 3.5), ("R2", "b"): (0.5, 0.5),
            ("S1", "l"): (0.5, 0.5), ("S1", "r"): (3.5, 0.5), ("S2", "l"): (0.5, 0.5), ("S2", "r"): (3.5, 0.5)}

I2 = np.array([[1, 0], [0, 1]])
ROT_P90 = np.array([[0, -1], [1, 0]])   # local z -> world -x
ROT_M90 = np.array([[0, 1], [-1, 0]])   # local z -> world +x

# pose per part per form: (T, t) with world = T @ local + t.
# Form A: all parts unrotated.  Form B: rails horizontal (top and bottom),
# rungs vertical (legs); each part can be turned end for end, which is the
# paper's orientation-association choice T_{k,i}.  We enumerate all 16.
FORM_A = {"R1": (I2, np.array([0, 0])), "R2": (I2, np.array([3, 0])),
          "S1": (I2, np.array([0, 3])), "S2": (I2, np.array([0, 0]))}


def form_b(r1_a_left, r2_a_left, s1_l_bottom, s2_l_bottom):
    return {
        "R1": (ROT_P90, np.array([4, 3])) if r1_a_left else (ROT_M90, np.array([0, 4])),
        "R2": (ROT_P90, np.array([4, 0])) if r2_a_left else (ROT_M90, np.array([0, 1])),
        "S1": (ROT_P90, np.array([1, 0])) if s1_l_bottom else (ROT_M90, np.array([0, 4])),
        "S2": (ROT_P90, np.array([4, 0])) if s2_l_bottom else (ROT_M90, np.array([3, 4])),
    }


FORMS = {"form A: ladder frame": FORM_A, "form B: table frame": None}
DIR = {"+x": np.array([1, 0]), "-x": np.array([-1, 0]), "+z": np.array([0, 1]), "-z": np.array([0, -1])}
ARROW = {"+x": "→", "-x": "←", "+z": "↑", "-z": "↓"}


def world_hj(form, hj):
    T, t = FORMS[form][hj[0]]
    return T @ np.array(LOCAL_HJ[hj]) + t


def mates(form):
    """Half joints whose world positions coincide are mated in this form."""
    out = []
    for h1, h2 in combinations(HALF, 2):
        if h1[0] != h2[0] and np.allclose(world_hj(form, h1), world_hj(form, h2)):
            out.append((h1, h2))
    assert len(out) == 4, out
    return out


MATES = {}


def name_dir(v):
    for k, d in DIR.items():
        if np.array_equal(v, d):
            return k
    raise ValueError(v)


def world_dirs(form, assign):
    """World free direction of each half joint under the form's poses."""
    return {hj: name_dir(FORMS[form][hj[0]][0] @ DIR[assign[hj]]) for hj in HALF}


def compatible(form, assign):
    wd = world_dirs(form, assign)
    return all(np.array_equal(DIR[wd[h1]], -DIR[wd[h2]]) for h1, h2 in MATES[form])


def mobile_subsets(form, assign):
    """Fu 2015 test: subset S moves along d iff at every joint crossing S's
    boundary, the S-side half joint's world direction equals d."""
    wd = world_dirs(form, assign)
    res = {}
    for r in range(1, len(PARTS)):
        for S in combinations(PARTS, r):
            free = []
            for d in DIR:
                ok = True
                for h1, h2 in MATES[form]:
                    in1, in2 = h1[0] in S, h2[0] in S
                    if in1 != in2:
                        side = h1 if in1 else h2
                        if wd[side] != d:
                            ok = False
                            break
                if ok:
                    free.append(d)
            if free:
                res[S] = free
    return res


def keys_of(form, assign):
    """Returns the key set if the form is (multi-key) interlocking: the only
    mobile subsets are subsets of the keys and their complements.  Returns
    None if not interlocking, which includes the degenerate case where every
    part is a key -- that assembly simply falls apart, and the paper's
    multi-key model still requires the non-key parts to stay stuck."""
    mob = mobile_subsets(form, assign)
    keys = {S[0] for S in mob if len(S) == 1}
    if not keys or len(keys) == len(PARTS):
        return None
    for S in mob:
        Sset = set(S)
        comp = set(PARTS) - Sset
        if not (Sset <= keys or comp <= keys):
            return None
    return keys


def functional_ok(form, assign):
    """Functional stability, as in the paper's ladder/stool: in the ladder no
    rung half joint may release downward (stepping load); in the table the
    top rail R1 may not release downward (a load on the top)."""
    wd = world_dirs(form, assign)
    if form.startswith("form A"):
        return all(wd[hj] != "-z" for hj in HALF if hj[0] in ("S1", "S2"))
    return all(wd[hj] != "-z" for hj in HALF if hj[0] == "R1")


# ------------------------------------------------------------ enumeration
forms = list(FORMS)
# Nested filters, applied in this order, counted twice: once for form A on its
# own (the single-design problem of Fu 2015) and once for both forms at once
# (this paper's problem).
FUNNEL = ["joints mate (compatible)", "+ interlocking", "+ functionally stable",
          "+ exactly one key"]
TOTAL = len(DIR) ** len(HALF)


def enumerate_assignments():
    counts = {k: [0, 0] for k in FUNNEL}   # [form A alone, both forms]
    solutions = []
    for combo in product(DIR, repeat=len(HALF)):
        assign = dict(zip(HALF, combo))
        cA = compatible(forms[0], assign)
        cB = compatible(forms[1], assign)
        if not cA:
            continue
        both = cA and cB
        counts[FUNNEL[0]][0] += 1
        counts[FUNNEL[0]][1] += both
        kA = keys_of(forms[0], assign)
        if kA is None:
            continue
        kB = keys_of(forms[1], assign) if both else None
        both = both and kB is not None
        counts[FUNNEL[1]][0] += 1
        counts[FUNNEL[1]][1] += both
        fsA = functional_ok(forms[0], assign)
        if not fsA:
            continue
        both = both and functional_ok(forms[1], assign)
        counts[FUNNEL[2]][0] += 1
        counts[FUNNEL[2]][1] += both
        if both:
            solutions.append((assign, kA, kB, len(kA) == 1 and len(kB) == 1, True))
        if len(kA) != 1:
            continue
        counts[FUNNEL[3]][0] += 1
        counts[FUNNEL[3]][1] += both and len(kB) == 1
    return counts, solutions


pose_results = []
for flags in product((True, False), repeat=4):
    FORMS[forms[1]] = form_b(*flags)
    MATES.clear()
    MATES.update({f: mates(f) for f in FORMS})
    counts, solutions = enumerate_assignments()
    pose_results.append((flags, counts, solutions))
    print(flags, [counts[k] for k in FUNNEL])
n_pose_ok = sum(1 for _, c, _ in pose_results if c[FUNNEL[2]][1] > 0)
n_pose_single = sum(1 for _, c, _ in pose_results if c[FUNNEL[3]][1] > 0)
print(f"{n_pose_ok} of 16 pose choices admit an interlocking, functionally stable "
      f"assignment for both forms; {n_pose_single} admit a single-key one")
# keep the pose choice with the most complete solutions (first on ties)
flags, counts, solutions = max(pose_results, key=lambda r: (r[1][FUNNEL[3]][1], r[1][FUNNEL[2]][1]))
FORMS[forms[1]] = form_b(*flags)
MATES.clear()
MATES.update({f: mates(f) for f in FORMS})
print("chosen poses (R1 a-left, R2 a-left, S1 l-bottom, S2 l-bottom):", flags)
for f, m in MATES.items():
    print(f, m)
print(f"{TOTAL:6d}  all direction assignments")
for k, v in counts.items():
    print(f"{v[0]:6d} {v[1]:6d}  {k}")

# components of the half-joint graph (edges = mates in any form)
adj = {h: set() for h in HALF}
for f in forms:
    for h1, h2 in MATES[f]:
        adj[h1].add(h2)
        adj[h2].add(h1)
comps = []
seen = set()
for h in HALF:
    if h in seen:
        continue
    stack, comp = [h], set()
    while stack:
        u = stack.pop()
        if u in comp:
            continue
        comp.add(u)
        stack.extend(adj[u] - comp)
    seen |= comp
    comps.append(comp)
print("half-joint graph components:", comps)

# `solutions` holds the assignments that interlock BOTH forms and are
# functionally stable in both; prefer a single-key one if any exists.
best = [s for s in solutions if s[3]] or solutions
best.sort(key=lambda s: (len(s[1]) + len(s[2]), tuple(s[0][h] for h in HALF)))
assign, kA, kB, single, fs = best[0]
print("shown assignment:", assign, "keys A:", kA, "keys B:", kB, "single:", single, "functional:", fs)
print("form A mobile subsets:", mobile_subsets(forms[0], assign))
print("form B mobile subsets:", mobile_subsets(forms[1], assign))


# ------------------------------------------------------------ figure 1
def draw_form(ax, form, assign, keys, title):
    wd = world_dirs(form, assign)
    fill = {"R1": WOOD, "R2": WOOD, "S1": WOOD2, "S2": WOOD2}
    for p in PARTS:
        T, t = FORMS[form][p]
        w, h = LOCAL_RECT[p]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        wc = (T @ corners.T).T + t
        ax.add_patch(plt.Polygon(wc, closed=True, facecolor=fill[p], edgecolor=INK2, lw=0.9,
                                 zorder=2 if p.startswith("R") else 3, alpha=0.95))
        c = wc.mean(axis=0)
        # label away from the corners
        ax.text(c[0], c[1], p, ha="center", va="center", fontsize=9, fontweight="semibold",
                color=INK, zorder=6)
        # 'a'/'l' end marker: a small notch so the reader can see the pose
        e = T @ np.array([0.5, 3.5] if p.startswith("R") else [0.5, 0.5]) + t
    # half joints: at each mate, print both parts' world directions
    for h1, h2 in MATES[form]:
        pos = world_hj(form, h1)
        ax.add_patch(plt.Rectangle(pos - 0.5, 1, 1, facecolor="none", edgecolor=INK, lw=1.2,
                                   zorder=5, linestyle=(0, (2, 1.5))))
        # place text outside the frame corner
        dx = -1 if pos[0] < 2 else 1
        dz = 1 if pos[1] > 2 else -1
        txt = f"{h1[0]} {ARROW[wd[h1]]}{wd[h1]}\n{h2[0]} {ARROW[wd[h2]]}{wd[h2]}"
        ax.text(pos[0] + 1.05 * dx, pos[1] + 0.95 * dz, txt, ha="right" if dx < 0 else "left",
                va="center", fontsize=7.4, color=INK2, zorder=7)
    # keys: arrow starting at the part's edge so the label lands clear of the frame
    for k in sorted(keys):
        T, t = FORMS[form][k]
        w, h = LOCAL_RECT[k]
        corners = (T @ np.array([[0, 0], [w, 0], [w, h], [0, h]]).T).T + t
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        hjs = [hj for hj in HALF if hj[0] == k]
        d = DIR[wd[hjs[0]]]
        anchor = (lo + hi) / 2.0 + d * ((hi - lo) / 2.0)
        ax.annotate("", xy=anchor + 1.5 * d, xytext=anchor + 0.1 * d,
                    arrowprops=dict(arrowstyle="-|>", color=GOOD, lw=2.5, mutation_scale=16),
                    zorder=8)
        ax.text(*(anchor + 1.75 * d), f"key {k}\nexits {wd[hjs[0]]}",
                ha="center" if d[0] == 0 else ("left" if d[0] > 0 else "right"),
                va="bottom" if d[1] > 0 else ("top" if d[1] < 0 else "center"),
                fontsize=7.6, color=INK, zorder=8)
    ax.set_xlim(-3.4, 7.4)
    ax.set_ylim(-2.9, 6.6)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_xticks(range(0, 5)); ax.set_yticks(range(0, 5))
    clean(ax)
    ax.set_title(title, fontsize=9)


fig = plt.figure(figsize=(8, 4.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.75], wspace=0.35, left=0.05, right=0.99,
                      top=0.8, bottom=0.12)
for i, f in enumerate(forms):
    ax = fig.add_subplot(gs[i])
    keys = kA if i == 0 else kB
    draw_form(ax, f, assign, keys, f + f"  —  keys: {', '.join(sorted(keys))}")

# half-joint graph
ax = fig.add_subplot(gs[2])
pos = {}
# rails' half joints on the left column, rungs' on the right column
left = [h for h in HALF if h[0].startswith("R")]
right = [h for h in HALF if h[0].startswith("S")]
for i, h in enumerate(left):
    pos[h] = (0, 3 - i)
for i, h in enumerate(right):
    pos[h] = (1.6, 3 - i)
drawn = set()
for f, col, ls in zip(forms, (BLUE, ORANGE), ("-", "-")):
    for h1, h2 in MATES[f]:
        key = frozenset((h1, h2))
        off = 0.0 if key not in drawn else 0.08  # offset a repeated edge so both show
        drawn.add(key)
        (x1, y1), (x2, y2) = pos[h1], pos[h2]
        ax.plot([x1, x2], [y1 + off, y2 + off], color=col, lw=2, ls=ls, zorder=1)
for h, (x, y) in pos.items():
    ax.add_patch(plt.Circle((x, y), 0.17, facecolor=SURFACE, edgecolor=INK2, lw=1, zorder=3))
    ax.text(x, y, f"{h[0]}{h[1]}", ha="center", va="center", fontsize=6.6, color=INK, zorder=4)
ax.plot([], [], color=BLUE, lw=2, label="mate in form A")
ax.plot([], [], color=ORANGE, lw=2, label="mate in form B")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), fontsize=7.5, ncol=1)
ax.set_xlim(-0.6, 2.2)
ax.set_ylim(-0.6, 3.6)
ax.set_axis_off()
ax.set_title(f"half-joint graph:\n{len(comps)} components, so only {len(comps)} free\n"
             f"choices for {len(HALF)} half joints", fontsize=9)
fig.suptitle("Reusing the parts ties the half joints together: both forms interlock, "
             "but only with two keys each", fontsize=10.5, fontweight="semibold", y=0.97)
save(fig, "song2017-reconfigurable-interlocking-furniture-1.png")

# ------------------------------------------------------------ figure 2
labels = ["all direction\nassignments"] + list(counts)
vals_a = [TOTAL] + [counts[k][0] for k in counts]
vals_b = [TOTAL] + [counts[k][1] for k in counts]
fig, ax = plt.subplots(figsize=(8, 3.4))
y = np.arange(len(labels))[::-1]
hh = 0.34
floor = 0.55
ax.barh(y + hh / 2, [max(v, floor) for v in vals_a], color=BLUE, height=hh, zorder=2,
        label="form A alone (one design)")
ax.barh(y - hh / 2, [max(v, floor) for v in vals_b], color=ORANGE, height=hh, zorder=2,
        label="both forms, one shared part set")
for yi, va, vb in zip(y, vals_a, vals_b):
    ax.text(max(va, floor) * 1.3, yi + hh / 2, f"{va:,}", va="center", fontsize=8, color=INK)
    ax.text(max(vb, floor) * 1.3, yi - hh / 2, f"{vb:,}", va="center", fontsize=8, color=INK)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xscale("log")
ax.set_xlim(0.5, 4e6)
ax.set_xlabel("direction assignments surviving the filter (count, log scale)")
ax.legend(loc="lower right", fontsize=8)
clean(ax)
ax.set_title(f"Sharing one part set costs the single key: {vals_a[-1]} single-key assignments "
             f"for form A alone, {vals_b[-1]} for both", fontsize=9.5)
fig.subplots_adjust(left=0.22, right=0.97, top=0.88, bottom=0.18)
save(fig, "song2017-reconfigurable-interlocking-furniture-2.png")

# ------------------------------------------------------------ figure 3
# Table 1: set -> (common parts, half joints, T-co-D s, T-co-C s)
t1 = {"Ladder-Stool-Hand Truck": (11, 50, 24.8, 35.6),
      "Step Ladder-Chair": (9, 46, 0.4, 0.7),
      "Bookshelf 1-4": (8, 27, 2.7, 1.1),
      "Bed-Cot-Desk": (11, 44, 0.6, 0.5),
      "Bookshelf-Table-Chairs": (16, 70, 275.0, 493.8),
      "Office Box 1-3": (11, 64, 38.0, 272.2),
      "Shoe Rack-Laundry Box": (12, 60, 0.5, 2.3)}
# Table 2, baseline B2 (compatibility-consistent random), 1e7 trials:
# set -> (valid solutions, minutes)
t2 = {"Bookshelf 1-4": (715029, 159.8), "Step Ladder-Chair": (59, 192.0),
      "Bed-Cot-Desk": (26, 93.5), "Ladder-Stool-Hand Truck": (5, 690.4),
      "Bookshelf-Table-Chairs": (0, 783.5)}
order = sorted(t1, key=lambda k: t1[k][1])
x = np.arange(len(order))
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
ax = axes[0]
w = 0.38
ax.bar(x - w / 2, [t1[k][2] for k in order], w, color=BLUE, label="co-decomposition")
ax.bar(x + w / 2, [t1[k][3] for k in order], w, color=ORANGE, label="co-construction (joint planning)")
ax.set_yscale("log")
ax.set_ylim(0.1, 2000)
ax.set_ylabel("time (s)")
ax.set_xticks(x)
ax.set_xticklabels([f"{k} ({t1[k][1]})" for k in order], fontsize=7, rotation=28, ha="right")
ax.set_xlabel("reconfigurable set (Table 1), (half joints)")
ax.set_title("method: seconds to minutes", fontsize=9.5)
ax.legend(fontsize=7.5, loc="upper left")
clean(ax)
ax = axes[1]
order2 = sorted(t2, key=lambda k: t1[k][1])
x2 = np.arange(len(order2))
v = [t2[k][0] for k in order2]
ax.bar(x2, [max(vi, 0.6) for vi in v], 0.6, color=BLUE)
for xi, vi, k in zip(x2, v, order2):
    ax.text(xi, max(vi, 0.6) * 1.4, f"{vi:,}\n{t2[k][1]:.0f} min", ha="center", va="bottom",
            fontsize=7, color=INK)
ax.set_yscale("log")
ax.set_ylim(0.4, 3e7)
ax.set_ylabel("valid solutions (count)")
ax.set_xticks(x2)
ax.set_xticklabels([f"{k} ({t1[k][1]})" for k in order2], fontsize=7, rotation=28, ha="right")
ax.set_xlabel("reconfigurable set (Table 2, baseline B2), (half joints)")
ax.set_title("random search, 10⁷ trials: hours, and nothing\nfor the largest set", fontsize=9.5)
clean(ax)
fig.suptitle("The half-joint graph plus backward interlocking make the search tractable where "
             "random search is not", fontsize=10.5, fontweight="semibold")
fig.subplots_adjust(top=0.82, bottom=0.34, wspace=0.42, left=0.09, right=0.98)
save(fig, "song2017-reconfigurable-interlocking-furniture-3.png")
