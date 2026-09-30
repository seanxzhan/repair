"""Figures for docs/larsson2020-tsugite.md.

A vectorised re-implementation of Tsugite's validity metrics for 2-timber joints on
the default 3x3x3 grid, following the released code (setup/Evaluation.py):
connectivity, bridging, checkerboard, slidability (with fixed-side padding) and
durability.  Figure 1 evaluates them on four concrete designs; Figure 2 re-runs
the exhaustive enumeration of all 4^9 = 262,144 height fields per joint type and
compares with the paper's Table 1; Figure 3 charts the user study (Table 3).
Run from docs/figures/:  python larsson2020-tsugite.py
"""
from _style import *  # noqa: F401,F403

import itertools

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

np.random.seed(0)
DIM = 3
SAX = 2  # sliding / milling axis = z, as in the code's default

# Joint types: fixed sides per timber as (axis, dir).  Timber 0 fills z < h, timber 1 fills z >= h.
# The released app's default is [[(2,0)],[(2,1)]] (I-axial).  Others follow the paper's Figure 24.
TYPES = {
    "I-axial": [[(2, 0)], [(2, 1)]],
    "I-perp": [[(0, 0)], [(0, 1)]],
    "L-axial": [[(0, 0)], [(2, 1)]],
    "L-perp": [[(0, 0)], [(1, 0)]],
    "T-axial": [[(0, 0), (0, 1)], [(2, 1)]],
    "T-perp": [[(0, 0), (0, 1)], [(1, 0)]],
    "X-perp": [[(0, 0), (0, 1)], [(1, 0), (1, 1)]],
}
PAPER = {  # Table 1 of the paper (removed in this order; last row = valid)
    "I-axial": [0, 0, 145608, 28028, 0, 88508],
    "I-perp": [99152, 0, 68822, 19114, 74872, 184],
    "L-axial": [52884, 0, 104020, 20500, 76358, 8382],
    "L-perp": [99690, 0, 68390, 13506, 80504, 54],
    "T-axial": [7083, 35146, 114494, 13691, 23622, 68108],
    "T-perp": [59046, 31864, 76986, 6870, 84166, 3212],
    "X-perp": [7083, 54829, 104864, 0, 25214, 70154],
}
STAGES = ["connectivity", "bridging", "checkerboard", "slidability", "durability"]


# ----------------------------------------------------------------------------
# Batched geometry helpers.  V has shape (n, 3, 3, 3) with values 0/1 (timber id), index order (x, y, z).
# ----------------------------------------------------------------------------
def voxels_from_heights(H):
    z = np.arange(DIM)[None, None, None, :]
    return (z >= H[..., None]).astype(np.int8)  # 0 below the height, 1 above


def pad_fixed(V, fixed):
    """Pad one layer on each fixed side with that timber's id, -1 elsewhere (add_fixed_sides)."""
    n = V.shape[0]
    P = -np.ones((n, DIM + 2, DIM + 2, DIM + 2), dtype=np.int8)
    P[:, 1:-1, 1:-1, 1:-1] = V
    for t, sides in enumerate(fixed):
        for ax, d in sides:
            idx = [slice(1, -1)] * 3
            idx[ax] = 0 if d == 0 else DIM + 1
            P[(slice(None), *idx)] = t
    return P


def flood(mask, seed):
    """Batched 6-connected flood fill of `seed` inside `mask` (both bool, shape (n, a, b, c))."""
    reach = seed & mask
    while True:
        grow = reach.copy()
        for ax in (1, 2, 3):
            sl = [slice(None)] * 4
            sl2 = [slice(None)] * 4
            sl[ax] = slice(1, None); sl2[ax] = slice(None, -1)
            grow[tuple(sl)] |= reach[tuple(sl2)]
            grow[tuple(sl2)] |= reach[tuple(sl)]
        grow &= mask
        if np.array_equal(grow, reach):
            return reach
        reach = grow


def side_seed(shape, ax, d, n):
    s = np.zeros(shape, dtype=bool)
    idx = [slice(None)] * 3
    idx[ax] = 0 if d == 0 else DIM - 1
    s[(slice(None), *idx)] = True
    return s


def connectivity_ok(V, fixed):
    """Every voxel of each timber reachable from one of its fixed sides (flood through the timber)."""
    ok = np.ones(V.shape[0], dtype=bool)
    for t, sides in enumerate(fixed):
        mask = V == t
        seed = np.zeros_like(mask)
        for ax, d in sides:
            seed |= side_seed(V.shape, ax, d, V.shape[0])
        reach = flood(mask, seed)
        ok &= ~(mask & ~reach).any(axis=(1, 2, 3))
    return ok


def bridging_ok(V, fixed):
    """Timbers with two fixed sides: flood from side 1 must reach side 2 (an empty timber is unbridged)."""
    ok = np.ones(V.shape[0], dtype=bool)
    for t, sides in enumerate(fixed):
        if len(sides) < 2:
            continue
        mask = V == t
        reach = flood(mask, side_seed(V.shape, *sides[0], V.shape[0]))
        touches = (reach & side_seed(V.shape, *sides[1], V.shape[0])).any(axis=(1, 2, 3))
        ok &= touches
    return ok


def checkerboard_free(V):
    """No 2x2 pattern [[a,b],[b,a]] (a != b) around an interior vertex in any layer perpendicular to SAX."""
    a = V[:, :-1, :-1, :]; b = V[:, 1:, :-1, :]; c = V[:, :-1, 1:, :]; d = V[:, 1:, 1:, :]
    chess = (a == d) & (b == c) & (a != b)
    return ~chess.any(axis=(1, 2, 3))


def free_directions(V, fixed):
    """Per timber, the list of (axis, dir) it can slide in, after padding with fixed sides.
    Blocked in +ax if, walking +ax along any column, some other timber's voxel comes after one of its own."""
    P = pad_fixed(V, fixed)
    out = []
    for t in range(2):
        mine = P == t
        other = (P != t) & (P != -1)
        dirs = []
        for ax in range(3):
            for d in (0, 1):
                m = mine if d == 1 else np.flip(mine, axis=ax + 1)
                o = other if d == 1 else np.flip(other, axis=ax + 1)
                seen = np.cumsum(m, axis=ax + 1) > 0  # own voxel at or before this index
                blocked = (seen & o).any(axis=(1, 2, 3))
                dirs.append(~blocked)
        out.append(np.array(dirs).T)  # (n, 6) bool: [x-, x+, y-, y+, z-, z+]
    return out


def slidability_ok(V, fixed):
    """Valid when each (end) timber has at most one free direction (Evaluation.update)."""
    fd = free_directions(V, fixed)
    return (fd[0].sum(axis=1) <= 1) & (fd[1].sum(axis=1) <= 1)


def flood2d(mask, seed):
    reach = seed & mask
    while True:
        grow = reach.copy()
        for ax in (1, 2):
            sl = [slice(None)] * 3; sl2 = [slice(None)] * 3
            sl[ax] = slice(1, None); sl2[ax] = slice(None, -1)
            grow[tuple(sl)] |= reach[tuple(sl2)]
            grow[tuple(sl2)] |= reach[tuple(sl)]
        grow &= mask
        if np.array_equal(grow, reach):
            return reach
        reach = grow


def breakable_voxels(V, fixed):
    """Durability (get_breakable_voxels).  Grain axis = axis of the timber's fixed side(s).
    Skipped if grain == SAX.  For each axis perpendicular to the grain, slice the grid into layers;
    a 2D region of the timber that does not reach a fixed side is 'potentially fragile'; it is flagged
    unless non-fragile material of the same timber anchors it in BOTH neighbouring layers."""
    n = V.shape[0]
    flagged = np.zeros((2, n, DIM, DIM, DIM), dtype=bool)
    for t, sides in enumerate(fixed):
        gax = sides[0][0]
        if gax == SAX:
            continue
        mine = V == t
        for pax in [a for a in range(3) if a != gax]:
            # move pax to the front so layers are index 1 of (n, layer, i, j)
            M = np.moveaxis(mine, pax + 1, 1)
            axes2d = [a for a in range(3) if a != pax]  # the two in-layer axes, in order
            g2 = axes2d.index(gax)
            PF = np.zeros_like(M)
            for l in range(DIM):
                lay = M[:, l]
                seed = np.zeros_like(lay)
                for ax, d in sides:
                    idx = [slice(None)] * 2
                    idx[g2] = 0 if d == 0 else DIM - 1
                    seed[(slice(None), *idx)] = True
                PF[:, l] = lay & ~flood2d(lay, seed)
            anchor = M & ~PF
            F = np.zeros_like(M)
            for l in range(DIM):
                pf = PF[:, l]
                up = anchor[:, l + 1] if l + 1 < DIM else np.zeros_like(pf)
                dn = anchor[:, l - 1] if l - 1 >= 0 else np.zeros_like(pf)
                # region-level OR: propagate 'has anchored neighbour' through the PF region
                a_up = flood2d(pf, pf & up)
                a_dn = flood2d(pf, pf & dn)
                F[:, l] = pf & ~(a_up & a_dn)
            flagged[t] |= np.moveaxis(F, 1, pax + 1)
    return flagged  # (2, n, 3, 3, 3)


def durability_ok(V, fixed):
    return ~breakable_voxels(V, fixed).any(axis=(0, 2, 3, 4))


def contact_and_friction(V, fixed, fd):
    """Face counts for timber 0 of a single design (index 0 of V): faces touching timber 1's voxels or
    timber 1's body beyond its fixed side; friction faces are those not perpendicular to a free direction."""
    P = pad_fixed(V[:1], fixed)[0]
    mine = P == 0
    other = P == 1
    contact = 0; friction = 0
    free_axes = {ax for ax in range(3) if fd[0][0, 2 * ax] or fd[0][0, 2 * ax + 1]}
    for ax in range(3):
        for d in (-1, 1):
            sh = np.roll(other, -d, axis=ax)  # neighbour in direction d
            faces = (mine & sh).sum()
            contact += faces
            if ax not in free_axes:
                friction += faces
    return int(contact), int(friction)


def evaluate_type(name):
    """Sequential failure counts as in Table 1."""
    fixed = TYPES[name]
    H = np.array(list(itertools.product(range(DIM + 1), repeat=DIM * DIM)), dtype=np.int8).reshape(-1, DIM, DIM)
    V = voxels_from_heights(H)
    alive = np.ones(len(V), dtype=bool)
    counts = []
    for stage, fn in [("connectivity", connectivity_ok), ("bridging", bridging_ok),
                      ("checkerboard", lambda v, f: checkerboard_free(v)),
                      ("slidability", slidability_ok), ("durability", durability_ok)]:
        ok = np.ones(len(V), dtype=bool)
        idx = np.flatnonzero(alive)
        ok[idx] = fn(V[idx], fixed)
        counts.append(int((alive & ~ok).sum()))
        alive &= ok
    counts.append(int(alive.sum()))
    return counts, H[alive]


# ----------------------------------------------------------------------------
# Figure 1: four concrete I-perp designs with every metric evaluated
# ----------------------------------------------------------------------------
def draw_voxels(ax, V, flagged, title, lines):
    filled = np.ones((DIM, DIM, DIM), dtype=bool)
    colors = np.empty(V.shape, dtype=object)
    colors[V == 0] = WOOD
    colors[V == 1] = WOOD2
    colors[flagged] = ORANGE
    ax.voxels(filled, facecolors=colors, edgecolor=INK2, linewidth=0.5)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=-58)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel("x", labelpad=-12, fontsize=8, color=INK2)
    ax.set_ylabel("y", labelpad=-12, fontsize=8, color=INK2)
    ax.set_zlabel("z", labelpad=-12, fontsize=8, color=INK2)
    ax.set_title(title, fontsize=9, pad=0)
    ax.text2D(0.02, 0.0, lines, transform=ax.transAxes, fontsize=7, color=INK2, ha="left", va="top", family="DejaVu Sans Mono")


def figure1(valid_iperp):
    fixed = TYPES["I-perp"]  # timber A body at -x, B body at +x, grain x, slide z
    designs = [
        ("half lap", np.array([[3, 3, 3], [3, 3, 3], [0, 0, 0]])),          # A keeps x<2, B x=2
        ("checkerboard", np.array([[3, 0, 3], [0, 3, 0], [3, 0, 3]])),
        ("tongue + lip (nondurable)", np.array([[3, 3, 3], [1, 1, 1], [2, 2, 2]])),
        ("valid (from search)", valid_iperp[0]),
    ]
    fig = plt.figure(figsize=(7.8, 3.9))
    names = ["x−", "x+", "y−", "y+", "z−", "z+"]
    for i, (name, H) in enumerate(designs):
        V = voxels_from_heights(H[None])
        fd = free_directions(V, fixed)
        conn = connectivity_ok(V, fixed)[0]
        chess = not checkerboard_free(V)[0]
        flagged = breakable_voxels(V, fixed)
        nflag = int(flagged.sum())
        slid_ok = slidability_ok(V, fixed)[0]
        contact, friction = contact_and_friction(V, fixed, fd)
        fa = [names[k] for k in np.flatnonzero(fd[0][0])]
        fb = [names[k] for k in np.flatnonzero(fd[1][0])]
        valid = conn and (not chess) and slid_ok and nflag == 0
        fails = [s for s, bad in [("connectivity", not conn), ("checkerboard", chess),
                                  ("slidability", not slid_ok), ("durability", nflag > 0)] if bad]
        lines = ("A slides: %s\nB slides: %s\nchecker %s, flagged %d\ncontact %d, friction %d\n%s"
                 % (" ".join(fa) or "none", " ".join(fb) or "none", "yes" if chess else "no", nflag,
                    contact, friction, "VALID" if valid else "FAILS " + "\n      ".join(fails)))
        print("Fig1", name, "heights\n", H, "\n", lines)
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        draw_voxels(ax, V[0], flagged[:, 0].any(axis=0), name, lines)
    handles = [Patch(facecolor=WOOD, edgecolor=INK2, label="timber A (body at −x)"),
               Patch(facecolor=WOOD2, edgecolor=INK2, label="timber B (body at +x)"),
               Patch(facecolor=ORANGE, edgecolor=INK2, label="nondurable voxel")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01), fontsize=8)
    fig.suptitle("Perpendicular I-joint (grain along x, slide along z): only one of these four passes every check",
                 fontsize=10.5, fontweight="semibold", y=0.99)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.3, wspace=0.05)
    save(fig, "larsson2020-tsugite-1.png")


# ----------------------------------------------------------------------------
# Figure 2: exhaustive enumeration vs Table 1
# ----------------------------------------------------------------------------
def figure2(results):
    names = list(TYPES)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6), gridspec_kw={"width_ratios": [1.4, 1.1]})
    ax = axes[0]
    clean(ax)
    y = np.arange(len(names))
    left = np.zeros(len(names))
    cols = [SEQ[1], SEQ[2], SEQ[3], SEQ[4], SEQ[6]]
    for j, stage in enumerate(STAGES):
        vals = np.array([PAPER[n][j] for n in names]) / 262144 * 100
        ax.barh(y, vals, left=left, color=cols[j], label="fails " + stage, height=0.62)
        left += vals
    valid = np.array([PAPER[n][5] for n in names]) / 262144 * 100
    ax.barh(y, valid, left=left, color=ORANGE, label="valid", height=0.62)
    for yi, n in zip(y, names):
        ax.text(101, yi, "{:,} valid".format(PAPER[n][5]), va="center", fontsize=8, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of the 262,144 height fields (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3, fontsize=7.5)
    ax.set_title("Paper's Table 1: durability is the big filter\nfor every perpendicular type", fontsize=9.5)

    ax = axes[1]
    clean(ax)
    paper_all = np.array([PAPER[n] for n in names], dtype=float)
    ours_all = np.array([results[n] for n in names], dtype=float)
    pv, ov = paper_all[:, 5], ours_all[:, 5]
    ax.hlines(y, np.minimum(pv, ov), np.maximum(pv, ov), color=GRID, lw=3, zorder=1)
    ax.plot(pv, y, "o", color=BLUE, ms=6, label="paper, Table 1", zorder=3)
    ax.plot(ov, y, "x", color=ORANGE, ms=7, mew=2, label="our re-run (this figure)", zorder=4)
    for yi, p_, o_ in zip(y, pv, ov):
        ax.text(3.2e5, yi, "{:,}".format(int(p_)) if p_ == o_ else "{:,} vs {:,}".format(int(p_), int(o_)),
                va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(10, 3e5)
    ax.set_xlabel("valid designs (log scale)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), fontsize=7.5)
    nrows = int((paper_all == ours_all).all(axis=1).sum())
    ax.set_title("Re-run of the search: %d of 7 columns\nreproduced exactly (X-perp differs)" % nrows, fontsize=9.5)
    fig.tight_layout()
    save(fig, "larsson2020-tsugite-2.png")


# ----------------------------------------------------------------------------
# Figure 3: user study, Table 3
# ----------------------------------------------------------------------------
def figure3():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    clean(ax)
    x = np.arange(2); wd = 0.36
    ax.bar(x - wd / 2, [8, 1], wd, color=BLUE, label="with feedback (n = 10)")
    ax.bar(x + wd / 2, [13, 8], wd, color=ORANGE, label="without feedback (n = 10)")
    for xi, vals in zip(x, [(8, 13), (1, 8)]):
        ax.text(xi - wd / 2, vals[0] + 0.3, str(vals[0]), ha="center", fontsize=8.5, color=INK2)
        ax.text(xi + wd / 2, vals[1] + 0.3, str(vals[1]), ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(["all errors", "errors excluding\ndurability"])
    ax.set_ylabel("total error count (2 tasks × 10 people)")
    ax.set_ylim(0, 19)
    ax.legend(loc="upper right")
    ax.set_title("Feedback removed almost every error\nexcept durability", fontsize=9.5)
    ax = axes[1]
    clean(ax)
    ax.bar([0, 1], [138, 173], 0.6, color=[BLUE, ORANGE])
    for xi, v, lab in zip([0, 1], [138, 173], ["2 min 18 s", "2 min 53 s"]):
        ax.text(xi, v + 3, lab, ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["with\nfeedback", "without\nfeedback"])
    ax.set_ylabel("average time per task (s)")
    ax.set_ylim(0, 200)
    ax.set_title("and was faster", fontsize=9.5)
    fig.tight_layout()
    save(fig, "larsson2020-tsugite-3.png")


if __name__ == "__main__":
    results = {}
    valid_iperp = None
    for name in TYPES:
        counts, valid = evaluate_type(name)
        results[name] = counts
        print("%-8s ours  %s\n         paper %s" % (name, counts, PAPER[name]))
        if name == "I-perp":
            valid_iperp = valid
    figure1(valid_iperp)
    figure2(results)
    figure3()
