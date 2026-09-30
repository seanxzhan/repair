"""Figures for proposal.md.

Everything here is computed. The input is one timber block with a rotten end.
The output is an *interface*: the mating surface along which the rotten part is
cut away and new wood is fitted. Nothing about the interface is given up front;
its shape is the design variable.

The model lives in `repair.proto2d.model` (shared with `examples/proto2d/inspect_splice.py`,
the interactive version of this figure). We use one interface family, a
mortise-and-tenon splice, over a 2D parameter slice: `a` = where the shoulder
sits, `ell` = how deep the tenon reaches. At each point the contact set is
rebuilt against the damage field and a 2D frictional-equilibrium LP gives the
largest bending moment the interface can transfer. Moment, not a point load,
because a couple has no lever arm to confound shape with position; it is also
what joint-strength tests measure (janikova2025).

The point of the figure: the interface parameters are continuous, the contact
set is not, and the objective's optimum sits exactly on a contact-topology
boundary.

Run from docs/figures/:  python proposal.py
"""
from _style import *  # noqa: F401,F403

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Polygon as MplPolygon
from scipy.ndimage import gaussian_filter1d

try:
    from repair.proto2d import model as sp
except ImportError:                       # not installed: fall back to the source tree
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from repair.proto2d import model as sp

SLUG = "proposal"

BLOCK = sp.Block()
DAMAGE = sp.Damage()
STATICS = sp.Statics()
SPLICE0 = sp.Splice()                     # cheeks and sampling; a and ell are swept
AREA = sp.Grid(BLOCK)                     # for sound-wood-removed integrals

A_GRID = np.linspace(8.2, 10.6, 121)      # shoulder position
L_GRID = np.linspace(0.0, 3.0, 76)        # tenon length
LAMBDA = 1.5                              # weight on sound wood removed, a design choice
LAMBDAS = (0.55, 1.0, 1.5, 2.0, 2.5, 3.0)  # the sweep reported at the end
CLIFF_TOL = 0.03                          # a jump this fraction of peak M is a cliff


def splice(a, ell):
    return replace(SPLICE0, a=float(a), ell=float(ell))


def contacts(a, ell):
    return sp.contact_set(splice(a, ell), DAMAGE, BLOCK)


def removed(a, ell):
    return AREA.sound_removed(splice(a, ell), DAMAGE)


# ------------------------------------------------------------------ drawing
DMG_CMAP = LinearSegmentedColormap.from_list("rot", [WOOD, WOOD2, BAD])
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", SEQ)
LEN, HGT = BLOCK.length, BLOCK.height
Y1, Y2 = SPLICE0.y1, SPLICE0.y2


def patch_polygon(a, ell):
    return np.array([[a, 0.0], [LEN, 0.0], [LEN, HGT], [a, HGT], [a, Y2],
                     [a - ell, Y2], [a - ell, Y1], [a, Y1]])


def retained_polygon(a, ell):
    return np.array([[0.0, 0.0], [a, 0.0], [a, Y1], [a - ell, Y1],
                     [a - ell, Y2], [a, Y2], [a, HGT], [0.0, HGT]])


def draw_block(ax):
    """The input: one block, partly rotten, no interface yet."""
    gx, gy = np.meshgrid(np.linspace(0, LEN, 400), np.linspace(0, HGT, 80))
    ax.imshow(DAMAGE.severity(gx, gy, BLOCK), extent=(0, LEN, 0, HGT), origin="lower",
              cmap=DMG_CMAP, vmin=0, vmax=1, aspect="auto", zorder=0)
    ys = np.linspace(0, HGT, 40)
    ax.plot(DAMAGE.front_x(ys, BLOCK), ys, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=3)
    ax.add_patch(MplPolygon([[0, 0], [LEN, 0], [LEN, HGT], [0, HGT]], closed=True,
                            facecolor="none", edgecolor=INK, lw=1.4, zorder=4))
    ax.annotate("sound", (6.6, 1.0), ha="center", va="center", fontsize=8.5, color=INK2)
    ax.annotate("rotten", (10.9, 1.0), ha="center", va="center", fontsize=8.5, color=SURFACE)


def draw_interface(ax, a, ell, title):
    """One candidate output drawn on the input block."""
    draw_block(ax)
    for txt in ax.texts:
        txt.remove()
    ax.add_patch(MplPolygon(patch_polygon(a, ell), closed=True, facecolor=SURFACE,
                            edgecolor=INK2, lw=1.0, alpha=0.95, zorder=2))
    ax.add_patch(MplPolygon(retained_polygon(a, ell), closed=True, facecolor="none",
                            edgecolor=INK, lw=1.4, zorder=4))
    cs = contacts(a, ell)
    for c in cs:
        ax.plot(*c.point, "o", ms=3.4, color=(GOOD if c.live else BAD),
                mec=SURFACE, mew=0.5, zorder=6)
    ax.add_patch(FancyArrowPatch((11.05, 0.62), (11.05, 1.38), zorder=7,
                                 connectionstyle="arc3,rad=0.45", color=INK,
                                 arrowstyle="-|>", mutation_scale=11, lw=1.3))
    ax.annotate("M", (11.7, 1.0), fontsize=10, color=INK, ha="center", va="center", zorder=7)
    nl = sum(c.live for c in cs)
    ax.set_title(title, fontsize=10, pad=36)
    ax.text(0.0, 1.03,
            f"a = {a:.1f}, $\\ell$ = {ell:.2f}, {nl}/{len(cs)} contacts live\n"
            f"M$_{{max}}$ = {sp.max_moment(cs, STATICS):.0f}, sound wood removed {removed(a, ell):.2f}",
            fontsize=8, color=INK, transform=ax.transAxes, va="bottom")


def frame(ax):
    ax.set_xlim(4.2, LEN + 0.3); ax.set_ylim(-0.15, HGT + 0.6)
    clean(ax, hide_axes=True)


def main():
    two, want = sp.sanity_checks(STATICS.cap)
    print(f"checks ok: opposing pair {two:.1f} vs closed form {want:.1f}, one face 0, plain cut 0")

    M, R = sp.landscape(A_GRID, L_GRID, SPLICE0, DAMAGE, STATICS, AREA)
    obj = M / M.max() - LAMBDA * R / R.max()
    cliffs = sp.cliff_mask(M, CLIFF_TOL * M.max())
    bi, bj = np.unravel_index(np.argmax(obj), obj.shape)
    a_star, l_star = A_GRID[bj], L_GRID[bi]
    ci, cj = np.nonzero(cliffs)
    dist_cells = np.min(np.hypot(ci - bi, cj - bj))

    fig = plt.figure(figsize=(14.6, 7.4))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.0, 1.35], hspace=0.62, wspace=1.1)

    # -- row 1: the input, then three candidate outputs of different shape
    ax = fig.add_subplot(gs[0, 0:3]); draw_block(ax); frame(ax)
    ax.set_title("(a) input: one block, partly rotten", fontsize=10, pad=36)
    ax.text(0.0, 1.03, "a damage field over one piece of wood;\nthe interface is not given, it is the output",
            fontsize=8, color=INK2, transform=ax.transAxes, va="bottom")
    for k, (a, ell, ttl) in enumerate([(8.8, 0.0, "(b) plain cut"),
                                        (9.8, 1.0, "(c) short tenon, in rot"),
                                        (a_star, l_star, "(d) the optimum in (f)")]):
        ax = fig.add_subplot(gs[0, 3 * (k + 1):3 * (k + 2)])
        draw_interface(ax, a, ell, ttl); frame(ax)

    ext = (A_GRID[0], A_GRID[-1], L_GRID[0], L_GRID[-1])
    ink_mask = LinearSegmentedColormap.from_list("ink", [(0, 0, 0, 0), INK])

    # -- (e) capacity over the interface parameters, with the active-set boundaries
    ax = fig.add_subplot(gs[1, 0:4])
    im = ax.imshow(M, extent=ext, origin="lower", aspect="auto", cmap=SEQ_CMAP)
    ax.imshow(np.where(cliffs, 1.0, 0.0), extent=ext, origin="lower", aspect="auto",
              cmap=ink_mask, vmin=0, vmax=1, alpha=0.55, interpolation="nearest")
    ax.plot([a_star], [l_star], "*", ms=15, color=AQUA, mec=INK, mew=0.6, zorder=6)
    ax.set_xlabel("shoulder position a"); ax.set_ylabel("tenon length $\\ell$")
    ax.set_title("(e) capacity over interface shapes")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label("M$_{max}$", fontsize=9)
    cb.outline.set_visible(False)
    ax.grid(False)

    # -- (f) the objective, whose optimum lands on a boundary
    ax = fig.add_subplot(gs[1, 4:8])
    ax.imshow(obj, extent=ext, origin="lower", aspect="auto", cmap=SEQ_CMAP)
    ax.imshow(np.where(cliffs, 1.0, 0.0), extent=ext, origin="lower", aspect="auto",
              cmap=ink_mask, vmin=0, vmax=1, alpha=0.55, interpolation="nearest")
    ax.plot([a_star], [l_star], "*", ms=15, color=AQUA, mec=INK, mew=0.6, zorder=6)
    ax.annotate(f"optimum (a, $\\ell$) = ({a_star:.2f}, {l_star:.2f}),\n"
                f"{dist_cells:.0f} grid cell{'s' if dist_cells != 1 else ''} from a cliff",
                (a_star, l_star), textcoords="offset points", xytext=(-118, 96),
                fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9),
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.92))
    ax.axhline(l_star, color=SURFACE, lw=0.9, ls=(0, (3, 3)), zorder=5)
    ax.set_xlabel("shoulder position a"); ax.set_yticklabels([])
    ax.set_title(f"(f) capacity − {LAMBDA:.2f}·(sound wood removed)")
    ax.grid(False)                            # the objective is normalized; no colorbar

    # -- (g) one slice through the optimum: what a frozen-active-set gradient sees
    ax = fig.add_subplot(gs[1, 8:12])
    row = M[bi]                                # M along a, at the optimal tenon length
    edges = A_GRID[1:][np.abs(np.diff(row)) > CLIFF_TOL * M.max()]
    for b in edges:
        ax.axvline(b, color=GRID, lw=1.0, zorder=0)
    ax.plot(A_GRID, row, color=BLUE, lw=2.2, zorder=3, label="capacity (LP)")
    j = int(np.argmin(np.abs(A_GRID - 8.7)))   # inside the first regime
    g = (row[j + 1] - row[j - 1]) / (A_GRID[j + 1] - A_GRID[j - 1])
    xs = np.linspace(A_GRID[j] - 0.4, A_GRID[j] + 1.7, 40)
    ax.plot(xs, row[j] + g * (xs - A_GRID[j]), color=ORANGE, lw=1.8, ls=(0, (5, 3)),
            zorder=4, label=f"frozen-active-set gradient ({abs(g):.0f})")
    ax.plot([A_GRID[j]], [row[j]], "o", color=ORANGE, zorder=5)
    smooth = gaussian_filter1d(row, sigma=5, mode="nearest")
    ax.plot(A_GRID, smooth, color=AQUA, lw=1.8, ls=(0, (1, 1.6)), zorder=3,
            label="smooth surrogate (target)")
    ax.plot([a_star], [row[bj]], "*", ms=15, color=AQUA, mec=INK, mew=0.6, zorder=6)
    zero = A_GRID[np.argmax(row <= 1e-6)] if np.any(row <= 1e-6) else A_GRID[-1]
    ax.annotate(f"gradient says “free”,\nbut capacity is 0\nby a = {zero:.1f}",
                (edges[0], row.max()), textcoords="offset points", xytext=(28, -78),
                fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9))
    ax.set_xlabel(f"shoulder position a   (at $\\ell$ = {l_star:.2f})")
    ax.set_ylabel("max transferred moment")
    ax.set_title("(g) the slice through the optimum")
    ax.set_ylim(-8, row.max() * 1.32)
    ax.legend(loc="lower left")
    clean(ax)

    save(fig, f"{SLUG}-1.png")

    ncells = int(cliffs.sum())
    print(f"landscape: {M.shape[1]} x {M.shape[0]} interfaces, capacity 0 .. {M.max():.1f}")
    print(f"cliff cells: {ncells} of {M.size} ({100 * ncells / M.size:.0f}%)")
    print(f"optimum (a, ell) = ({a_star:.2f}, {l_star:.2f}); M = {M[bi, bj]:.1f}, "
          f"removed = {R[bi, bj]:.2f}; {dist_cells:.0f} cells from the nearest cliff")
    print(f"slice at ell = {l_star:.2f}: flat at {row[0]:.1f} to a = {edges[0]:.2f}, "
          f"{len(edges)} cliffs, zero by a = {zero:.2f}; frozen gradient at a = {A_GRID[j]:.2f}: {g:+.3f}")
    # The weight is a design choice, so report how the optimum moves with it.
    print("lambda sweep (a*, ell*, cells from nearest cliff, interior in ell?):")
    for lam in LAMBDAS:
        o = M / M.max() - lam * R / R.max()
        i, k = np.unravel_index(np.argmax(o), o.shape)
        d = np.min(np.hypot(ci - i, cj - k))
        print(f"  lambda={lam:4.2f}: ({A_GRID[k]:.2f}, {L_GRID[i]:.2f}), {d:.0f} cells, "
              f"{'interior' if 0 < i < len(L_GRID) - 1 else 'on the box edge'}")


if __name__ == "__main__":
    main()
