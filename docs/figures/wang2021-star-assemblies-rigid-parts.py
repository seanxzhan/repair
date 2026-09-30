"""Figures for docs/wang2021-star-assemblies-rigid-parts.md (run from docs/figures/).

1. Motion cone of a 2-part joint as the dual of its contact normals (Eq. 4 of the STAR),
   computed on three toy cross-sections: flat contact, housing (U-notch), dovetail.
2. Blocking graphs (DBG-based interlocking test) on a computed 3-part 2D assembly.
3. Tilt analysis with rigid-block equilibrium (Eqs. 1-3): critical tilt of a block found by
   linear programming, compared with the closed-form sliding / tipping limits.
4. Where the 53 papers of Table 1 sit on the objective x problem-framing grid.
"""
from _style import *  # noqa: F401,F403

import numpy as np
from matplotlib.patches import Polygon, FancyArrowPatch, Wedge, Circle
from scipy.optimize import linprog
from scipy.sparse.csgraph import connected_components

SLUG = "wang2021-star-assemblies-rigid-parts"
np.random.seed(0)


# ----------------------------------------------------------------------------------------
# Helpers: planar rigid motion (vx, vz, omega) and non-penetration rows  (v + w x r) . n >= 0
# ----------------------------------------------------------------------------------------
def rows(contacts):
    """contacts: list of (r, n). Returns matrix B with B @ (vx, vz, w) >= 0 meaning no penetration.
    In the plane, w x r = (-w r_z, w r_x)."""
    B = []
    for (rx, rz), (nx, nz) in contacts:
        B.append([nx, nz, -rz * nx + rx * nz])
    return np.array(B)


def free_directions(B, ntheta=3600):
    """Pure translations (vx, vz) = (cos t, sin t) that satisfy every row (omega = 0)."""
    t = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    V = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], 1)
    ok = np.all(V @ B.T >= -1e-12, axis=1)
    return t, ok


def rotation_possible(B):
    """LP: is there a motion with omega = +-1 that violates no row?  (fix |omega| = 1, free v)"""
    for s in (1.0, -1.0):
        # variables vx, vz ; constraint B[:, :2] v + B[:, 2] s >= 0  ->  -B[:, :2] v <= B[:, 2] s
        res = linprog(c=[0, 0], A_ub=-B[:, :2], b_ub=B[:, 2] * s, bounds=[(None, None)] * 2, method="highs")
        if res.status == 0:
            return True
    return False


# Three toy joints, all cross-sections of a prism swept along y (like an LHF cut), in mm.
W_B, H_B = 30.0, 24.0  # block B: width, height
D = 12.0  # notch depth
FLARE = 6.0  # dovetail: bottom is wider than the top by 2*FLARE

def joint_geometry(kind):
    """Returns (A polygon, B polygon, contacts list) with B's reference point at its bottom centre."""
    if kind == "flat":
        A = [(-40, -12), (40, -12), (40, 0), (-40, 0)]
        B = [(-W_B / 2, 0), (W_B / 2, 0), (W_B / 2, H_B), (-W_B / 2, H_B)]
        contacts = [((-W_B / 2, 0), (0, 1)), ((W_B / 2, 0), (0, 1))]
    elif kind == "housing":
        A = [(-40, -12), (40, -12), (40, D), (W_B / 2, D), (W_B / 2, 0), (-W_B / 2, 0), (-W_B / 2, D), (-40, D)]
        B = [(-W_B / 2, 0), (W_B / 2, 0), (W_B / 2, H_B), (-W_B / 2, H_B)]
        contacts = [((-W_B / 2, 0), (0, 1)), ((W_B / 2, 0), (0, 1)),
                    ((-W_B / 2, 0), (1, 0)), ((-W_B / 2, D), (1, 0)),
                    ((W_B / 2, 0), (-1, 0)), ((W_B / 2, D), (-1, 0))]
    else:  # dovetail
        wb, wt = W_B / 2 + FLARE, W_B / 2  # half-widths at the bottom and at the notch top
        A = [(-40, -12), (40, -12), (40, D), (wt, D), (wb, 0), (-wb, 0), (-wt, D), (-40, D)]
        B = [(-wb, 0), (wb, 0), (wt, D), (wt, H_B), (-wt, H_B), (-wt, D)]
        nl = np.array([D, -(wb - wt)]); nl /= np.linalg.norm(nl)  # left wall normal, into B: +x and down
        nr = np.array([-D, -(wb - wt)]); nr /= np.linalg.norm(nr)
        contacts = [((-wb, 0), (0, 1)), ((wb, 0), (0, 1)),
                    ((-wb, 0), tuple(nl)), ((-wt, D), tuple(nl)),
                    ((wb, 0), tuple(nr)), ((wt, D), tuple(nr))]
    return A, B, contacts


def fig1():
    kinds = [("flat", "Flat contact"), ("housing", "Housing (U-notch)"), ("dovetail", "Dovetail")]
    fig, axes = plt.subplots(2, 3, figsize=(8, 5.2), gridspec_kw={"height_ratios": [1.05, 1]})
    for col, (kind, title) in enumerate(kinds):
        A, Bp, contacts = joint_geometry(kind)
        B = rows(contacts)
        # --- top: geometry with contact normals
        ax = axes[0, col]
        ax.add_patch(Polygon(A, closed=True, fc=WOOD, ec=INK2, lw=0.8, zorder=1))
        ax.add_patch(Polygon(Bp, closed=True, fc=WOOD2, ec=INK2, lw=0.8, zorder=2))
        seen = set()
        for (r, n) in contacts:
            key = (round(n[0], 3), round(n[1], 3), round(r[0]))
            if key in seen:
                continue
            seen.add(key)
            ax.add_patch(FancyArrowPatch(r, (r[0] + 9 * n[0], r[1] + 9 * n[1]), arrowstyle="-|>",
                                         mutation_scale=8, color=BLUE, lw=1.4, zorder=5))
            ax.plot(*r, "o", ms=3, color=BLUE, zorder=6)
        ax.text(0, H_B / 2 + (D if kind != "flat" else 0) / 2, "B", ha="center", va="center", color=INK, fontsize=10)
        ax.text(-33, -6, "A", ha="center", va="center", color=INK, fontsize=10)
        ax.set_xlim(-42, 42); ax.set_ylim(-14, 36)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([-40, -20, 0, 20, 40]); ax.set_yticks([0, 20])
        if col == 0:
            ax.set_ylabel("z (mm)")
        ax.set_xlabel("x (mm)", labelpad=1)
        clean(ax)
        # --- bottom: translational motion cone (dual of the normals) + rotation verdict
        ax = axes[1, col]
        t, ok = free_directions(B)
        R = 1.0
        if ok.sum() > 3:  # a 2D wedge
            tt = t[ok]
            # contiguous arc (handles wrap-around)
            gaps = np.where(np.diff(tt) > 2 * (t[1] - t[0]))[0]
            if len(gaps):
                tt = np.concatenate([tt[gaps[0] + 1:], tt[:gaps[0] + 1] + 2 * np.pi])
            ax.add_patch(Wedge((0, 0), R, np.degrees(tt[0]), np.degrees(tt[-1]), fc=BLUE, alpha=0.18, ec="none", zorder=1))
            ax.plot([0, R * np.cos(tt[0])], [0, R * np.sin(tt[0])], color=BLUE, lw=2, zorder=3)
            ax.plot([0, R * np.cos(tt[-1])], [0, R * np.sin(tt[-1])], color=BLUE, lw=2, zorder=3)
            verdict = "translations: a half-plane\n(lift off, slide along x)"
        elif ok.sum() > 0:  # a ray
            for a in t[ok]:
                ax.plot([0, R * np.cos(a)], [0, R * np.sin(a)], color=BLUE, lw=2.5, zorder=3)
            verdict = "translations: one ray\n(lift out along +z only)"
        else:
            verdict = "translations: only v = 0\n(locked in the plane)"
        ax.plot(0, 0, "o", ms=5, color=BLUE, zorder=4)
        # every contact normal, drawn from the origin as a thin grey ray for reference
        for (_, n) in contacts:
            ax.plot([0, 0.55 * n[0]], [0, 0.55 * n[1]], color=MUTED, lw=1, zorder=2)
        rot = rotation_possible(B)
        verdict += "\nrotation about y: " + ("tipping allowed" if rot else "locked")
        ax.text(0, -1.05, verdict, ha="center", va="top", fontsize=8, color=INK2)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.9, 1.2)
        ax.set_aspect("equal")
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        ax.set_xlabel("$v_x$ (unit velocity)", labelpad=1)
        if col == 0:
            ax.set_ylabel("$v_z$ (unit velocity)")
        clean(ax)
    fig.suptitle("A joint's motion space is the set of velocities that push into no contact: fewer escapes as walls flare",
                 fontsize=10, y=1.0)
    fig.text(0.5, 0.505, "Top: cross-section with the contact normals $n$ pointing into B. Bottom: pure translations $v$ with $v\\cdot n \\geq 0$ "
             "at every contact (blue); grey rays repeat the normals. Sliding along the sweep axis $y$ stays free in all three.",
             ha="center", va="center", fontsize=7.5, color=INK2, wrap=True)
    fig.tight_layout(h_pad=2.6)
    save(fig, f"{SLUG}-1.png")


# ----------------------------------------------------------------------------------------
# Figure 2: tilt analysis of a single block with the RBE linear program (Eqs. 1-3)
# ----------------------------------------------------------------------------------------
def rbe_feasible(phi, mu, w, h, Wt=1.0):
    """Block (width w, height h) on ground tilted by phi (rad). Unknowns per corner: f_n >= 0, f_t.
    Block frame: x along the slope (downhill +x), z normal. Gravity = Wt*(sin phi, -cos phi).
    Equalities: sum f_n = Wt cos phi ; sum f_t = -Wt sin phi ; torque about the centre of mass = 0.
    Coulomb: |f_t| <= mu f_n. Returns feasibility and a force solution (f_n1, f_t1, f_n2, f_t2)."""
    # variables x = [fn1, ft1, fn2, ft2]
    A_eq = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [-w / 2, h / 2, w / 2, h / 2]])  # r x f with r = (+-w/2, -h/2): r_x f_z - r_z f_x
    b_eq = np.array([Wt * np.cos(phi), -Wt * np.sin(phi), 0.0])
    A_ub = np.array([[-mu, 1, 0, 0], [-mu, -1, 0, 0], [0, 0, -mu, 1], [0, 0, -mu, -1]])
    b_ub = np.zeros(4)
    # objective: keep tangential forces small (equalises the friction usage); any feasible point is enough
    res = linprog(c=[0, 0, 0, 0], A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, None), (None, None), (0, None), (None, None)], method="highs")
    return res.status == 0, (res.x if res.status == 0 else None)


def critical_tilt(mu, w, h):
    lo, hi = 0.0, np.pi / 2 - 1e-3
    assert rbe_feasible(lo, mu, w, h)[0] and not rbe_feasible(hi, mu, w, h)[0]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if rbe_feasible(mid, mu, w, h)[0]:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fig2():
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 3.6), gridspec_kw={"width_ratios": [1, 1.15]})
    # --- left: the block at a chosen tilt with friction cones and the force solution to scale
    mu, w, h, phi = 0.5, 1.0, 1.5, np.radians(20)
    ok, x = rbe_feasible(phi, mu, w, h)
    assert ok
    fn1, ft1, fn2, ft2 = x
    # proportional split of friction (inside both cones whenever tan(phi) <= mu); check it against the LP rows
    fn = np.array([fn1, fn2]); ft = -np.sin(phi) * fn / fn.sum()
    assert np.allclose(fn.sum(), np.cos(phi)) and np.all(np.abs(ft) <= mu * fn + 1e-12)
    assert abs((-w / 2) * fn[0] + (h / 2) * ft[0] + (w / 2) * fn[1] + (h / 2) * ft[1]) < 1e-9
    Rm = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])  # block frame -> world (ground tilted by phi)
    def T(p):
        return Rm @ np.array(p)
    blk = [T(p) for p in [(-w / 2, 0), (w / 2, 0), (w / 2, h), (-w / 2, h)]]
    ax0.add_patch(Polygon(blk, closed=True, fc=WOOD2, ec=INK2, lw=0.8, zorder=2))
    gnd = [T(p) for p in [(-1.6, 0), (1.6, 0), (1.6, -0.25), (-1.6, -0.25)]]
    ax0.add_patch(Polygon(gnd, closed=True, fc=WOOD, ec=INK2, lw=0.8, zorder=1))
    scale = 1.1  # drawing length per unit weight
    for i, rx in enumerate([-w / 2, w / 2]):
        p = T((rx, 0))
        # friction cone: half-angle arctan(mu) about the contact normal
        ang = np.degrees(np.arctan(mu))
        ax0.add_patch(Wedge(p, 0.55, 90 + np.degrees(phi) - ang, 90 + np.degrees(phi) + ang, fc=BLUE, alpha=0.15, ec=BLUE, lw=0.8, zorder=3))
        f = T((ft[i], fn[i])) * scale
        ax0.add_patch(FancyArrowPatch(p, p + f, arrowstyle="-|>", mutation_scale=10, color=ORANGE, lw=2, zorder=5))
    c = T((0, h / 2))
    ax0.add_patch(FancyArrowPatch(c, c + np.array([0, -scale]), arrowstyle="-|>", mutation_scale=10, color=INK, lw=2, zorder=5))
    ax0.text(c[0] + 0.08, c[1] - 0.5, "W", color=INK, fontsize=9)
    ax0.text(0.55, -0.62, "contact forces from\nthe LP (orange)", color=INK2, fontsize=8, ha="left")
    ax0.text(-1.62, 1.62, f"friction cones (blue),\nhalf-angle arctan $\\mu$ = {np.degrees(np.arctan(mu)):.1f}$^\\circ$", color=INK2, fontsize=8, va="bottom")
    ax0.set_xlim(-1.7, 1.7); ax0.set_ylim(-0.85, 2.05)
    ax0.set_aspect("equal")
    ax0.set_xlabel("x (block widths)"); ax0.set_ylabel("z (block widths)")
    ax0.set_title(f"Tilt {np.degrees(phi):.0f}$^\\circ$, $\\mu$ = {mu}, h/w = {h / w}: in equilibrium", fontsize=9.5)
    clean(ax0)
    # --- right: critical tilt vs mu for two aspect ratios, LP vs closed form
    mus = np.linspace(0.05, 1.2, 24)
    for (hw, colr, lab) in [(1.0, BLUE, "squat block, h/w = 1"), (2.0, ORANGE, "tall block, h/w = 2")]:
        crit = np.degrees([critical_tilt(m, 1.0, hw) for m in mus])
        analytic = np.degrees(np.minimum(np.arctan(mus), np.arctan(1.0 / hw)))
        assert np.max(np.abs(crit - analytic)) < 0.05, np.max(np.abs(crit - analytic))
        ax1.plot(mus, analytic, color=colr, lw=2, label=lab + " (closed form)")
        ax1.plot(mus, crit, "o", ms=4, mfc="white", mec=colr, mew=1.2, label=lab + " (LP)")
    ax1.axvline(0.5, color=MUTED, lw=1, ls=":")
    ax1.text(0.52, 2, "$\\mu$ = 0.5", color=INK2, fontsize=8)
    ax1.text(0.85, np.degrees(np.arctan(0.5)) + 1.5, "tips first: $\\tan\\phi = w/h$", color=INK2, fontsize=8)
    ax1.text(0.2, 32, "slides first:\n$\\tan\\phi = \\mu$", color=INK2, fontsize=8)
    ax1.set_xlabel("friction coefficient $\\mu$ (–)"); ax1.set_ylabel("critical tilt angle $\\phi$ (degrees)")
    ax1.set_title("Critical tilt = min(sliding limit, tipping limit)", fontsize=9.5)
    ax1.set_ylim(0, 50)
    ax1.legend(loc="upper left", fontsize=7.5)
    clean(ax1)
    fig.tight_layout(w_pad=2)
    save(fig, f"{SLUG}-3.png")


# ----------------------------------------------------------------------------------------
# Figure 3: directional blocking graphs on a computed 3-part 2D assembly
# ----------------------------------------------------------------------------------------
DIRS = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}


def blocks(ci, cj, d):
    """Part j blocks part i translating along d if any j cell lies ahead of any i cell in its row/column."""
    dx, dy = d
    for (x, y) in ci:
        for (u, v) in cj:
            if dx and v == y and (u - x) * dx > 0:
                return True
            if dy and u == x and (v - y) * dy > 0:
                return True
    return False


def dbg(parts, d):
    n = len(parts)
    A = np.zeros((n, n), int)
    for i in range(n):
        for j in range(n):
            if i != j and blocks(parts[i], parts[j], d):
                A[i, j] = 1
    return A


def fig3(parts, key, names=("A", "B", "K")):
    fig, axes = plt.subplots(1, 5, figsize=(8, 2.5), gridspec_kw={"width_ratios": [1.5, 1, 1, 1, 1]})
    ax = axes[0]
    cols = [BLUE, ORANGE, AQUA]
    for p, colr, nm in zip(parts, cols, names):
        for (x, y) in p:
            ax.add_patch(Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)], fc=colr, ec=SURFACE, lw=1.2, alpha=0.85))
        cx = np.mean([x for x, _ in p]) + 0.5; cy = np.mean([y for _, y in p]) + 0.5
        # label at the cell nearest the centroid
        best = min(p, key=lambda c: (c[0] + 0.5 - cx) ** 2 + (c[1] + 0.5 - cy) ** 2)
        ax.text(best[0] + 0.5, best[1] + 0.5, nm, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    xs = [x for p in parts for x, _ in p]; ys = [y for p in parts for _, y in p]
    ax.set_xlim(min(xs) - 0.3, max(xs) + 1.3); ax.set_ylim(min(ys) - 0.3, max(ys) + 1.3)
    ax.set_aspect("equal")
    ax.set_xticks(range(min(xs), max(xs) + 2)); ax.set_yticks(range(min(ys), max(ys) + 2))
    ax.set_xlabel("x (cells)"); ax.set_ylabel("y (cells)")
    ax.set_title("3-part assembly, key K", fontsize=9)
    clean(ax)
    pos = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.5, 0.9)}
    for ax, (dname, d) in zip(axes[1:], DIRS.items()):
        A = dbg(parts, d)
        k, lab = connected_components(A, directed=True, connection="strong")
        for i in range(3):
            for j in range(3):
                if A[i, j]:
                    p, q = np.array(pos[i]), np.array(pos[j])
                    u = (q - p) / np.linalg.norm(q - p)
                    # offset antiparallel arrows so both directions stay visible
                    off = np.array([-u[1], u[0]]) * (0.06 if A[j, i] else 0.0)
                    ax.add_patch(FancyArrowPatch(p + 0.17 * u + off, q - 0.17 * u + off, arrowstyle="-|>",
                                                 mutation_scale=9, color=INK2, lw=1.2, shrinkA=0, shrinkB=0, zorder=2))
        for i in range(3):
            ax.add_patch(Circle(pos[i], 0.14, fc=cols[i], ec="none", zorder=3))
            ax.text(*pos[i], names[i], ha="center", va="center", color="white", fontsize=9, fontweight="bold", zorder=4)
        free = [names[i] for i in range(3) if A[i].sum() == 0]
        if k == 1:
            note = "strongly connected:\nnothing moves"
        elif free:
            note = f"components {{A,B}}, {{K}}:\nK free along {dname}"
        else:
            note = "components {A,B}, {K}:\nK blocked by A and B"
        ax.text(0.5, -0.42, note, ha="center", va="top", fontsize=7.5, color=INK2)
        ax.set_title(f"DBG for {dname}", fontsize=9)
        ax.set_xlim(-0.35, 1.35); ax.set_ylim(-0.95, 1.2)
        ax.set_aspect("equal")
        clean(ax, hide_axes=True)
    fig.suptitle("Interlocking test with blocking graphs: an arrow i → j means part j blocks part i along the direction", fontsize=9.5, y=1.02)
    fig.tight_layout(w_pad=0.6)
    save(fig, f"{SLUG}-2.png")


# ----------------------------------------------------------------------------------------
# Figure 4: Table 1 of the STAR as a count grid (main objective only; secondary in grey text)
# ----------------------------------------------------------------------------------------
def fig4():
    objectives = ["3D printing", "CNC milling", "Laser cutting", "Mixed fabr.", "Equilibrium", "Interlocking",
                  "Free reconfig.", "Hinged reconfig.", "LEGO bricks", "Tileable blocks"]
    groups = [("Fabricability", 0, 4), ("Stability", 4, 6), ("Reconfigurability", 6, 8), ("Tileability", 8, 10)]
    framings = ["Shape decomposition", "Shape approximation", "Parts optimization", "Joint planning",
                "Shape co-decomposition", "Layout optimization"]
    # counted from Table 1 (p. 651): entries printed in black = main objective
    main = np.array([
        [12, 2, 0, 0, 1, 3, 0, 2, 0, 1],
        [0, 0, 6, 3, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 3, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 2, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 1, 4, 1],
    ])
    secondary = np.zeros_like(main)
    secondary[0, 1] = 1  # Polycube map [FCM*18] under CNC milling
    secondary[0, 5] = 1  # Interlock object [SFLF15] under Interlocking
    secondary[3, 5] = 2  # CofiFab [SDW*16], Reconfig. furniture [SFJ*17] under Interlocking
    assert main.sum() == 53 and secondary.sum() == 4
    fig, ax = plt.subplots(figsize=(8, 3.9))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#ffffff"] + SEQ)
    ax.imshow(np.minimum(main, 7), cmap=cmap, vmin=0, vmax=7, aspect="auto")
    for i in range(main.shape[0]):
        for j in range(main.shape[1]):
            if main[i, j]:
                ax.text(j, i, str(main[i, j]), ha="center", va="center", fontsize=10,
                        color="white" if main[i, j] >= 4 else INK)
            if secondary[i, j]:
                ax.text(j + 0.42, i + 0.4, f"+{secondary[i, j]}", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(range(len(objectives))); ax.set_xticklabels(objectives, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(framings))); ax.set_yticklabels(framings, fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(objectives)), minor=True); ax.set_yticks(np.arange(-0.5, len(framings)), minor=True)
    ax.grid(which="minor", color=GRID, lw=0.8); ax.tick_params(which="minor", length=0)
    for (g, a, b) in groups:
        ax.plot([a - 0.5, b - 0.5], [-0.75, -0.75], color=INK2, lw=1, clip_on=False)
        ax.text((a + b - 1) / 2, -0.95, g, ha="center", va="bottom", fontsize=9, color=INK)
    for (_, a, _) in groups[1:]:
        ax.axvline(a - 0.5, color=INK2, lw=1)
    ax.set_xlabel("design objective (columns of Table 1);  '+n' = secondary objective, printed grey in Table 1")
    ax.set_ylabel("problem framing (rows of Table 1)")
    ax.set_title("Each design goal has a favourite framing: 53 papers of Table 1 cluster on the diagonal", pad=28)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    save(fig, f"{SLUG}-4.png")


# A 5x5-cell assembly found by hand and verified by the blocking computation below: K is a cap
# whose legs stop A (right) and B (left) from sliding apart; A and B hook each other in y.
LAYOUT = """K K K K K
K B A A K
K B B A .
. B A A .
. B B . ."""


def parse_layout(s):
    rows = s.strip().split("\n")[::-1]
    parts = {"A": [], "B": [], "K": []}
    for y, r in enumerate(rows):
        for x, ch in enumerate(r.split()):
            if ch in parts:
                parts[ch].append((x, y))
    return [parts["A"], parts["B"], parts["K"]]


if __name__ == "__main__":
    fig1()
    fig2()
    parts = parse_layout(LAYOUT)
    # sanity: the STAR's DBG rule holds with K as the single-part component, free along +y only
    for dname, d in DIRS.items():
        A = dbg(parts, d)
        k, lab = connected_components(A, directed=True, connection="strong")
        assert k == 1 or (k == 2 and (lab == lab[2]).sum() == 1), dname
        assert (A[2].sum() == 0) == (dname == "+y"), dname
    assert not blocks(parts[0], parts[1], (1, 0))  # without K, A slides out along +x
    fig3(parts, key=2)
    fig4()
