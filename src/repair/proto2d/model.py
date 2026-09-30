"""A 2D model of one repair interface on a partly rotten block.

This is the model behind docs/figures/proposal.py and examples/proto2d/inspect_splice.py.
It is deliberately small: a cross-section, one interface family (a
mortise-and-tenon splice), a graded damage field, and a rigid-body frictional
equilibrium LP that returns the largest bending moment the interface can
transfer. Everything is numpy + scipy; no meshes, no polyscope.

    Block     the piece of wood, x along its length, y up
    Damage    a logistic decay front; sound iff severity < crit
    Splice    the interface: shoulder at x = a, tenon of length ell between
              cheeks y1 < y2. ell = 0 is a plain cut.
    Statics   friction and the force cap

The contact set is rebuilt from scratch for every (Splice, Damage): each face is
sampled at N midpoints, and a sample is *live* only if the retained wood just
behind it is sound. The geometry is continuous in the parameters; `live` is not.
That discontinuity is the point of the figure and the GUI.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import linprog

FACES = ["mortise back", "top cheek", "bottom cheek", "upper shoulder", "lower shoulder"]


@dataclass(frozen=True)
class Block:
    length: float = 12.0
    height: float = 2.0


@dataclass(frozen=True)
class Damage:
    """Sound iff x < x0 + slope * (y / height); logistic transition of `width`."""
    x0: float = 8.6
    slope: float = 0.9          # the front leans: rot runs further along the bottom
    width: float = 0.18
    crit: float = 0.5           # a contact sample is dead where severity exceeds this

    def front_x(self, y, block: Block):
        return self.x0 + self.slope * (np.asarray(y) / block.height)

    def severity(self, x, y, block: Block):
        return 1.0 / (1.0 + np.exp(-(np.asarray(x) - self.front_x(y, block)) / self.width))

    def sound(self, x, y, block: Block):
        return self.severity(x, y, block) < self.crit


@dataclass(frozen=True)
class Splice:
    a: float = 9.48             # shoulder position
    ell: float = 0.76           # tenon length; 0 is a plain cut
    y1: float = 0.65            # lower cheek
    y2: float = 1.35            # upper cheek
    n_samp: int = 7             # contact samples per face

    def with_cheeks(self, center: float, thickness: float) -> "Splice":
        return replace(self, y1=center - thickness / 2, y2=center + thickness / 2)


@dataclass(frozen=True)
class Statics:
    mu: float = 0.5             # wood-on-wood friction
    cap: float = 100.0          # cap on sum(f_n): Mosemann's fix for the squeeze artifact


@dataclass
class Contact:
    face: int
    point: np.ndarray           # (2,)
    normal: np.ndarray          # (2,) unit, pointing into the new wood
    live: bool


EPS = 0.02                      # how far inside the retained wood we probe for soundness


def _span(p0, p1, n):
    """n sample points at segment midpoints, so no sample lands on an endpoint."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = (np.arange(n) + 0.5) / n
    return p0 + np.outer(t, p1 - p0)


def contact_set(s: Splice, d: Damage, b: Block = Block()) -> list[Contact]:
    """Every contact sample of the interface, with its live flag."""
    xb, n = s.a - s.ell, s.n_samp
    specs = [(0, _span((xb, s.y1), (xb, s.y2), n), (+1.0, 0.0), (-EPS, 0.0))]
    if s.ell > 1e-9:
        specs += [(1, _span((xb, s.y2), (s.a, s.y2), n), (0.0, -1.0), (0.0, +EPS)),
                  (2, _span((xb, s.y1), (s.a, s.y1), n), (0.0, +1.0), (0.0, -EPS))]
    specs += [(3, _span((s.a, s.y2), (s.a, b.height), n), (+1.0, 0.0), (-EPS, 0.0)),
              (4, _span((s.a, 0.0), (s.a, s.y1), n), (+1.0, 0.0), (-EPS, 0.0))]
    out = []
    for fi, pts, nrm, off in specs:
        for p in pts:
            q = p + np.asarray(off)          # probe just inside the retained wood
            out.append(Contact(fi, p, np.asarray(nrm, float), bool(d.sound(q[0], q[1], b))))
    return out


def solve_moment(contacts: list[Contact], st: Statics = Statics(), budgets=None):
    """Largest hogging moment the live contacts can transfer, and the forces that do it.

    Free body: the new wood, loaded by a pure couple M (no net force, so no
    lever arm). Unknowns are f_n >= 0 and f_t per live sample (the 2D friction
    cone is exactly two half-planes) plus M. Maximise M subject to force and
    moment balance, |f_t| <= mu f_n, and sum(f_n) <= cap.

    `budgets` optionally splits the cap: a list of (contact indices, cap) pairs,
    each bounding sum(f_n) over its own contacts. families.py uses this to give a
    layer that spans a fraction w of the beam's width a budget of w * cap.
    Without it there is one budget, st.cap, over every contact.

    Returns (M, forces) where forces is an (n_contacts, 2) array of the contact
    force vectors, zero on dead samples.

    The cap is not cosmetic. The tenon's two cheeks have opposing normals, so
    without it the pair can squeeze arbitrarily hard, buy unbounded friction and
    make M unbounded -- the peg-in-hole artifact of mosemann1997.
    """
    forces = np.zeros((len(contacts), 2))
    idx = [i for i, c in enumerate(contacts) if c.live]
    if not idx:
        return 0.0, forces
    m = len(idx)
    col = {i: k for k, i in enumerate(idx)}         # contact index -> LP slot
    if budgets is None:
        budgets = [(idx, st.cap)]
    nz = 2 * m + 1                            # [f_n, f_t] per contact, then M
    A_eq = np.zeros((3, nz))
    tans = []
    for k, i in enumerate(idx):
        p, n = contacts[i].point, contacts[i].normal
        t = np.array([-n[1], n[0]])           # in-plane tangent
        tans.append(t)
        A_eq[0, 2 * k], A_eq[0, 2 * k + 1] = n[0], t[0]
        A_eq[1, 2 * k], A_eq[1, 2 * k + 1] = n[1], t[1]
        A_eq[2, 2 * k] = p[0] * n[1] - p[1] * n[0]      # moment about the origin
        A_eq[2, 2 * k + 1] = p[0] * t[1] - p[1] * t[0]
    A_eq[2, -1] = 1.0                         # the applied couple, no force rows

    A_ub, b_ub = [], []
    for k in range(m):
        r = np.zeros(nz); r[2 * k] = -st.mu; r[2 * k + 1] = +1.0; A_ub.append(r); b_ub.append(0.0)
        r = np.zeros(nz); r[2 * k] = -st.mu; r[2 * k + 1] = -1.0; A_ub.append(r); b_ub.append(0.0)
    for members, cap in budgets:
        r = np.zeros(nz)
        for i in members:
            if i in col:
                r[2 * col[i]] = 1.0
        A_ub.append(r); b_ub.append(cap)

    c = np.zeros(nz); c[-1] = -1.0            # maximise M
    bounds = [(0, None), (None, None)] * m + [(0, None)]
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=A_eq,
                  b_eq=np.zeros(3), bounds=bounds, method="highs")
    if res.status != 0:
        return 0.0, forces
    for k, i in enumerate(idx):
        forces[i] = res.x[2 * k] * contacts[i].normal + res.x[2 * k + 1] * tans[k]
    return max(float(res.x[-1]), 0.0) + 0.0, forces


def max_moment(contacts: list[Contact], st: Statics = Statics()) -> float:
    return solve_moment(contacts, st)[0]


class Grid:
    """A fixed sampling of the block, so area integrals are cheap to repeat."""

    def __init__(self, b: Block = Block(), nx: int = 480, ny: int = 80):
        self.block, self.nx, self.ny = b, nx, ny
        xs = (np.arange(nx) + 0.5) * b.length / nx
        ys = (np.arange(ny) + 0.5) * b.height / ny
        self.X, self.Y = np.meshgrid(xs, ys)
        self.cell = (b.length / nx) * (b.height / ny)

    def removed_mask(self, s: Splice):
        """Everything right of the shoulder, plus the mortise pocket."""
        return (self.X > s.a) | ((self.X > s.a - s.ell) & (self.Y > s.y1) & (self.Y < s.y2))

    def sound_removed(self, s: Splice, d: Damage) -> float:
        """Area of sound wood the interface throws away."""
        return float((d.sound(self.X, self.Y, self.block) & self.removed_mask(s)).sum()) * self.cell


def landscape(a_grid, l_grid, s: Splice, d: Damage, st: Statics, grid: Grid):
    """Capacity and sound-wood-removed over a (tenon length x shoulder) slice of
    the interface space, with the other Splice fields held at `s`."""
    M = np.zeros((len(l_grid), len(a_grid)))
    R = np.zeros_like(M)
    for i, ell in enumerate(l_grid):
        for j, a in enumerate(a_grid):
            sj = replace(s, a=float(a), ell=float(ell))
            M[i, j] = max_moment(contact_set(sj, d, grid.block), st)
            R[i, j] = grid.sound_removed(sj, d)
    return M, R


def cliff_mask(M, tol, robust=False):
    """Cells where the score jumps against a neighbour: the active-set boundaries.

    A jump is a neighbour difference above `tol`. With `robust=True` the
    threshold along each axis is raised to at least 3x the median non-zero
    difference along that axis, so the smooth within-cell slope (which scales
    with the grid step) is never mistaken for a cliff on a coarse grid.
    """
    J = np.zeros_like(M)
    for axis in (0, 1):
        dd = np.abs(np.diff(M, axis=axis))
        t = tol
        if robust and np.any(dd > 1e-12):
            t = max(tol, 3.0 * float(np.median(dd[dd > 1e-12])))
        hit = (dd > t).astype(float) * dd
        if axis == 0:
            J[:-1] = np.maximum(J[:-1], hit); J[1:] = np.maximum(J[1:], hit)
        else:
            J[:, :-1] = np.maximum(J[:, :-1], hit); J[:, 1:] = np.maximum(J[:, 1:], hit)
    return J > 0


def sanity_checks(cap: float = 100.0):
    """Hand-computable cases, so the LP's signs and moments are pinned down."""
    st = Statics(mu=0.0, cap=cap)
    W, n = 2.0, 9
    up = [Contact(0, p, np.array([0.0, 1.0]), True) for p in _span((0.0, 0.0), (W, 0.0), n)]
    dn = [Contact(1, p, np.array([0.0, -1.0]), True) for p in _span((0.0, 1.0), (W, 1.0), n)]
    # One face: every normal points the same way, so no couple has zero net force.
    assert max_moment(up, st) < 1e-6
    # Two opposing faces, frictionless: push up at the far end of one with cap/2
    # and down at the far end of the other with cap/2, giving M = cap * s / 2,
    # where s is the span between the outermost samples (midpoints, so s < W).
    two, want = max_moment(up + dn, st), cap * (W * (n - 1) / n) / 2.0
    assert abs(two - want) / want < 1e-6, (two, want)
    # No live contact carries nothing; a plain cut in sound wood carries nothing.
    assert max_moment([Contact(0, np.zeros(2), np.array([0.0, 1.0]), False)]) == 0.0
    assert max_moment(contact_set(Splice(a=8.0, ell=0.0), Damage())) < 1e-6
    return two, want
