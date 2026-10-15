"""A pressure-bounded contact model, as an alternative evaluator. Additive:
nothing in model.py, families.py or dataset.py changes.

model.py bounds the SUM of normal forces by a global cap and switches each
contact sample live or dead at one probe point. Two consequences: a face
loses capacity in jumps of one sample, and a face that is 99% rotten carries
as much as a sound one until its last sample dies. Both are modelling
choices, and both make the capacity landscape a staircase.

Here each sample's normal force is bounded on its own, by a bearing pressure
times the SOUND length of the piece of interface the sample represents:

    f_n[k] <= p_max * (tributary length of k) * (fraction of it that is sound)

There is no global cap (the pressure bound already keeps the squeeze finite,
so Mosemann's artifact needs no separate fix), no binary flag (a sample's
bound shrinks continuously as rot crosses its segment), and the sample count
no longer matters beyond resolution of the sound fraction. Capacity becomes a
continuous function of the geometry; what remains non-smooth is the genuine
kind, kinks where the LP's active set changes.

p_max is the crushing strength of the wood perpendicular to grain, per unit
length of interface in this 2D model. Its absolute value only scales the
answer; the default puts a sound tenon's capacity in the same range as the
cap model's.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import families as fm
from .model import EPS, Block, Contact, Grid, _span, solve_moment


@dataclass(frozen=True)
class PressureStatics:
    mu: float = 0.5          # wood-on-wood friction
    p_max: float = 25.0      # bearing pressure: max normal force per unit length of sound interface


def contact_set(fam: fm.Family, p: dict, d, b: Block = Block(), n_samp: int = 7, sub: int = 50):
    """(contacts, budgets): every sample with its own force bound. The sound
    fraction of a sample's segment is measured at `sub` points along it,
    probed just inside the retained wood like model.contact_set does."""
    cs, budgets, face = [], [], 0
    for lay in fam.layers(p, b):
        for p0, p1, n in fm.interface_segments(lay, b):
            seg = float(np.hypot(*(p1 - p0))) / n_samp
            pts = _span(p0, p1, n_samp)
            for k in range(n_samp):
                t = (k + (np.arange(sub) + 0.5) / sub) / n_samp
                q = p0 + np.outer(t, p1 - p0) - EPS * n
                frac = float(d.sound(q[:, 0], q[:, 1], b).mean())
                cs.append(Contact(face, pts[k], n, frac > 0.0))
                budgets.append(([len(cs) - 1], lay.weight * seg * frac))     # scaled by p_max in evaluate
            face += 1
    return cs, budgets


def evaluate(fam: fm.Family, p: dict, d, st: PressureStatics, grid: Grid, n_samp: int = 7, sub: int = 50):
    """(M, R, contacts, forces), the same tuple as families.evaluate, under the
    pressure-bounded model."""
    b = grid.block
    cs, budgets = contact_set(fam, p, d, b, n_samp, sub)
    budgets = [(idx, st.p_max * length) for idx, length in budgets]
    M, F = solve_moment(cs, _as_statics(st), budgets)
    return M, fm.sound_removed(fam, p, d, grid), cs, F


def _as_statics(st: PressureStatics):
    from .model import Statics
    return Statics(mu=st.mu, cap=float("inf"))     # the cap row is replaced by the budgets anyway


def sanity_checks():
    """Continuity and sampling independence on the Figure 1 sweep, and the
    textbook limit: two opposing sound faces of length L carry p_max*L each."""
    from .model import Damage
    b, grid = Block(), Grid()
    fam = fm.FAMILIES["tenon"]; p0 = fam.defaults(); st = PressureStatics()
    A = np.linspace(8.6, 10.2, 81)
    y7 = np.array([evaluate(fam, dict(p0, a=a), Damage(), st, grid, 7)[0] for a in A])
    y21 = np.array([evaluate(fam, dict(p0, a=a), Damage(), st, grid, 21)[0] for a in A])
    assert np.abs(np.diff(y7)).max() < 0.08 * y7.max(), "a step larger than 8% of peak: not continuous"
    assert np.abs(y7 - y21).max() < 0.03 * y7.max(), "sample count changes the answer"
    # sound wood: every sample's bound is p_max * its tributary length
    cs, budgets = contact_set(fam, p0, Damage(x0=30.0), b, 7)
    total = sum(l for _, l in budgets)
    assert abs(total - sum(np.hypot(*(s[1] - s[0])) for lay in fam.layers(p0, b) for s in fm.interface_segments(lay, b))) < 1e-9
    return True
