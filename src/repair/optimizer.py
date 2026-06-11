"""Per-template parameter fit and the oracle that produces training labels.

For a fixed template we minimize the proxy energy over its continuous parameters with a
gradient-free optimizer (constraints are folded in as large penalties inside ``energy``).
The ``oracle`` runs every template and returns the lowest-energy one -- this is the gold
standard the learned prior is trained to imitate, and the upper bound it is measured against.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from shapely.geometry import LineString

from .dataset import Member
from .damage import Damage
from .energy import EnergyTerms, energy
from .templates import TEMPLATES, Template


@dataclass
class FitResult:
    template: str
    theta: np.ndarray
    energy: EnergyTerms
    success: bool

    @property
    def total(self) -> float:
        return self.energy.total


def _objective(theta, member, damage, template, mate):
    cut = template.apply(member, damage, theta)
    return energy(member, damage, cut, mate).total


def fit_template(member: Member, damage: Damage, template: Template, mate: LineString,
                 theta0: np.ndarray | None = None, n_starts: int = 3,
                 rng=None) -> FitResult:
    rng = rng or np.random.default_rng(0)
    bounds = template.bounds
    starts = []
    starts.append(theta0 if theta0 is not None else template.default_theta(member, damage))
    starts.append(template.default_theta(member, damage))
    for _ in range(max(0, n_starts - 2)):
        starts.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds]))

    best = None
    for s in starts:
        res = minimize(_objective, np.asarray(s, dtype=float),
                       args=(member, damage, template, mate),
                       method="Nelder-Mead",
                       options=dict(maxiter=200, xatol=1e-3, fatol=1e-3))
        theta = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])
        cut = template.apply(member, damage, theta)
        e = energy(member, damage, cut, mate)
        if best is None or e.total < best.energy.total:
            best = FitResult(template=template.name, theta=theta, energy=e,
                             success=bool(res.success) and e.feasible)
    return best


def oracle(member: Member, damage: Damage, mate: LineString,
           templates: dict = TEMPLATES, rng=None) -> tuple[str, FitResult, dict]:
    fits = {name: fit_template(member, damage, t, mate, rng=rng)
            for name, t in templates.items()}
    best_name = min(fits, key=lambda n: fits[n].energy.total)
    return best_name, fits[best_name], fits
