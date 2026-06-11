"""Compose the learned prior with the classical optimizer: the prior picks the template,
the optimizer fits only that template's parameters (warm-started). Optionally fall back to
the next-ranked template if the chosen fit is infeasible / high-energy."""
from __future__ import annotations

import numpy as np
from shapely.geometry import LineString

from .dataset import Member
from .damage import Damage
from .energy import BIG
from .optimizer import FitResult, fit_template
from .prior import TorchCNNPrior
from .rasterize import rasterize
from .templates import TEMPLATES


def plan_repair(member: Member, damage: Damage, mate: LineString,
                prior: TorchCNNPrior, raster: np.ndarray | None = None,
                top_k: int = 1, rng=None) -> FitResult:
    if raster is None:
        raster = rasterize(member, damage, mate)
    ranked = prior.rank_templates(raster)[:max(1, top_k)]
    best = None
    for name in ranked:
        fit = fit_template(member, damage, TEMPLATES[name], mate, rng=rng)
        if best is None or fit.energy.total < best.energy.total:
            best = fit
        if fit.energy.feasible and fit.energy.total < BIG:
            break
    return best
