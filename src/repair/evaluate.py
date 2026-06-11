"""Evaluation: energy regret of the prior vs the oracle, against baselines.

Energy regret = E(prior's repair) - E(oracle's repair), in oracle-energy units. The prior
is good if its regret is near 0 (matching the oracle) while only fitting one template. We
compare against picking a random template and the most-frequent template; the oracle is the
zero-regret upper bound.

Because the dataset stores every template's oracle energy per sample, regret for *any*
template choice is a table lookup -- no re-optimization needed.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .energy import BIG
from .prior import TorchCNNPrior
from .templates import TEMPLATE_NAMES

FEASIBLE = 0.5 * BIG  # an instance's optimized energy below this is feasible


def _regret_for(sample, template_name: str) -> float:
    return sample.all_energies[template_name] - sample.energy_star


def prior_choice(prior: TorchCNNPrior, sample) -> str:
    """Feasibility-aware selection, mirroring inference.plan_repair: take the highest-
    probability template whose optimized repair is feasible for this instance."""
    proba = prior.predict_proba(sample.raster)
    for idx in np.argsort(proba)[::-1]:
        name = TEMPLATE_NAMES[idx]
        if sample.all_energies[name] < FEASIBLE:
            return name
    return TEMPLATE_NAMES[int(np.argmax(proba))]


def regret_oracle(samples) -> float:
    return 0.0


def regret_random(samples, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    vals = [_regret_for(s, TEMPLATE_NAMES[rng.integers(len(TEMPLATE_NAMES))])
            for s in samples]
    return float(np.mean(vals))


def regret_most_frequent(train, test) -> float:
    freq = Counter(s.label_name for s in train)
    mf = freq.most_common(1)[0][0]
    return float(np.mean([_regret_for(s, mf) for s in test]))


def regret_prior(prior: TorchCNNPrior, samples) -> float:
    vals = [_regret_for(s, prior_choice(prior, s)) for s in samples]
    return float(np.mean(vals))


def template_accuracy(prior: TorchCNNPrior, samples) -> float:
    return float(np.mean([prior_choice(prior, s) == s.label_name for s in samples]))


def confusion_matrix(prior: TorchCNNPrior, samples) -> np.ndarray:
    n = len(TEMPLATE_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for s in samples:
        pred = TEMPLATE_NAMES.index(prior_choice(prior, s))
        cm[s.label, pred] += 1
    return cm


def evaluate_split(prior: TorchCNNPrior, train, test) -> dict:
    return {
        "n_test": len(test),
        "accuracy": template_accuracy(prior, test),
        "regret_prior": regret_prior(prior, test),
        "regret_random": regret_random(test),
        "regret_most_frequent": regret_most_frequent(train, test),
        "regret_oracle": 0.0,
    }
