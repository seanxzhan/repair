"""Step 4a: the repair objective and four ways to climb it.

    objective   J(p) = M / M_ref  -  lam * R / R_ref  -  pen * max(0, M_req - M)^2 / M_ref^2
                M capacity, R sound wood removed; M_ref is the family's training peak,
                R_ref half the block's area

Routes, all in the family's normalized parameter space [0, 1]^P, all projected
to feasibility after every step, all re-scored by the hard LP at the end:

    surrogate   gradient ascent; dM/dp from the network's autograd, dR/dp by
                central differences on the exact grid integral (R is smooth)
    soft_lp     gradient ascent on the LP with softened contact flags: every
                contact is live with a force budget w * cap, w = sigmoid((crit -
                severity) / tau); the landscape is then continuous and its
                gradient comes from central differences
    hard_fd     gradient ascent with central differences on the hard LP: the
                staircase, exactly as a classical optimizer would feel it
    reference   the best available: a random search on the hard LP refined by
                Nelder-Mead from the best few points; the upper bound the
                others are measured against

Everything is counted in hard-LP calls so cost can be compared.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import torch
from scipy.optimize import minimize

from . import families as fm
from .model import Block, Grid, Statics, EPS, solve_moment


@dataclass(frozen=True)
class Objective:
    lam: float = 1.5           # weight on sound wood removed
    m_req: float = 0.0         # required capacity (0: no load constraint)
    pen: float = 4.0           # penalty weight on the shortfall below m_req
    m_ref: float = 100.0       # capacity scale (the family's training peak)
    r_ref: float = 12.0        # sound-wood scale (half the block)

    def __call__(self, M, R):
        short = np.maximum(0.0, self.m_req - M)
        return M / self.m_ref - self.lam * R / self.r_ref - self.pen * short ** 2 / self.m_ref ** 2


@dataclass
class Problem:
    """One field, one family, one surrogate: everything a route needs."""
    fam: fm.Family
    f: object                      # damage field (FieldDamage or Damage)
    obj: Objective
    model: object = None           # CapacityNet (optional: only the surrogate route needs it)
    ck: dict = None
    raster: torch.Tensor = None    # (1, H, W)
    device: str = "cpu"
    grid: Grid = field(default_factory=Grid)
    st: Statics = field(default_factory=Statics)
    b: Block = field(default_factory=Block)
    tau: float = 0.1               # softness of the contact flags, in severity units
    lp_calls: int = 0

    def __post_init__(self):
        self.lo = np.array([q.lo for q in self.fam.params]); self.hi = np.array([q.hi for q in self.fam.params])
        self.names = [q.name for q in self.fam.params]
        if self.model is not None:
            with torch.no_grad():
                self.code = self.model.encode(self.raster.unsqueeze(0).to(self.device))
            n = self.ck["normalization"]
            assert n["param_names"] == self.names

    # ---- parameter space
    def denorm(self, x) -> dict:
        v = self.lo + np.clip(np.asarray(x, float), 0, 1) * (self.hi - self.lo)
        return dict(zip(self.names, map(float, v)))

    def norm(self, p: dict) -> np.ndarray:
        return np.clip((np.array([p[k] for k in self.names]) - self.lo) / (self.hi - self.lo), 0, 1)

    def project(self, x) -> np.ndarray:
        return self.norm(self.fam.feasible(self.denorm(x), self.b))

    def on_bound(self, x, tol=1e-3) -> list[str]:
        x = np.asarray(x); return [n for n, v in zip(self.names, x) if v <= tol or v >= 1 - tol]

    # ---- evaluators
    def hard(self, x):
        """(M, R) from the hard LP. Counted."""
        self.lp_calls += 1
        M, R, _, _ = fm.evaluate(self.fam, self.denorm(x), self.f, self.st, self.grid)
        return M, R

    def sound_removed(self, x) -> float:
        return fm.sound_removed(self.fam, self.denorm(x), self.f, self.grid)

    def soft(self, x) -> float:
        """Capacity with softened contact flags. Counted as one LP call."""
        self.lp_calls += 1
        p = self.denorm(x)
        layers = self.fam.layers(p, self.b)
        cs = fm.contact_set(self.fam, p, self.f, self.b)
        if not cs:
            return 0.0
        q = np.array([c.point - EPS * c.normal for c in cs])
        sev = np.asarray(self.f.severity(q[:, 0], q[:, 1], self.b), float)
        w = 1.0 / (1.0 + np.exp(np.clip((sev - self.f.crit) / self.tau, -60, 60)))
        live = [replace(c, live=True) for c in cs]
        budgets = fm.budgets(live, layers, self.st) + [([k], float(w[k]) * self.st.cap) for k in range(len(cs))]
        return solve_moment(live, self.st, budgets)[0]

    def surrogate(self, x) -> tuple[float, np.ndarray]:
        """(M, dM/dx) from the network, in capacity units and per normalized parameter."""
        xt = torch.tensor(np.asarray(x, np.float32), device=self.device).unsqueeze(0).requires_grad_(True)
        M = self.model.from_code(self.code, xt)
        M.backward()
        peak = self.ck["normalization"]["peak"]
        return float(M.item()) * peak, xt.grad[0].cpu().numpy() * peak

    def fd(self, fn, x, h):
        """Central differences of a scalar function in normalized space."""
        g = np.zeros(len(x))
        for i in range(len(x)):
            e = np.zeros(len(x)); e[i] = h
            g[i] = (fn(np.clip(x + e, 0, 1)) - fn(np.clip(x - e, 0, 1))) / (2 * h)
        return g


@dataclass
class Run:
    route: str
    x0: np.ndarray
    x: np.ndarray                  # end point
    traj: list                     # [(x, M_est, R)] per step
    M_true: float
    R_true: float
    J_true: float
    J_est: float                   # the route's own estimate at the end point
    lp_calls: int
    steps: int
    on_bound: list


def ascend(prob: Problem, route: str, x0, steps=80, lr=0.05, h=0.02, tol=1e-5, log=None) -> Run:
    """Projected gradient ascent on J with Adam, using the route's gradient."""
    x = prob.project(np.asarray(x0, float)); obj = prob.obj
    m = np.zeros_like(x); v = np.zeros_like(x); b1, b2 = 0.9, 0.999
    calls0 = prob.lp_calls; traj = []
    R_fn = prob.sound_removed

    def est(xx):
        if route == "surrogate":
            return prob.surrogate(xx)[0]
        return prob.soft(xx) if route == "soft_lp" else prob.hard(xx)[0]

    for t in range(1, steps + 1):
        R = R_fn(x); dR = prob.fd(R_fn, x, h)
        if route == "surrogate":
            M, dM = prob.surrogate(x)
        elif route == "soft_lp":
            M = prob.soft(x); dM = prob.fd(prob.soft, x, h)
        elif route == "hard_fd":
            M = prob.hard(x)[0]; dM = prob.fd(lambda xx: prob.hard(xx)[0], x, h)
        else:
            raise ValueError(route)
        traj.append((x.copy(), M, R))
        short = max(0.0, obj.m_req - M)
        dJ = dM / obj.m_ref - obj.lam * dR / obj.r_ref + 2 * obj.pen * short * dM / obj.m_ref ** 2
        m = b1 * m + (1 - b1) * dJ; v = b2 * v + (1 - b2) * dJ ** 2
        step = lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-8)
        x_new = prob.project(x + step)
        moved = np.abs(x_new - x).max(); x = x_new
        if log: log(route, t, x, M, R)
        if moved < tol and t > 5:
            break
    M_est = est(x); R = R_fn(x); traj.append((x.copy(), M_est, R))
    calls_used = prob.lp_calls - calls0
    M_true, R_true = prob.hard(x)
    return Run(route, np.asarray(x0, float), x, traj, M_true, R_true, float(obj(M_true, R_true)), float(obj(M_est, R)),
               calls_used, len(traj) - 1, prob.on_bound(x))


def reference(prob: Problem, n_random=1500, n_refine=3, seed=0, maxiter=150) -> Run:
    """Best available on the hard LP: random search, then Nelder-Mead from the best few."""
    rng = np.random.default_rng(seed); calls0 = prob.lp_calls; obj = prob.obj
    X = [prob.project(rng.random(len(prob.names))) for _ in range(n_random)]
    J = []
    for x in X:
        M, R = prob.hard(x); J.append(obj(M, R))
    order = np.argsort(J)[::-1]
    best_x, best_J = X[order[0]], J[order[0]]
    for k in order[:n_refine]:
        res = minimize(lambda xx: -obj(*prob.hard(prob.project(xx))), X[k], method="Nelder-Mead",
                       options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-5))
        xr = prob.project(res.x); Jr = -res.fun
        if Jr > best_J:
            best_x, best_J = xr, Jr
    M, R = prob.hard(best_x)
    return Run("reference", best_x, best_x, [(best_x, M, R)], M, R, float(obj(M, R)), float(obj(M, R)),
               prob.lp_calls - calls0, 0, prob.on_bound(best_x))
