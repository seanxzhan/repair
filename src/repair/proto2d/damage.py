"""Damage as a field on the block, not only a front.

model.Damage is a logistic front with three numbers. This module adds a
severity field on a grid, and a generator that makes erosion-like fields
cheaply: severity is a logistic of the anisotropic distance from seeds on the
boundary, so rot enters from the end face, runs further along the grain than
across it, may bite a bay into one face, and has a ragged front. The rot is
always one region touching the end face.

    FieldDamage     severity on a grid; `severity` / `sound` interpolate it, so
                    it is a drop-in for model.Damage wherever a (x, y, block)
                    query is made (contact probes, the sound-wood grid)
    Knobs           four numbers: reach, end_frac, aniso, patch
    random_field    one field from a seed and Knobs
    sample_knobs    Knobs drawn from stated ranges, for data generation

None of this is calibrated to real decay. It is a synthetic distribution with
four stated knobs, chosen so that a damage field is a picture the network has
to read rather than three numbers it can memorize.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, map_coordinates

from .model import Block, Damage


# Fixed by design: they change how the raster looks, not what the label sees
# (the LP and the sound-wood mask threshold severity at CRIT, and the front sits
# at distance `reach` whatever the falloff).
WIDTH = 0.15         # severity goes 0.73 -> 0.27 over this length across the front
NOISE = 0.10         # the front is displaced by this much, in block units
NOISE_LEN = 0.40     # over this correlation length; keep NOISE below it or the front fragments
CRIT = 0.5           # a point is dead where severity exceeds this


@dataclass(frozen=True)
class Knobs:
    """The generator's four parameters. Lengths are in block units. Everything
    else about a field (where the end-face run sits, which face the patch is
    on and where, the noise realization) comes from the seed."""
    reach: float = 2.0           # how far the rot runs from the end face along the grain
    end_frac: float = 0.7        # fraction of the end face that is seeded
    aniso: float = 0.3           # how much shorter it runs across the grain (fraction of reach)
    patch: float = 0.5           # size of one surface pocket as a fraction of reach; 0 for none


@dataclass
class FieldDamage:
    """Severity on cell centres of an (ny, nx) grid over the whole block."""
    sev: np.ndarray
    block: Block = Block()
    crit: float = CRIT
    token: tuple = field(default=())      # (seed, knobs): identifies the field cheaply

    @property
    def shape(self):
        return self.sev.shape

    def _index(self, x, y):
        ny, nx = self.sev.shape
        ix = np.asarray(x, float) / self.block.length * nx - 0.5
        iy = np.asarray(y, float) / self.block.height * ny - 0.5
        return iy, ix

    def severity(self, x, y, block: Block | None = None):
        iy, ix = self._index(x, y)
        out = map_coordinates(self.sev, [np.ravel(iy), np.ravel(ix)], order=1, mode="nearest")
        return out.reshape(np.shape(iy)) if np.ndim(iy) else float(out[0])

    def sound(self, x, y, block: Block | None = None):
        return self.severity(x, y, block) < self.crit

    def front_points(self, b: Block | None = None) -> np.ndarray:
        """Cell centres where soundness flips against the right-hand neighbour:
        a cheap stand-in for the crit contour, for drawing."""
        ny, nx = self.sev.shape
        s = self.sev < self.crit
        flip = s[:, :-1] != s[:, 1:]
        i, j = np.nonzero(flip)
        xs = (j + 1.0) * self.block.length / nx
        ys = (i + 0.5) * self.block.height / ny
        return np.column_stack([xs, ys])

    @classmethod
    def from_front(cls, d: Damage, b: Block = Block(), nx: int = 240, ny: int = 40) -> "FieldDamage":
        """Rasterize model.Damage, so the two representations can be compared."""
        xs = (np.arange(nx) + 0.5) * b.length / nx
        ys = (np.arange(ny) + 0.5) * b.height / ny
        X, Y = np.meshgrid(xs, ys)
        return cls(d.severity(X, Y, b), b, d.crit, ("front", d))


def random_field(seed: int, k: Knobs = Knobs(), b: Block = Block(), nx: int = 240, ny: int = 40) -> FieldDamage:
    """Rot eroding inward from the boundary: severity is a logistic of the
    anisotropic distance from the seeds. The rot is one connected region that
    touches the end face; a surface pocket is a bay in that region, never an
    island. Placements come from `seed`, the distribution from `k`."""
    rng = np.random.default_rng(seed)
    dx, dy = b.length / nx, b.height / ny
    reach = max(k.reach, 1e-6)
    # anisotropic distance: along the grain in block units, across the grain stretched by 1 / aniso
    sampling = (dy / max(k.aniso, 1e-3), dx)

    def dist(src):
        return distance_transform_edt(~src, sampling=sampling) if src.any() else np.full(src.shape, np.inf)

    # the end face: a contiguous run of end_frac of the height, placed at random
    end = np.zeros((ny, nx), bool)
    n_end = max(1, int(round(k.end_frac * ny)))
    y0 = rng.integers(0, ny - n_end + 1)
    end[y0:y0 + n_end, -1] = True
    u = dist(end) / reach                                 # normalized distance: the front is at u = 1
    # a ragged front: displace the normalized distance by a smooth noise of
    # NOISE block units, so the contour moves without fragmenting
    z = gaussian_filter(rng.standard_normal((ny, nx)), sigma=(NOISE_LEN / dy, NOISE_LEN / dx), mode="reflect")
    z *= (NOISE / reach) / max(z.std(), 1e-12)
    u = u + z

    def one_region(u):
        lab, n = label(u < 1.0)
        return n == 1 and lab[end].max() > 0

    # one surface pocket, on the top or bottom face, centred near where the
    # end rot ends along that face so it reads as a bay extending the rot, and
    # never an island. A few placements are tried; if none joins the end rot
    # the field has no pocket.
    if k.patch > 0:
        p_reach = k.patch * reach
        row = -1 if rng.random() < 0.5 else 0
        inside = np.nonzero(u[row] < 1.0)[0]
        x_front = (inside.min() + 0.5) * dx if inside.size else b.length
        lo, hi = max(0.55 * b.length, x_front - 0.9 * p_reach), min(b.length, x_front + 0.3 * p_reach)
        for _ in range(8):
            xc = rng.uniform(lo, max(lo, hi))
            j0 = max(0, int((xc - 0.5 * p_reach) / dx)); j1 = min(nx, int((xc + 0.5 * p_reach) / dx) + 1)
            patch = np.zeros((ny, nx), bool); patch[row, j0:j1] = True
            u_try = np.minimum(u, dist(patch) / p_reach + z)
            if one_region(u_try):
                u = u_try
                break
    # guard: the noise can still pinch off a sliver; mirror anything not in the
    # seed run's component back across the front so severity stays continuous
    dead = u < 1.0
    lab, n = label(dead)
    keep = lab == lab[end].max()
    u = np.where(dead & ~keep, 2.0 - u, u)
    # severity: 1 at the seeds, 0.5 on the front, falling off over WIDTH
    sev = 1.0 / (1.0 + np.exp(np.clip((u - 1.0) * reach / WIDTH, -60, 60)))
    return FieldDamage(np.clip(sev, 0.0, 1.0), b, CRIT, (seed, k))


KNOB_RANGES = dict(reach=(0.5, 4.0), end_frac=(0.3, 1.0), aniso=(0.15, 0.6), patch=(0.2, 1.0))
P_NO_PATCH = 0.3     # probability that a sampled field has no surface pocket


def sample_knobs(rng: np.random.Generator) -> Knobs:
    """Knobs drawn uniformly from KNOB_RANGES; no pocket with probability P_NO_PATCH."""
    r = KNOB_RANGES
    return Knobs(reach=rng.uniform(*r["reach"]), end_frac=rng.uniform(*r["end_frac"]),
                 aniso=rng.uniform(*r["aniso"]),
                 patch=0.0 if rng.random() < P_NO_PATCH else rng.uniform(*r["patch"]))


def ascii_field(d: FieldDamage, nx: int = 96, ny: int = 16, x0: float = 6.0) -> str:
    """Severity over x in [x0, length] in ten shades, '.' sound to '@' rotten."""
    b = d.block
    xs = x0 + (np.arange(nx) + 0.5) * (b.length - x0) / nx
    ys = (np.arange(ny) + 0.5) * b.height / ny
    X, Y = np.meshgrid(xs, ys)
    s = d.severity(X, Y)
    shades = " .:-=+*#%@"
    idx = np.clip((s * (len(shades) - 1)).round().astype(int), 0, len(shades) - 1)
    dead = ~d.sound(X, Y)
    rows = []
    for i in range(ny - 1, -1, -1):
        rows.append("".join((shades[idx[i, j]] if not dead[i, j] else shades[max(idx[i, j], 5)]) for j in range(nx)))
    return "\n".join(rows)


def sanity_checks(seed: int = 0):
    """The rasterized front agrees with the analytic one away from the front;
    random fields are sound at the far end, dead at the end face, and one
    connected region of rot."""
    b = Block()
    d = Damage(); fd = FieldDamage.from_front(d, b)
    rng = np.random.default_rng(seed)
    pts = np.column_stack([rng.uniform(0, b.length, 4000), rng.uniform(0, b.height, 4000)])
    far = np.abs(pts[:, 0] - d.front_x(pts[:, 1], b)) > 0.1
    assert np.array_equal(d.sound(pts[far, 0], pts[far, 1], b), fd.sound(pts[far, 0], pts[far, 1]))
    for s in range(30):
        k = sample_knobs(rng); f = random_field(s, k, b)
        assert f.sev.min() >= 0 and f.sev.max() <= 1
        assert f.sound(0.5, 0.5 * b.height), (s, k)                      # the far end is untouched
        ys = np.linspace(0.05, b.height - 0.05, 30)
        assert (~f.sound(np.full(30, b.length - 0.02), ys)).any(), (s, k)   # something died at the end face
        _, n = label(f.sev >= f.crit)
        assert n == 1, (s, k, n)                                            # one region of rot
    return True
