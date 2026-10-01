"""Step 2: the labelled dataset behind the surrogate.

Sampling is fields times interfaces. A field is one random damage field (a
seed and four knobs); on every field, every family gets `per_field` interfaces
drawn uniformly within its bounds and clipped to feasibility. Each row records
the clipped parameters, the field index, and what the solver says: capacity,
sound wood removed, and how many samples of each contact face are live.

The split is by field, so "unseen damage" means unseen: a held-out field
contributes no row to training in any family.

Output, in `out_dir`:
    fields.npz        seed, knobs, raster of the interface window, test flag
    <family>.npz      one file per family, rows as above

The raster is the damage severity on the window x in [6, 12] over the full
height, at 96 x 32 (WINDOW, RASTER), which is the region every family's
bounds allow the interface to occupy: the network's damage input.
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np

from . import damage as dm
from . import families as fm
from .model import Block, Grid, Statics

WINDOW = (6.0, 12.0, 0.0, 2.0)          # x0, x1, y0, y1 of the raster
RASTER = (32, 96)                       # ny, nx
MAX_FACES = 6                           # per-face live counts are padded to this
KNOB_NAMES = ("reach", "end_frac", "aniso", "patch")


def raster(f: dm.FieldDamage) -> np.ndarray:
    x0, x1, y0, y1 = WINDOW
    ny, nx = RASTER
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    X, Y = np.meshgrid(xs, ys)
    return f.severity(X, Y).astype(np.float16)


def make_field(index: int, master_seed: int, b: Block = Block()):
    """The field with this index: its own seed and its own knobs."""
    rng = np.random.default_rng([master_seed, index])
    knobs = dm.sample_knobs(rng)
    seed = int(rng.integers(0, 2**31 - 1))
    return seed, knobs, dm.random_field(seed, knobs, b)


def sample_params(fam: fm.Family, rng: np.random.Generator, b: Block, tries: int = 20) -> dict:
    """Uniform within the bounds, resampled until feasible so the rows are
    uniform over the feasible set rather than piled on its boundary; clipped
    only if `tries` draws all fail."""
    for _ in range(tries):
        p = {q.name: float(rng.uniform(q.lo, q.hi)) for q in fam.params}
        q = fam.feasible(p, b)
        if all(abs(q[k] - p[k]) < 1e-9 for k in p):
            return q
    return q


def evaluate_field(args):
    """Worker: all rows of one field. Returns (index, seed, knobs, raster, rows per family)."""
    index, master_seed, keys, per_field = args
    b, grid, st = Block(), Grid(), Statics()
    seed, knobs, f = make_field(index, master_seed, b)
    out = {}
    for key in keys:
        fam = fm.FAMILIES[key]
        rng = np.random.default_rng([master_seed, index, hash(key) % (2**31)])
        P = np.zeros((per_field, len(fam.params)), np.float32)
        M = np.zeros(per_field, np.float32); R = np.zeros(per_field, np.float32)
        live = np.zeros(per_field, np.int16); n_c = np.zeros(per_field, np.int16)
        face_live = np.full((per_field, MAX_FACES), -1, np.int16); face_n = np.full((per_field, MAX_FACES), -1, np.int16)
        for i in range(per_field):
            p = sample_params(fam, rng, b)
            m, r, cs, _ = fm.evaluate(fam, p, f, st, grid)
            P[i] = fam.vector(p); M[i] = m; R[i] = r
            live[i] = sum(c.live for c in cs); n_c[i] = len(cs)
            faces = sorted({c.face for c in cs})[:MAX_FACES]
            for k, fi in enumerate(faces):
                cc = [c for c in cs if c.face == fi]
                face_live[i, k] = sum(c.live for c in cc); face_n[i, k] = len(cc)
        out[key] = dict(params=P, M=M, R=R, live=live, n_contacts=n_c, face_live=face_live, face_n=face_n)
    return index, seed, knobs, raster(f), out


def generate(out_dir, n_fields: int = 2000, per_field: int = 20, keys=None, master_seed: int = 0,
             test_frac: float = 0.1, workers: int | None = None, log=print):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(keys or fm.FAMILIES)
    jobs = [(i, master_seed, keys, per_field) for i in range(n_fields)]
    seeds = np.zeros(n_fields, np.int64); knobs = np.zeros((n_fields, len(KNOB_NAMES)), np.float32)
    rasters = np.zeros((n_fields, *RASTER), np.float16)
    rows = {k: [None] * n_fields for k in keys}
    workers = workers or max(1, mp.cpu_count() - 1)
    with mp.get_context("fork").Pool(workers) as pool:
        for n_done, (i, seed, kn, ras, out) in enumerate(pool.imap_unordered(evaluate_field, jobs, chunksize=4), 1):
            seeds[i] = seed; knobs[i] = [getattr(kn, a) for a in KNOB_NAMES]; rasters[i] = ras
            for k in keys:
                rows[k][i] = out[k]
            if n_done % max(1, n_fields // 10) == 0 or n_done == n_fields:
                log(f"  {n_done}/{n_fields} fields")
    # split by field: the last test_frac of a fixed permutation is held out
    perm = np.random.default_rng(master_seed).permutation(n_fields)
    test = np.zeros(n_fields, bool); test[perm[: int(round(test_frac * n_fields))]] = True
    np.savez_compressed(out_dir / "fields.npz", seed=seeds, knobs=knobs, knob_names=np.array(KNOB_NAMES),
                        raster=rasters, test=test, window=np.array(WINDOW), master_seed=master_seed)
    for k in keys:
        fam = fm.FAMILIES[k]
        cat = {name: np.concatenate([r[name] for r in rows[k]]) for name in rows[k][0]}
        field = np.repeat(np.arange(n_fields), per_field).astype(np.int32)
        np.savez_compressed(out_dir / f"{k}.npz", field=field, test=test[field],
                            param_names=np.array([q.name for q in fam.params]), **cat)
    return out_dir


def load(out_dir):
    """(fields, {family: rows}) as dicts of arrays."""
    out_dir = Path(out_dir)
    fields = dict(np.load(out_dir / "fields.npz"))
    fams = {p.stem: dict(np.load(p)) for p in sorted(out_dir.glob("*.npz")) if p.stem != "fields"}
    return fields, fams


def field_of(fields: dict, index: int, b: Block = Block()) -> dm.FieldDamage:
    """Rebuild a stored field exactly from its seed and knobs."""
    kn = dm.Knobs(**{a: float(v) for a, v in zip(fields["knob_names"], fields["knobs"][index])})
    return dm.random_field(int(fields["seed"][index]), kn, b)
