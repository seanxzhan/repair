"""Reverse-engineer a *relational* joint model from the per-part jwood CSG.

The dataset parametrizes every part independently: a contact between two members
is stored twice (a pocket cut into A and, separately, a matching tenon on B),
with nothing tying the two halves together. This module recovers the shared
structure instead:

    Joint = members (stock blocks) + interfaces (the mating features),
    where each interface is authored ONCE and lists its incident parts + roles.

Pipeline (`import_joint` / `reconstruct_parts`):
  1. Parse jwood -> per-part stock LHF + cut LHFs (`jwood.PartSpec`).
  2. **Resolve the in-plane orientation** of the stock and every cut against the
     STL ground truth (`resolve_orientations`). Each LHF's plane is fixed by its
     normal but its 2D sketch can sit in 8 orientations; we pick the combination
     whose analytic solid matches the STL. This yields exact reconstructions
     (incl. holes, rectangular stock, and 45deg-normal joints). Results are
     cached under `out/orient_cache/` so re-viewing is instant.
  3. Build each part's solid = stock - union(cuts) with the resolved orientations.
  4. For every ordered pair (A, B), the *male feature* of A into B is
     `solid(A) & union(cuts(B))` -- A's material in B's removed pocket. A
     non-trivial volume means A and B share an interface.
  5. Each unordered pair's interface = the two male features with per-part roles
     (`male` / `female`); a part that is only ever male and sits enclosed inside
     the others is a derived **filler** (key / peg / wedge).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import trimesh

from repair.config import OUT_DIR, DATASET_ROOT
from repair.jwood import N_ORIENT, LHF, PartSpec, lhf_contains, lhf_to_mesh

# A cut/feature whose volume falls below this (in normalized member^3 units) is
# treated as numerical dust rather than a real interface.
VOL_EPS = 1e-3

# A part whose solid is enclosed by the other members above this fraction is a
# derived filler (key / peg / wedge) rather than a structural member.
FILLER_ENCLOSURE = 0.9

# Orientation resolution: sample count and the bump in agreement that counts as a
# real improvement (guards against ties between equivalent symmetric orientations).
RESOLVE_SAMPLES = 20000
_TIE = 1e-12
_CACHE_VERSION = 2


def _safe_union(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh | None:
    if not meshes:
        return None
    if len(meshes) == 1:
        return meshes[0]
    try:
        return trimesh.boolean.union(meshes)
    except Exception:
        acc = meshes[0]
        for m in meshes[1:]:
            try:
                acc = acc.union(m)
            except Exception:
                pass
        return acc


def _safe_vol(mesh: trimesh.Trimesh | None) -> float:
    if mesh is None or mesh.is_empty:
        return 0.0
    try:
        v = float(mesh.volume)
    except Exception:
        return 0.0
    return v if np.isfinite(v) else 0.0


def _intersection_vol(a: trimesh.Trimesh, b: trimesh.Trimesh | None) -> float:
    if b is None:
        return 0.0
    try:
        return _safe_vol(a.intersection(b))
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# Orientation resolution against the STL ground truth.
# --------------------------------------------------------------------------- #
def resolve_orientations(
    stock: LHF, cuts: list[LHF], stl: trimesh.Trimesh,
    n_samples: int = RESOLVE_SAMPLES, rng_seed: int = 0,
) -> tuple[int, list[int], float]:
    """Resolve (stock_ori, [cut_ori...]) so the analytic solid matches the STL.

    Coordinate descent over the 8 orientations of each LHF, scored by agreement
    of point-membership with the STL. Masks are precomputed per (LHF, ori) so
    each descent step is a cheap boolean combine. Returns the agreement too
    (1.0 == exact) so callers can flag any part that did not resolve.
    """
    lo, hi = stl.bounds
    pts = np.random.RandomState(rng_seed).uniform(lo - 0.05, hi + 0.05, (n_samples, 3))
    in_stl = stl.contains(pts)

    stock_masks = [lhf_contains(stock, o, pts) for o in range(N_ORIENT)]
    cut_masks = [[lhf_contains(c, o, pts) for o in range(N_ORIENT)] for c in cuts]

    def member(so: int, co: list[int]) -> np.ndarray:
        m = stock_masks[so].copy()
        for ci, o in enumerate(co):
            m &= ~cut_masks[ci][o]
        return m

    def agree(so: int, co: list[int]) -> float:
        return float((member(so, co) == in_stl).mean())

    so, co = 0, [0] * len(cuts)
    for _ in range(8):
        changed = False
        best = (agree(so, co), so)
        for o in range(N_ORIENT):
            a = agree(o, co)
            if a > best[0] + _TIE:
                best = (a, o)
        if best[1] != so:
            so, changed = best[1], True
        for i in range(len(cuts)):
            best = (agree(so, co), co[i])
            for o in range(N_ORIENT):
                trial = co[:]
                trial[i] = o
                a = agree(so, trial)
                if a > best[0] + _TIE:
                    best = (a, o)
            if best[1] != co[i]:
                co[i], changed = best[1], True
        if not changed:
            break
    return so, co, agree(so, co)


def build_solid(
    stock: LHF, stock_ori: int, cuts: list[LHF], cut_oris: list[int],
) -> tuple[trimesh.Trimesh, "trimesh.Trimesh | None", list[trimesh.Trimesh]]:
    """(solid, cut_union, cut_meshes): solid = stock - union(cuts), all oriented."""
    solid = lhf_to_mesh(stock, stock_ori)
    cut_meshes = [lhf_to_mesh(c, o) for c, o in zip(cuts, cut_oris)]
    cut_union = _safe_union(cut_meshes)
    if cut_union is not None:
        try:
            solid = solid.difference(cut_union)
        except Exception:
            pass
    return solid, cut_union, cut_meshes


# --------------------------------------------------------------------------- #
# Orientation cache (resolution is the only slow step; results are static).
# --------------------------------------------------------------------------- #
def _cache_path(key: str, variant: str) -> Path:
    return OUT_DIR / "orient_cache" / f"{key}_{variant}.json"


def _load_cache(key: str, variant: str) -> dict:
    p = _cache_path(key, variant)
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
        return d.get("parts", {}) if d.get("version") == _CACHE_VERSION else {}
    except Exception:
        return {}


def _save_cache(key: str, variant: str, parts: dict) -> None:
    p = _cache_path(key, variant)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"version": _CACHE_VERSION, "parts": parts}, indent=1))


# --------------------------------------------------------------------------- #
# Per-part reconstruction artifacts.
# --------------------------------------------------------------------------- #
@dataclass
class PartRecon:
    """All artifacts of reconstructing one part -- for inspection / viewing."""

    name: str
    spec: PartSpec
    stock_ori: int
    cut_oris: list[int]
    agreement: float           # point-membership agreement with the STL (1=exact)
    stl: trimesh.Trimesh
    stock_mesh: trimesh.Trimesh
    cut_meshes: list[trimesh.Trimesh]
    cut_union: "trimesh.Trimesh | None"
    solid: trimesh.Trimesh

    @property
    def vol_error(self) -> float:
        gt = _safe_vol(self.stl)
        return abs(_safe_vol(self.solid) - gt) / max(gt, 1e-6)

    def param_text(self) -> str:
        """Human-readable dump of the parameters used for reconstruction."""
        lines = [f"{self.name}: {self.spec.expression}",
                 f"  agreement={self.agreement * 100:.1f}%  "
                 f"vol_error={self.vol_error * 100:.1f}%",
                 f"  stock {_lhf_text(self.spec.stock, self.stock_ori)}"]
        for cut, ori in zip(self.spec.cuts, self.cut_oris):
            lines.append(f"  cut   {_lhf_text(cut, ori)}")
        return "\n".join(lines)


def _lhf_text(lhf: LHF, ori: int) -> str:
    nrings = len(lhf.rings)
    ring = f"{nrings} rings" if nrings > 1 else f"{len(lhf.rings[0])}-gon"
    return (f"{lhf.name}: n={_vec(lhf.normal)} o={_vec(lhf.origin)} "
            f"amount={lhf.amount:+.2f} ori={ori} ({ring})")


def _vec(v) -> str:
    return "[" + ",".join(f"{x:+.2f}" for x in v) + "]"


def _stl_paths(key: str, variant: str, n: int) -> list[Path]:
    d = DATASET_ROOT / key / "meshes" / variant
    return [d / f"{i}.stl" for i in range(n)]


def reconstruct_parts(key: str, variant: str = "base",
                      use_cache: bool = True) -> list[PartRecon]:
    """Per-part reconstruction artifacts (geometry + resolved orientations)."""
    jw = json.loads((DATASET_ROOT / key / "vis_files" / f"{variant}_jwood.json").read_text())
    specs = [PartSpec.from_json(p) for p in jw["parts"]]
    stl_paths = _stl_paths(key, variant, len(specs))

    cache = _load_cache(key, variant) if use_cache else {}
    dirty = False
    recons: list[PartRecon] = []
    for spec, stl_path in zip(specs, stl_paths):
        stl = trimesh.load(stl_path, force="mesh")
        if spec.name in cache:
            c = cache[spec.name]
            so, co, ag = c["stock_ori"], c["cut_oris"], c["agreement"]
        else:
            so, co, ag = resolve_orientations(spec.stock, spec.cuts, stl)
            cache[spec.name] = {"stock_ori": so, "cut_oris": co, "agreement": ag}
            dirty = True
        solid, cut_union, cut_meshes = build_solid(spec.stock, so, spec.cuts, co)
        recons.append(PartRecon(
            name=spec.name, spec=spec, stock_ori=so, cut_oris=co, agreement=ag,
            stl=stl, stock_mesh=lhf_to_mesh(spec.stock, so),
            cut_meshes=cut_meshes, cut_union=cut_union, solid=solid,
        ))
    if dirty and use_cache:
        _save_cache(key, variant, cache)
    return recons


# --------------------------------------------------------------------------- #
# Relational data model.
# --------------------------------------------------------------------------- #
@dataclass
class Member:
    """A stock block (joint node). `is_filler` parts are derived, not authored."""

    name: str
    stock_volume: float
    solid_volume: float
    is_filler: bool = False


@dataclass
class Incidence:
    part: str
    role: str            # "male" (protrudes) | "female" (hollowed) | "fill"
    volume: float        # volume of this part's contribution to the interface


@dataclass
class Interface:
    """One mating feature shared by >=2 parts (a graph edge / hyperedge)."""

    id: str
    incidences: list[Incidence]
    volume: float        # total interlocking volume

    @property
    def parts(self) -> list[str]:
        return [inc.part for inc in self.incidences]

    @property
    def degree(self) -> int:
        return len(self.incidences)


@dataclass
class Joint:
    key: str
    variant: str
    members: dict[str, Member]
    interfaces: list[Interface] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"{self.key} ({self.variant}): "
                 f"{len(self.members)} members, {len(self.interfaces)} interfaces"]
        for m in self.members.values():
            tag = "  [FILLER, derived]" if m.is_filler else ""
            lines.append(f"  member {m.name}: stock_vol={m.stock_volume:.2f} "
                         f"solid_vol={m.solid_volume:.2f}{tag}")
        for itf in self.interfaces:
            roles = ", ".join(f"{i.part}:{i.role}({i.volume:.2f})" for i in itf.incidences)
            lines.append(f"  interface {itf.id} (deg {itf.degree}, "
                         f"vol={itf.volume:.2f}): {roles}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Importer.
# --------------------------------------------------------------------------- #
def import_joint(key: str, variant: str = "base") -> Joint:
    """Reverse-engineer the relational `Joint` for one joint folder + variant."""
    recons = reconstruct_parts(key, variant)

    solids = {r.name: r.solid for r in recons}
    cut_unions = {r.name: r.cut_union for r in recons}
    members: dict[str, Member] = {
        r.name: Member(name=r.name,
                       stock_volume=_safe_vol(r.stock_mesh),
                       solid_volume=_safe_vol(r.solid))
        for r in recons
    }
    names = [r.name for r in recons]

    # Directed male features -- A's material inside B's removed pockets.
    male: dict[tuple[str, str], float] = {}
    for a, b in combinations(names, 2):
        male[(a, b)] = _intersection_vol(solids[a], cut_unions[b])
        male[(b, a)] = _intersection_vol(solids[b], cut_unions[a])

    # One interface per contacting pair, with per-part roles.
    interfaces: list[Interface] = []
    for a, b in combinations(names, 2):
        va, vb = male[(a, b)], male[(b, a)]
        if max(va, vb) < VOL_EPS:
            continue
        incs = []
        if va >= VOL_EPS:
            incs.append(Incidence(a, "male", va))
        if vb >= VOL_EPS:
            incs.append(Incidence(b, "male", vb))
        present = {i.part for i in incs}
        for p in (a, b):
            if p not in present:
                incs.append(Incidence(p, "female", 0.0))
        interfaces.append(Interface(id=f"{a}__{b}", incidences=incs, volume=va + vb))

    # Filler classification by enclosure: a key / peg / wedge sits inside the
    # other members (solid contained in their convex hull); a structural member
    # -- even an interlocking splice half -- spans out to the assembly's extremes.
    for name in names:
        sv = members[name].solid_volume
        if sv <= VOL_EPS:
            continue
        others = [solids[o] for o in names if o != name]
        if not others:
            continue
        try:
            hull = trimesh.util.concatenate(others).convex_hull
            enclosure = _intersection_vol(solids[name], hull) / sv
        except Exception:
            enclosure = 0.0
        if enclosure > FILLER_ENCLOSURE:
            members[name].is_filler = True
            for itf in interfaces:
                for inc in itf.incidences:
                    if inc.part == name and inc.role == "male":
                        inc.role = "fill"

    return Joint(key=key, variant=variant, members=members, interfaces=interfaces)
