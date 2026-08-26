"""Forward construction: authoring NEW Linked Height Fields.

`jwood.py` *reads* the dataset's LHFs and `relational.py` reverse-engineers the
joint they describe. This module goes the other way -- it *writes* them, which
is what repair-patch synthesis needs: a repair is a new cut authored into a
damaged member, and it must be expressible in the same parametrization the
dataset uses, or it is not millable by the same process that made the joint.

The one fact this module exists to make concrete:

    The in-plane orientation index `ori` is an artifact of READING, not WRITING.

The dataset stores a sketch as 2D coordinates plus a plane (`plane_normal`,
`plane_origin`) but never records how the sketch's local (x, y) axes embed in
that plane. A reader must therefore recover it -- `relational.resolve_orientations`
brute-forces all 8 candidates against the ground-truth STL. An author has no such
problem: you *choose* the frame. Author every sketch in `jwood.seed_basis(normal)`
-- the same deterministic frame the reader tries first -- and `ori=0` is correct
by construction. That matters because synthesized repair geometry has no ground
truth to resolve against, so the STL-resolution crutch is unavailable.

`verify_reader_recovers` proves it end to end: author a part, treat the authored
solid as if it were the ground-truth STL, hand it to the unmodified reader, and
check the reader comes back with ori=0 (or an orientation in the same symmetry
class, which is geometrically the same solid).

Conventions used throughout:
  * `amount > 0` always. The extrusion runs along +normal, so choose the normal
    to point the way the cut travels. (`jwood._placement` flips the axis for
    negative amounts, producing a left-handed frame; authoring avoids it.)
  * Sketch coordinates are in `seed_basis(normal)` = (u, v), so a sketch point
    (a, b) sits at `origin + a*u + b*v` in world space.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import shapely
import trimesh

from repair.jwood import LHF, N_ORIENT, lhf_contains, lhf_to_mesh, orientation_frame, seed_basis

Ring = np.ndarray          # (n, 2) closed polygon in sketch coords
Rings = list[Ring]


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------------------- #
# Constructing an LHF.
# --------------------------------------------------------------------------- #
def make_lhf(name: str, rings: Rings, signs: list[int], normal, origin,
             amount: float) -> LHF:
    """An LHF from a sketch already expressed in `seed_basis(normal)` coords.

    This is the honest constructor: what you pass is what the reader will read,
    with no orientation guesswork, because both sides agree the sketch frame is
    the seed basis (i.e. `ori=0`).
    """
    if amount <= 0:
        raise ValueError("author with amount > 0; point `normal` the way the cut travels")
    return LHF(name=name, rings=[np.asarray(r, float)[:, :2] for r in rings],
               signs=list(signs), origin=np.asarray(origin, float),
               normal=unit(normal), amount=float(amount))


def lhf_from_world(name: str, rings_3d, signs, normal, amount: float,
                   origin=None) -> tuple[LHF, float]:
    """An LHF from world-space base-face rings; returns (lhf, planarity_residual).

    Use when you know the cut's footprint in the assembled frame (e.g. you traced
    it off a damaged region) rather than in sketch coords. The world points are
    projected into `seed_basis(normal)`; the residual is the largest out-of-plane
    component, which should be ~0 if the points really are coplanar.
    """
    n = unit(normal)
    rings_3d = [np.asarray(r, float) for r in rings_3d]
    if origin is None:
        origin = rings_3d[0].mean(axis=0)
    origin = np.asarray(origin, float)
    frame = orientation_frame(n, 0)
    rings, residual = [], 0.0
    for r in rings_3d:
        rel = r - origin
        rings.append(rel @ frame[:, :2])
        residual = max(residual, float(np.abs(rel @ frame[:, 2]).max()))
    return make_lhf(name, rings, signs, n, origin, amount), residual


def box_lhf(name: str, center, extents, axis: int = 1) -> LHF:
    """An axis-aligned box as an LHF, extruded along +`axis` (0=X, 1=Y, 2=Z).

    The workhorse for damage volumes and simple pockets. Written via
    `lhf_from_world` so it exercises the same projection path as traced geometry.
    """
    center, extents = np.asarray(center, float), np.asarray(extents, float)
    n = np.zeros(3)
    n[axis] = 1.0
    lo = center - extents / 2.0
    other = [i for i in range(3) if i != axis]
    corners = []
    for sx, sy in [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]:
        p = lo.copy()
        p[other[0]] += sx * extents[other[0]]
        p[other[1]] += sy * extents[other[1]]
        corners.append(p)
    base_origin = lo.copy()
    base_origin[other[0]] += extents[other[0]] / 2.0
    base_origin[other[1]] += extents[other[1]] / 2.0
    lhf, _ = lhf_from_world(name, [np.array(corners)], [1], n, float(extents[axis]),
                            origin=base_origin)
    return lhf


# --------------------------------------------------------------------------- #
# A small library of 2D sketches, in seed-basis coords.
# Each builder returns (rings, signs) so holes are first-class, not an add-on.
# --------------------------------------------------------------------------- #
def _closed(pts) -> np.ndarray:
    pts = np.asarray(pts, float)
    return np.vstack([pts, pts[:1]])


def sk_rect(w: float, h: float, _unused: float = 0.0):
    return [_closed([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)])], [1]


def sk_dovetail(w: float, h: float, flare: float):
    """Trapezoid: width `w` at the mouth (v=-h/2), flaring by `flare` at the root."""
    a, b = w / 2, w / 2 + flare
    return [_closed([(-a, -h / 2), (a, -h / 2), (b, h / 2), (-b, h / 2)])], [1]


def sk_lap(w: float, h: float, notch: float):
    """L-shape: a rectangle with one corner squared out -- the half-lap footprint."""
    a, b = w / 2, h / 2
    return [_closed([(-a, -b), (a, -b), (a, b - notch), (a - notch, b - notch),
                     (a - notch, b), (-a, b)])], [1]


def sk_rect_hole(w: float, h: float, r: float):
    """Rectangle with a circular hole -- the `poly_signs = -1` case (cf. CJ_AKT)."""
    outer = sk_rect(w, h)[0][0]
    t = np.linspace(0, 2 * np.pi, 33)
    return [outer, np.column_stack([r * np.cos(t), r * np.sin(t)])], [1, -1]


def sk_disc(r: float, _h: float = 0.0, _f: float = 0.0):
    """A round dowel/peg footprint (cf. RJM_SDT)."""
    t = np.linspace(0, 2 * np.pi, 49)
    return [np.column_stack([r * np.cos(t), r * np.sin(t)])], [1]


@dataclass
class Sketch:
    label: str
    params: tuple[str, str, str]
    defaults: tuple[float, float, float]
    build: Callable[[float, float, float], tuple[Rings, list[int]]]


SKETCHES: list[Sketch] = [
    Sketch("rect", ("width", "height", "-"), (0.6, 0.4, 0.0),
           lambda a, b, c: sk_rect(a, b, c)),
    Sketch("dovetail", ("mouth", "depth", "flare"), (0.5, 0.5, 0.18),
           lambda a, b, c: sk_dovetail(a, b, c)),
    Sketch("lap (L)", ("width", "height", "notch"), (0.9, 0.9, 0.35),
           lambda a, b, c: sk_lap(a, b, c)),
    Sketch("rect + hole", ("width", "height", "hole r"), (0.8, 0.6, 0.15),
           lambda a, b, c: sk_rect_hole(a, b, c)),
    Sketch("disc", ("radius", "-", "-"), (0.25, 0.0, 0.0),
           lambda a, b, c: sk_disc(a, b, c)),
]


# --------------------------------------------------------------------------- #
# Emitting the dataset's own JSON form, and proving the round trip.
# --------------------------------------------------------------------------- #
def to_jwood_lhf(lhf: LHF) -> dict:
    """The dict shape the dataset stores (2D sketch points padded to z=0)."""
    return {
        "plane_normal": [float(x) for x in lhf.normal],
        "plane_origin": [float(x) for x in lhf.origin],
        "amount": [float(lhf.amount)],
        "polysets": [[[float(p[0]), float(p[1]), 0.0] for p in r] for r in lhf.rings],
        "poly_signs": [int(s) for s in lhf.signs],
    }


def to_jwood_part(name: str, stock: LHF, cuts: list[LHF]) -> dict:
    """A full `parts[i]` entry: `Difference(lhf_0, Union(lhf_1, ...))`."""
    lhfs = {"lhf_0": to_jwood_lhf(stock)}
    keys = []
    for i, c in enumerate(cuts, start=1):
        lhfs[f"lhf_{i}"] = to_jwood_lhf(c)
        keys.append(f"lhf_{i}")
    if not keys:
        expr = "Difference(lhf_0, lhf_0)"
    else:
        rest = keys[0] if len(keys) == 1 else f"Union({', '.join(keys)})"
        expr = f"Difference(lhf_0, {rest})"
    return {"name": name, "expression": expr, "lhfs": lhfs}


def roundtrip_error(lhf: LHF) -> tuple[float, float]:
    """(symmetric-difference volume, intended volume) after a JSON encode/decode.

    Encodes with `to_jwood_lhf`, decodes with the *unmodified* reader
    (`LHF.from_json`), evaluates at ori=0, compares to the intended solid.
    """
    intended = lhf_to_mesh(lhf, 0)
    decoded = lhf_to_mesh(LHF.from_json(lhf.name, to_jwood_lhf(lhf)), 0)
    try:
        sym = (abs(intended.difference(decoded).volume)
               + abs(decoded.difference(intended).volume))
    except Exception:
        sym = float("nan")
    return float(sym), float(abs(intended.volume))


# --------------------------------------------------------------------------- #
# Orientation: symmetry classes, and the reader-recovery proof.
# --------------------------------------------------------------------------- #
def orientation_classes(lhf: LHF, n_samples: int = 6000, seed: int = 0) -> list[int]:
    """`cls[ori]` = class id; same id => the two orientations give the same solid.

    Uses the same point-membership test `resolve_orientations` scores with, so a
    "tie" here is exactly a tie the reader cannot break. This is why a symmetric
    stock block (a square cross-section) resolves to an arbitrary-looking index:
    all four rotations are the same solid.
    """
    meshes = [lhf_to_mesh(lhf, o) for o in range(N_ORIENT)]
    lo = np.min([m.bounds[0] for m in meshes], axis=0) - 0.05
    hi = np.max([m.bounds[1] for m in meshes], axis=0) + 0.05
    pts = np.random.RandomState(seed).uniform(lo, hi, (n_samples, 3))
    masks = [lhf_contains(lhf, o, pts) for o in range(N_ORIENT)]
    cls, reps = [-1] * N_ORIENT, []
    for o in range(N_ORIENT):
        for ci, r in enumerate(reps):
            if np.array_equal(masks[o], masks[r]):
                cls[o] = ci
                break
        else:
            cls[o] = len(reps)
            reps.append(o)
    return cls


def verify_reader_recovers(stock: LHF, cuts: list[LHF], n_samples: int = 8000):
    """Author -> pretend it is an STL -> let the reader resolve orientations.

    Returns (stock_ori, cut_oris, agreement, ok). `ok` is True when every
    recovered index is in the same symmetry class as 0, i.e. the reader got the
    authored solid back exactly -- no STL of a real part involved anywhere.
    """
    from repair.relational import build_solid, resolve_orientations

    authored, _, _ = build_solid(stock, 0, cuts, [0] * len(cuts))
    so, co, ag = resolve_orientations(stock, cuts, authored, n_samples=n_samples)
    ok = orientation_classes(stock)[so] == orientation_classes(stock)[0]
    for c, o in zip(cuts, co):
        ok = ok and orientation_classes(c)[o] == orientation_classes(c)[0]
    return so, co, float(ag), bool(ok)


# --------------------------------------------------------------------------- #
# Manufacturability of an authored cut.
# --------------------------------------------------------------------------- #
def tool_radius_violation(lhf: LHF, radius: float) -> float:
    """Sketch area a cutter of `radius` cannot reach, as a fraction of the region.

    Morphological opening: erode by r then dilate by r. Whatever does not come
    back (inside corners, thin slivers) is unmillable at that tool size.
    """
    region = lhf.region_2d()
    if region.is_empty or region.area <= 0:
        return 0.0
    opened = region.buffer(-radius).buffer(radius)
    return float(max(0.0, region.area - opened.area) / region.area)


def buried_fraction(lhf: LHF, part_solid: trimesh.Trimesh, n_samples: int = 4000,
                    seed: int = 0) -> float:
    """Fraction of the cut's *entry face* that starts inside the material.

    A milled pocket has to be open: the cutter enters from outside. If the base
    plane of the extrusion is already buried in the part, the cut needs a plunge
    or an undercut and is not a simple 2.5D operation. 0.0 = fully open.
    """
    frame = orientation_frame(lhf.normal, 0)
    region = lhf.region_2d()
    if region.is_empty:
        return 0.0
    lo, hi = np.array(region.bounds).reshape(2, 2)
    rs = np.random.RandomState(seed)
    uv = rs.uniform(lo, hi, (n_samples * 4, 2))
    inside = shapely.contains(region, shapely.points(uv[:, 0], uv[:, 1]))
    uv = uv[inside][:n_samples]
    if len(uv) == 0:
        return 0.0
    pts = lhf.origin + uv @ frame[:, :2].T - 1e-3 * frame[:, 2]
    return float(part_solid.contains(pts).mean())


# --------------------------------------------------------------------------- #
# The repair operation itself.
# --------------------------------------------------------------------------- #
@dataclass
class RepairEval:
    """What an authored patch pocket costs and whether it does the job."""

    repaired: trimesh.Trimesh | None      # part after the pocket is milled out
    patch: trimesh.Trimesh | None         # the Dutchman: material the pocket removed
    damage_total: float                   # decayed material inside the part
    damage_left: float                    # decayed material the pocket missed
    sound_removed: float                  # healthy material sacrificed
    patch_volume: float

    def text(self) -> str:
        frac = self.damage_left / max(self.damage_total, 1e-9)
        return (f"damage in part   {self.damage_total:.4f}\n"
                f"damage LEFT      {self.damage_left:.4f}  ({frac * 100:5.1f}% missed)\n"
                f"sound removed    {self.sound_removed:.4f}  <- minimize this\n"
                f"patch volume     {self.patch_volume:.4f}")


def _vol(m) -> float:
    if m is None or m.is_empty:
        return 0.0
    try:
        v = float(m.volume)
    except Exception:
        return 0.0
    return abs(v) if np.isfinite(v) else 0.0


def _isect(a, b):
    try:
        return a.intersection(b)
    except Exception:
        return None


def evaluate_repair(part_solid: trimesh.Trimesh, damage: trimesh.Trimesh,
                    pocket: trimesh.Trimesh) -> RepairEval:
    """Score a patch pocket against a damage volume on a part.

    The objective mirrors `config.ENERGY_WEIGHTS`: `w_sound` is the primary
    minimum-intervention term, so a pocket that swallows the damage while taking
    the least healthy wood wins.
    """
    dmg_in = _isect(part_solid, damage)
    dmg_total = _vol(dmg_in)
    try:
        repaired = part_solid.difference(pocket)
    except Exception:
        repaired = None
    patch = _isect(part_solid, pocket)
    left = 0.0
    if dmg_in is not None and repaired is not None:
        left = _vol(_isect(repaired, dmg_in))
    caught = _vol(_isect(patch, dmg_in)) if (patch is not None and dmg_in is not None) else 0.0
    return RepairEval(repaired=repaired, patch=patch, damage_total=dmg_total,
                      damage_left=left, sound_removed=max(0.0, _vol(patch) - caught),
                      patch_volume=_vol(patch))


# --------------------------------------------------------------------------- #
# Canonicalization: making a READ LHF safe to WRITE.
# --------------------------------------------------------------------------- #
def canonicalize(lhf: LHF, ori: int) -> LHF:
    """Rewrite an LHF read at orientation `ori` into the equivalent one at ori=0.

    This is the bridge between the two halves of the repo, and it is not
    optional. A cut read out of the dataset carries a *resolved* orientation
    (e.g. CJ_DT's `lhf_1` resolves to ori=3), but that index lives only in
    `out/orient_cache/`, not in the file. Emit such a cut verbatim next to a
    freshly authored one and you get a file whose LHFs disagree about which
    frame their sketches are in -- unreadable without the original STL.

    The fix is a pure 2D change of basis on the rings: the plane, the normal and
    the extrusion depth are untouched (`orientation_frame` always keeps the
    normal in column 2, so `ori` only ever permutes the in-plane axes). After
    this, every LHF in the part means what it says at ori=0.
    """
    if ori == 0:
        return lhf
    f0 = orientation_frame(lhf.normal, 0)[:, :2]
    fo = orientation_frame(lhf.normal, ori)[:, :2]
    m = f0.T @ fo                           # sketch(ori) coords -> sketch(0) coords
    # `m` is provably a signed permutation matrix: the rotations are multiples of
    # 90deg, so every entry is exactly 0 or +-1. Snap it. Without this, entries
    # come out as -0.0 and 6e-17, the rotated ring picks up last-bit noise, and
    # the extruded solid triangulates with degenerate slivers -- non-watertight,
    # which makes manifold refuse the boolean and `build_solid` silently return
    # the *uncut stock*. Cost us a 100% volume error before we caught it.
    snapped = np.round(m)
    if np.abs(m - snapped).max() > 1e-9 or not np.array_equal(
            np.abs(snapped).sum(axis=0), np.ones(2)):
        raise AssertionError(f"orientation map is not a signed permutation:\n{m}")
    m = snapped.T                           # rows-of-points convention
    return LHF(name=lhf.name, rings=[r @ m for r in lhf.rings], signs=list(lhf.signs),
               origin=lhf.origin.copy(), normal=lhf.normal.copy(), amount=lhf.amount)


def canonical_part(name: str, stock: LHF, stock_ori: int, cuts: list[LHF],
                   cut_oris: list[int], extra_cuts: list[LHF] | None = None) -> dict:
    """A jwood part dict that is correct at ori=0 with no STL alongside it.

    `extra_cuts` (already authored in the seed basis) are appended as-is. This is
    the emission path a repair pipeline needs: read a joint, add a patch pocket,
    write a file the reader can interpret on its own.
    """
    return to_jwood_part(
        name,
        canonicalize(stock, stock_ori),
        [canonicalize(c, o) for c, o in zip(cuts, cut_oris)] + list(extra_cuts or []),
    )


# --------------------------------------------------------------------------- #
# Export: a self-contained LHF parametrization on disk.
# --------------------------------------------------------------------------- #
EXPORT_SCHEMA = "repair.lhf/1"

_EXPORT_NOTE = (
    "Every LHF here is CANONICAL: its sketch coordinates are expressed in "
    "jwood.seed_basis(plane_normal), i.e. orientation index 0. Evaluate with "
    "ori=0 for all LHFs -- no STL and no orientation search required. This is the "
    "difference from the dataset's own vis_files/*_jwood.json, which omit the "
    "in-plane orientation and so cannot be evaluated without the ground-truth mesh."
)


def joint_export_dict(key: str, variant: str = "base", relational: bool = True) -> dict:
    """A self-contained parametrization of one joint, ready to serialize.

    `parts` keeps the dataset's own shape, so `jwood.PartSpec.from_json` reads an
    entry unchanged -- the extra top-level keys are additive and a plain jwood
    reader ignores them:

      parts        jwood-compatible, but canonicalized to ori=0
      provenance   what the resolver recovered, so the export is auditable
      relational   the recovered members / interfaces / fillers
    """
    from repair.relational import DATASET_ROOT, import_joint, reconstruct_parts

    recons = reconstruct_parts(key, variant)
    parts, provenance = [], {}
    for r in recons:
        parts.append(canonical_part(r.name, r.spec.stock, r.stock_ori,
                                    r.spec.cuts, r.cut_oris))
        provenance[r.name] = {
            "source_expression": r.spec.expression,
            "resolved_stock_ori": int(r.stock_ori),
            "resolved_cut_oris": [int(o) for o in r.cut_oris],
            "stl_agreement": round(float(r.agreement), 6),
            "vol_error": round(float(r.vol_error), 6),
        }

    out = {
        "schema": EXPORT_SCHEMA,
        "key": key,
        "variant": variant,
        "source": str(DATASET_ROOT / key / "vis_files" / f"{variant}_jwood.json"),
        "note": _EXPORT_NOTE,
        "parts": parts,
        "provenance": provenance,
    }
    if relational:
        try:
            j = import_joint(key, variant)
            out["relational"] = {
                "members": [{"name": m.name, "stock_volume": round(m.stock_volume, 6),
                             "solid_volume": round(m.solid_volume, 6),
                             "is_filler": bool(m.is_filler)}
                            for m in j.members.values()],
                "interfaces": [{"id": i.id, "volume": round(i.volume, 6),
                                "incidences": [{"part": c.part, "role": c.role,
                                                "volume": round(c.volume, 6)}
                                               for c in i.incidences]}
                               for i in j.interfaces],
            }
        except Exception as e:
            out["relational"] = {"error": repr(e)}
    return out


def verify_export(d: dict, variant: str = "base") -> list[dict]:
    """Rebuild every part straight from an export dict and compare to the STL.

    The only honest check: read the file with the unmodified reader, evaluate at
    ori=0 with no orientation search, and see whether the solid matches the
    ground truth. Returns one row per part.
    """
    import trimesh

    from repair.jwood import PartSpec
    from repair.relational import DATASET_ROOT, build_solid

    rows = []
    for i, p in enumerate(d["parts"]):
        spec = PartSpec.from_json(p)
        solid, _, _ = build_solid(spec.stock, 0, spec.cuts, [0] * len(spec.cuts))
        stl = trimesh.load(DATASET_ROOT / d["key"] / "meshes" / variant / f"{i}.stl",
                           force="mesh")
        gt = abs(float(stl.volume))
        rows.append({"part": spec.name, "volume": float(solid.volume), "stl_volume": gt,
                     "vol_error": abs(float(solid.volume) - gt) / max(gt, 1e-9)})
    return rows


def export_joint(key: str, variant: str = "base", out_dir=None,
                 relational: bool = True):
    """Write one joint's canonical parametrization; returns (path, verify_rows)."""
    import json
    from pathlib import Path

    from repair.config import OUT_DIR

    out_dir = Path(out_dir) if out_dir is not None else OUT_DIR / f"lhf_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = joint_export_dict(key, variant, relational=relational)
    path = out_dir / f"{key}.json"
    path.write_text(json.dumps(d, indent=1))
    return path, verify_export(d, variant)


def load_export(path) -> dict:
    """Read an export written by `export_joint`, checking the schema tag."""
    import json
    from pathlib import Path

    d = json.loads(Path(path).read_text())
    got = d.get("schema")
    if got != EXPORT_SCHEMA:
        raise ValueError(f"{path}: expected schema {EXPORT_SCHEMA!r}, got {got!r}")
    return d


def export_part_solids(d: dict) -> list[dict]:
    """Evaluate every part of an export dict into meshes.

    Touches no dataset file: the export is canonical, so every LHF evaluates at
    ori=0 and nothing has to be resolved against a ground-truth mesh. This is the
    function that makes the exports stand on their own.
    """
    from repair.jwood import PartSpec
    from repair.relational import build_solid

    out = []
    for p in d["parts"]:
        spec = PartSpec.from_json(p)
        solid, cut_union, cut_meshes = build_solid(spec.stock, 0, spec.cuts,
                                                   [0] * len(spec.cuts))
        out.append({"name": spec.name, "spec": spec, "solid": solid,
                    "stock": lhf_to_mesh(spec.stock, 0), "cut_meshes": cut_meshes,
                    "cut_union": cut_union})
    return out
