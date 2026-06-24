"""Reading and evaluating the MiGumi `*_jwood.json` parametrization.

A joint folder stores each part as a CSG `expression` over Linked Height Fields
(LHFs). An LHF is a 2D region (`polysets` + `poly_signs`) embedded on a plane
(`plane_origin`, `plane_normal`) and extruded by `amount` along that normal.

Facts that hold across the whole dataset and the importer relies on:

  * **Stock vs. cut.** Every part's expression is `Difference(lhf_0, <rest>)`,
    so `lhf_0` is the raw stock block and every other LHF is *subtracted*
    (a joinery cut). Verified for all 65 parts / 30 joints.

  * **Assembled frame = identity.** The LHF coordinates are already in the
    shared assembled frame; `state_map` holds assembly-animation keyframes
    (parts pulled apart), not the joint pose. So we evaluate at identity.

  * **2D region.** `polysets` is a list of rings and `poly_signs` gives each
    ring's sign: +1 outer boundary, -1 hole. The region is the union of the
    positive rings minus the negative rings (holes). Earlier code took only the
    first ring, which dropped holes (e.g. the pin hole in CJ_AKT's key) and
    disjoint lobes (e.g. CJ_MT) -- a source of reconstruction error.

The one genuinely ambiguous degree of freedom is the **in-plane orientation** of
the 2D sketch on its plane: `plane_normal` fixes the sketch plane but not how the
sketch's local (x, y) axes map into it. There are 8 candidates (4 rotations x an
optional reflection). The dataset uses a deterministic per-normal convention --
for orientation-pinned (asymmetric) features it is unanimous:

    +X -> 1   -X -> 3   -Y -> 1   +Z -> 0   -Z -> 0   (+Y only ever symmetric)

but symmetric features (square stock, 180deg-symmetric cuts) admit several
equivalent orientations, and 45deg-normal joints need their own value. Rather
than hard-code a partial table we *resolve* the orientation of every LHF against
the STL ground truth (`relational.resolve_orientations`), which is robust,
covers the 45deg cases, and yields exact reconstructions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import shapely
import trimesh
import trimesh.repair
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

# In-plane orientation candidates: 4 rotations (0..3) x optional reflection (4..7).
N_ORIENT = 8


@dataclass
class LHF:
    """A Linked Height Field: a 2D region (with holes) extruded along a normal."""

    name: str
    rings: list[np.ndarray]    # list of (n, 2) sketch-plane polygons
    signs: list[int]           # +1 outer, -1 hole, per ring
    origin: np.ndarray         # (3,) plane origin in assembled coords
    normal: np.ndarray         # (3,) unit plane normal
    amount: float              # signed extrusion distance along the normal

    @classmethod
    def from_json(cls, name: str, d: dict) -> "LHF":
        n = np.asarray(d["plane_normal"], float)
        n = n / np.linalg.norm(n)
        rings = [np.asarray(r, float)[:, :2] for r in d["polysets"]]
        signs = list(d.get("poly_signs", [1] * len(rings)))
        return cls(name=name, rings=rings, signs=signs,
                   origin=np.asarray(d["plane_origin"], float),
                   normal=n, amount=float(d["amount"][0]))

    def region_2d(self):
        """The 2D sketch region as a shapely (Multi)Polygon: + rings minus holes."""
        pos = [Polygon(r) for r, s in zip(self.rings, self.signs) if s > 0]
        g = unary_union(pos) if pos else Polygon()
        for r, s in zip(self.rings, self.signs):
            if s < 0:
                g = g.difference(Polygon(r))
        return g


def seed_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A deterministic right-handed frame (u, v, n) with ``u x v == n``."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v, n


def orientation_frame(normal: np.ndarray, ori: int) -> np.ndarray:
    """3x3 frame whose columns are (u, v, n) for orientation index ``ori`` (0..7).

    ori % 4 rotates u in the plane by 0/90/180/270 deg; ori >= 4 reflects v.
    """
    u0, v0, n = seed_basis(normal)
    t = (ori % 4) * (np.pi / 2.0)
    u = np.cos(t) * u0 + np.sin(t) * v0
    v = np.cross(n, u)
    if ori >= 4:
        v = -v
    return np.column_stack([u, v, n])


def _placement(lhf: LHF, ori: int) -> tuple[np.ndarray, np.ndarray]:
    """(3x3 frame with extrude axis in col 2, origin) for the given orientation."""
    frame = orientation_frame(lhf.normal, ori).copy()
    if lhf.amount < 0:
        frame[:, 2] = -frame[:, 2]
    return frame, lhf.origin


def _make_watertight(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Clean a mesh into a valid volume so manifold booleans accept it.

    Extruded polygons can carry a duplicated closing vertex (e.g. the round
    dowel in RJM_SDT) which yields a degenerate face and a non-watertight solid;
    manifold then refuses the boolean. Merging vertices and dropping
    degenerate/duplicate faces restores watertightness.
    """
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
    mesh.fix_normals()
    return mesh


def lhf_to_mesh(lhf: LHF, ori: int = 0) -> trimesh.Trimesh:
    """Extruded solid for an LHF (all rings/holes) under orientation ``ori``."""
    region = lhf.region_2d()
    polys = list(region.geoms) if isinstance(region, MultiPolygon) else [region]
    solids = [trimesh.creation.extrude_polygon(p, height=abs(lhf.amount))
              for p in polys if p.area > 1e-12]
    mesh = trimesh.util.concatenate(solids) if len(solids) > 1 else solids[0]
    frame, origin = _placement(lhf, ori)
    T = np.eye(4)
    T[:3, :3] = frame
    T[:3, 3] = origin
    mesh.apply_transform(T)
    return _make_watertight(mesh)


def lhf_contains(lhf: LHF, ori: int, points: np.ndarray) -> np.ndarray:
    """Boolean mask: which `points` (N,3) lie inside the LHF solid (orientation ori).

    Analytic (no mesh booleans): project into the plane frame, test the 2D
    region membership and the extrude height range. Robust where mesh booleans
    fail on coplanar faces.
    """
    frame, origin = _placement(lhf, ori)
    rel = points - origin
    uv = rel @ frame[:, :2]               # in-plane coords
    h = rel @ frame[:, 2]                  # height along (signed) extrude axis
    in_2d = shapely.contains(lhf.region_2d(), shapely.points(uv[:, 0], uv[:, 1]))
    return in_2d & (h >= -1e-9) & (h <= abs(lhf.amount) + 1e-9)


# Matches `lhf_<n>` tokens; the first one is the stock by the Difference rule.
_LHF_TOKEN = re.compile(r"lhf_\d+")


def parse_expression(expr: str) -> tuple[str, list[str]]:
    """Split a part expression into (stock_name, [cut_names]).

    Every part is ``Difference(lhf_0, <rest>)`` where <rest> is a single LHF or
    ``Union(...)`` of LHFs. The first token is stock, the remainder are cuts.
    """
    tokens = _LHF_TOKEN.findall(expr)
    if not tokens or not expr.startswith("Difference("):
        raise ValueError(f"unexpected expression form: {expr!r}")
    return tokens[0], tokens[1:]


@dataclass
class PartSpec:
    """Parsed jwood for one part: its stock LHF and its cut LHFs."""

    name: str
    expression: str
    stock: LHF
    cuts: list[LHF]

    @classmethod
    def from_json(cls, d: dict) -> "PartSpec":
        stock_name, cut_names = parse_expression(d["expression"])
        lhfs = {k: LHF.from_json(k, v) for k, v in d["lhfs"].items()}
        return cls(name=d["name"], expression=d["expression"],
                   stock=lhfs[stock_name], cuts=[lhfs[c] for c in cut_names])
