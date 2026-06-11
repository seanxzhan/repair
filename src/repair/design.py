"""Forward model for the *network-output* representation: a blank timber column cut by
a predicted set of cuts into mating pieces.

This mirrors what a model would emit from a blank (later: damaged) column. Each :class:`Cut`
is exactly one MXG millable extrusion -- a plane (``normal`` + ``offset``) bearing a 2D
rectangular ``profile`` extruded a ``depth`` along the normal -- which is the parametrization
we recommended a network predict (normal as a class over ``CANONICAL_NORMALS`` + a few
scalars; the in-plane frame is *derived* from the normal, so it is never predicted).

Pieces are a PARTITION of the stock, so they always mate (no overlap, shared surfaces):

    piece 0 (kept base) = stock - union(all cut regions)
    piece j >= 1        = the cut regions labelled j  (carved against already-claimed space)

So "predict the cuts" directly yields interlocking pieces -- mating is structural, not a
constraint the network must learn to satisfy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from .jwood import _BOOL, _clean, get_frame_from_normal

# The direction vocabulary a classifier head would choose from (6 axes + 8 face diagonals).
# A network predicts an index here (+ optional residual), not a free vector.
CANONICAL_NORMALS: dict[str, tuple[float, float, float]] = {
    "+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0), "-Y": (0, -1, 0),
    "+Z": (0, 0, 1), "-Z": (0, 0, -1),
    "+X+Z": (0.7071, 0, 0.7071), "+X-Z": (0.7071, 0, -0.7071),
    "-X+Z": (-0.7071, 0, 0.7071), "-X-Z": (-0.7071, 0, -0.7071),
    "+X+Y": (0.7071, 0.7071, 0), "+X-Y": (0.7071, -0.7071, 0),
    "+Y+Z": (0, 0.7071, 0.7071), "+Y-Z": (0, 0.7071, -0.7071),
}


@dataclass
class Stock:
    """The blank timber column, centred at the origin; long axis = Y."""
    length: float = 3.0     # Y
    width: float = 1.0      # X
    height: float = 1.0     # Z

    def solid(self) -> trimesh.Trimesh:
        return _clean(trimesh.creation.box(extents=[self.width, self.length, self.height]))

    @property
    def span(self) -> float:
        return float(max(self.length, self.width, self.height))


@dataclass
class Cut:
    """One predicted millable extrusion = a plane + an in-plane rectangle + a depth.

    Fields map 1:1 to a network's output: ``normal`` (a class in CANONICAL_NORMALS),
    ``offset`` (plane position along the normal, from stock centre), ``(uc, vc, uw, vh)``
    (the 2D profile rectangle in the plane's *derived* frame), ``depth`` (extrusion
    length along +normal), and ``piece`` (which piece this carved region becomes)."""
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    offset: float = 0.0
    uc: float = 0.0
    vc: float = 0.0
    uw: float = 2.0         # profile size in-plane (u); large => spans the cross-section
    vh: float = 2.0         # profile size in-plane (v)
    depth: float = 4.0      # extrude length along +normal; large => a through cut
    piece: int = 1          # label of the piece this region becomes (>=1)


def cut_solid(cut: Cut) -> trimesh.Trimesh:
    """The 3D region removed by a cut: a (uw x vh) rectangle on the plane at ``offset``,
    extruded ``depth`` along +normal."""
    u, v, n = get_frame_from_normal(cut.normal)
    box = trimesh.creation.box(extents=[cut.uw, cut.vh, cut.depth])
    centre = cut.offset * n + cut.uc * u + cut.vc * v + (cut.depth / 2.0) * n
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = u, v, n, centre
    box.apply_transform(T)
    return _clean(box)


def _union(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh | None:
    meshes = [m for m in meshes if m is not None and len(m.vertices)]
    if not meshes:
        return None
    out = meshes[0]
    for m in meshes[1:]:
        try:
            out = _clean(trimesh.boolean.union([out, m], engine=_BOOL))
        except Exception:
            pass
    return out


def _diff(a: trimesh.Trimesh, b: trimesh.Trimesh | None) -> trimesh.Trimesh:
    if b is None or not len(b.vertices):
        return a
    try:
        return _clean(trimesh.boolean.difference([a, b], engine=_BOOL))
    except Exception:
        return a


def _intersect(a: trimesh.Trimesh, b: trimesh.Trimesh) -> trimesh.Trimesh | None:
    try:
        r = _clean(trimesh.boolean.intersection([a, b], engine=_BOOL))
        return r if len(r.vertices) else None
    except Exception:
        return None


def pieces_from_cuts(stock: Stock, cuts: list[Cut]) -> list[trimesh.Trimesh]:
    """Partition the stock into mating pieces. Index 0 is the kept base; index j>=1 is
    the union of cuts labelled j, carved against lower-index pieces so all are disjoint."""
    M = stock.solid()
    if not cuts:
        return [M]
    labels = sorted({c.piece for c in cuts})
    claimed = None                      # union of everything assigned to a piece so far
    labelled: list[trimesh.Trimesh] = []
    for label in labels:
        region = _union([cut_solid(c) for c in cuts if c.piece == label])
        region = _intersect(region, M) if region is not None else None
        region = _diff(region, claimed) if region is not None else None
        labelled.append(region if region is not None else trimesh.Trimesh())
        if region is not None and len(region.vertices):
            claimed = region if claimed is None else _union([claimed, region])
    base = _diff(M, claimed)
    return [base, *labelled]
