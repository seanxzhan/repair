"""Procedural damage (rot) and natural features (knots, checks) on a member.

Rot is *not* uniform-random: it follows realistic modes -- central rot, end rot,
and ground-contact (bottom-edge) rot -- so the learned prior trains on a distribution
that resembles reality. ``must_replace`` is the union of rot regions clipped to the
member; it is the hard coverage constraint any repair must remove.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from .dataset import Member

DAMAGE_KINDS = ["central_rot", "end_rot", "ground_contact"]


@dataclass
class Damage:
    must_replace: Polygon              # union of rot regions (hard coverage constraint)
    features: list = field(default_factory=list)  # knots/checks (soft penalty)
    kind: str = "central_rot"

    @property
    def centroid(self) -> np.ndarray:
        c = self.must_replace.representative_point()
        return np.array([c.x, c.y])


def _rand_interior_point(member: Member, rng) -> Point:
    x0, y0, x1, y1 = member.bbox
    # bias toward the right/interior so the left-end mate face is usually preserved
    for _ in range(200):
        p = Point(rng.uniform(x0 + 0.3 * (x1 - x0), x1), rng.uniform(y0, y1))
        if member.poly_shapely.contains(p):
            return p
    return member.poly_shapely.representative_point()


def central_rot(member: Member, rng, radius_frac=(0.16, 0.28)) -> Polygon:
    """Hidden interior rot (per the paper: detected by drilling, not visible). Kept away
    from the free edges so it is a genuine interior pocket suitable for an inlay."""
    from shapely import affinity
    x0, y0, x1, y1 = member.bbox
    W, H = x1 - x0, y1 - y0
    r = rng.uniform(*radius_frac) * H                 # radius relative to height
    seed = Point(rng.uniform(x0 + 0.45 * W, x0 + 0.80 * W),
                 rng.uniform(y0 + 0.40 * H, y0 + 0.60 * H))
    blob = affinity.scale(seed.buffer(r), 1.7, 0.85, origin=seed)  # elongate along grain
    return blob.intersection(member.poly_shapely)


def end_rot(member: Member, rng, depth_frac=(0.15, 0.32)) -> Polygon:
    x0, y0, x1, y1 = member.bbox
    depth = rng.uniform(*depth_frac) * (x1 - x0)
    # right end (the left end is the frozen mate face)
    band = Polygon([(x1 - depth, y0 - 1), (x1 + 1, y0 - 1),
                    (x1 + 1, y1 + 1), (x1 - depth, y1 + 1)])
    return band.intersection(member.poly_shapely)


def ground_contact_rot(member: Member, rng, depth_frac=(0.15, 0.30)) -> Polygon:
    x0, y0, x1, y1 = member.bbox
    depth = rng.uniform(*depth_frac) * (y1 - y0)
    # a band along the bottom edge, on the right portion (left end is the mate face)
    lx0 = rng.uniform(x0 + 0.35 * (x1 - x0), x0 + 0.55 * (x1 - x0))
    lx1 = rng.uniform(x1 - 0.2 * (x1 - x0), x1)
    band = Polygon([(lx0, y0 - 1), (lx1, y0 - 1), (lx1, y0 + depth), (lx0, y0 + depth)])
    return band.intersection(member.poly_shapely)


def add_knots(member: Member, rng, n=(0, 3)) -> list[Polygon]:
    out = []
    for _ in range(rng.integers(n[0], n[1] + 1)):
        p = _rand_interior_point(member, rng)
        out.append(p.buffer(rng.uniform(0.03, 0.07)).intersection(member.poly_shapely))
    return [g for g in out if not g.is_empty]


def add_checks(member: Member, rng, n=(0, 2)) -> list[Polygon]:
    """Drying cracks: thin slivers along the grain (long axis)."""
    x0, y0, x1, y1 = member.bbox
    out = []
    for _ in range(rng.integers(n[0], n[1] + 1)):
        cy = rng.uniform(y0, y1)
        cx0 = rng.uniform(x0, (x0 + x1) / 2)
        cx1 = rng.uniform((x0 + x1) / 2, x1)
        line = LineString([(cx0, cy), (cx1, cy)])
        out.append(line.buffer(0.012).intersection(member.poly_shapely))
    return [g for g in out if not g.is_empty]


_MAKERS = {"central_rot": central_rot, "end_rot": end_rot,
           "ground_contact": ground_contact_rot}
MIN_ROT_AREA = 0.01  # fraction of member area; below this, resample


def sample_damage(member: Member, rng, kind: str | None = None) -> Damage:
    want = kind
    min_area = MIN_ROT_AREA * member.poly_shapely.area
    rot = None
    for attempt in range(8):
        k = want or DAMAGE_KINDS[rng.integers(0, len(DAMAGE_KINDS))]
        g = _MAKERS[k](member, rng).intersection(member.poly_shapely)
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda x: x.area)
        if (not g.is_empty) and g.area >= min_area:
            rot, kind = g, k
            break
        if attempt >= 3:           # give up on the requested kind; force central rot
            want = "central_rot"
    if rot is None:                # last resort: a blob at the member centre
        rot = central_rot(member, rng)
        kind = "central_rot"
    features = add_knots(member, rng) + add_checks(member, rng)
    return Damage(must_replace=rot, features=features, kind=kind)
