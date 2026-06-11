"""Proxy energy and hard constraints for a candidate repair.

The energy is multi-objective and deliberately encodes the *central tension*:
``sound_removed`` pulls toward small, tight cuts; ``structural`` rewards long load-
transferring interfaces; ``fabrication`` penalizes long/complex interfaces. The repair
that minimizes the total balances these, and *which template wins depends on the damage*.

Structural and grain terms are geometric proxies, NOT finite-element analysis (v1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import (GeometryCollection, LineString, MultiLineString,
                              Polygon)
from shapely.ops import unary_union

from .config import ENERGY_WEIGHTS, TOOL_RADIUS
from .dataset import Member
from .damage import Damage
from .templates import CutResult

BIG = 100.0  # penalty per violated hard constraint


@dataclass
class EnergyTerms:
    sound_removed: float = 0.0
    structural: float = 0.0
    fabrication: float = 0.0
    grain: float = 0.0
    defect: float = 0.0
    total: float = BIG
    feasible: bool = False
    violations: dict = field(default_factory=dict)

    @property
    def n_violations(self) -> int:
        return sum(1 for v in self.violations.values() if v)


def _segments(geom):
    lines = []
    if geom is None or geom.is_empty:
        return lines
    if isinstance(geom, (GeometryCollection, MultiLineString)):
        parts = list(geom.geoms)
    elif isinstance(geom, LineString):
        parts = [geom]
    else:
        parts = []
    for ls in parts:
        if not isinstance(ls, LineString):
            continue
        cs = np.asarray(ls.coords)
        for a, b in zip(cs[:-1], cs[1:]):
            lines.append((np.asarray(a), np.asarray(b)))
    return lines


def _min_feature_ok(poly: Polygon, r: float) -> bool:
    """Negative-buffer test: a poly with features thinner than 2r loses area when
    eroded then dilated."""
    if poly.is_empty:
        return False
    shrunk = poly.buffer(-r).buffer(r)
    if shrunk.is_empty:
        return False
    return shrunk.area >= 0.85 * poly.area


def check_constraints(member: Member, damage: Damage, cut: CutResult, mate: LineString,
                      tool_radius: float = TOOL_RADIUS) -> dict:
    v = {}
    if not cut.feasible:
        return {"feasible": True}
    # coverage: all must-replace rot is inside the removed insert
    uncovered = damage.must_replace.difference(cut.insert).area
    v["coverage"] = uncovered > 0.02 * member.poly_shapely.area
    # mate preserved: the removed region must not eat a meaningful length of the
    # coupling face (touching a corner is fine; removing the face is not)
    eaten = mate.intersection(cut.insert.buffer(tool_radius * 0.5)).length
    v["mate"] = eaten > 0.1 * mate.length
    # manufacturable: both the new insert and the remaining member are millable
    v["manufacturable"] = not (_min_feature_ok(cut.insert, tool_radius)
                               and _min_feature_ok(cut.kept, tool_radius))
    return v


def _interface_length(segs):
    return float(sum(np.linalg.norm(b - a) for a, b in segs))


def _grain_penalty(segs):
    """Fraction of interface (by length) that runs across the grain (vertical),
    i.e. a butt-like cut perpendicular to the long axis is penalized."""
    total = _interface_length(segs)
    if total < 1e-9:
        return 1.0
    cross = 0.0
    for a, b in segs:
        d = b - a
        n = np.linalg.norm(d)
        if n < 1e-9:
            continue
        cross += n * abs(d[1]) / n  # |component along y| share -> 1 if vertical
    return cross / total


def _corner_count(insert) -> int:
    """Corners beyond a simple quad, summed over polygon piece(s)."""
    if insert.is_empty:
        return 0
    polys = insert.geoms if insert.geom_type == "MultiPolygon" else [insert]
    return sum(max(0, len(p.exterior.coords) - 1 - 4) for p in polys)


def energy(member: Member, damage: Damage, cut: CutResult, mate: LineString,
           weights: dict = ENERGY_WEIGHTS) -> EnergyTerms:
    viol = check_constraints(member, damage, cut, mate)
    if not cut.feasible:
        return EnergyTerms(total=BIG * 3, feasible=False, violations={"feasible": True})

    area = member.poly_shapely.area
    segs = _segments(cut.interface)
    iface_len = _interface_length(segs)

    sound_removed = max(0.0, cut.insert.area
                        - cut.insert.intersection(damage.must_replace).area) / area
    structural = 1.0 / (iface_len + 0.1)                 # long interface -> low penalty
    fabrication = iface_len + 0.3 * _corner_count(cut.insert)
    grain = _grain_penalty(segs)
    # knots/checks left straddling the glue line
    band = cut.interface.buffer(TOOL_RADIUS) if not cut.interface.is_empty else None
    feats = unary_union(damage.features) if damage.features else None
    if band is not None and feats is not None and not feats.is_empty:
        defect = feats.intersection(band).intersection(cut.kept).area / area
    else:
        defect = 0.0

    total = (weights["w_sound"] * sound_removed
             + weights["w_struct"] * structural
             + weights["w_fab"] * fabrication
             + weights["w_grain"] * grain
             + weights["w_defect"] * defect
             + BIG * sum(1 for x in viol.values() if x))

    return EnergyTerms(sound_removed=sound_removed, structural=structural,
                       fabrication=fabrication, grain=grain, defect=defect,
                       total=float(total), feasible=True, violations=viol)
