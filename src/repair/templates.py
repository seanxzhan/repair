"""Parametric repair templates: the discrete repair vocabulary.

Each template, given a member and parameters ``theta``, produces a *cut region* (the
volume to remove = the new insert ``I``). The cut is anchored to the damage so coverage
is achievable; ``theta`` tunes its extent/shape. ``K = M \\ I`` is the kept original.

The templates differ mainly in *interface shape*, which is what the proxy energy scores:
  * LedgeStep      - rectangular block, short L-shaped interface (a re-entrant corner).
  * Scarf          - long slanted interface (structurally strong, removes more sound wood).
  * DovetailLap    - half-lap with a flared (interlocking) inner face.
  * DutchmanInlay  - closed interior patch around the rot (smallest removal for central rot).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, MultiLineString, Polygon, box

from .dataset import Member
from .damage import Damage

PAD = 5.0  # clip slop so cut polygons cleanly exceed member bounds


@dataclass
class CutResult:
    kept: Polygon
    insert: Polygon
    interface: LineString | MultiLineString  # new cut surface inside the member
    feasible: bool


def _finish(member: Member, cut_region: Polygon) -> CutResult:
    M = member.poly_shapely
    insert = M.intersection(cut_region)
    kept = M.difference(cut_region)
    feasible = (not insert.is_empty) and (not kept.is_empty) and insert.area > 1e-5
    # interface = boundary of the insert that lies strictly inside M (the new surface)
    if insert.is_empty:
        iface = LineString()
    else:
        iface = insert.boundary.difference(M.boundary.buffer(1e-6))
    return CutResult(kept=kept, insert=insert, interface=iface, feasible=feasible)


def _anchor(member: Member, damage: Damage) -> dict:
    x0, y0, x1, y1 = member.bbox
    cx, cy = damage.centroid
    dminx, dminy, dmaxx, dmaxy = damage.must_replace.bounds
    return dict(x0=x0, y0=y0, x1=x1, y1=y1, W=x1 - x0, H=y1 - y0,
                right=abs(cx - x1) <= abs(cx - x0),
                top=abs(cy - y1) <= abs(cy - y0),
                d=(dminx, dminy, dmaxx, dmaxy))


class Template(Protocol):
    name: str
    n_params: int
    bounds: list[tuple[float, float]]
    def default_theta(self, member: Member, damage: Damage) -> np.ndarray: ...
    def apply(self, member: Member, damage: Damage, theta: np.ndarray) -> CutResult: ...


def _clip(theta, bounds):
    return np.array([min(max(t, lo), hi) for t, (lo, hi) in zip(theta, bounds)])


# --------------------------------------------------------------------- ledge

class LedgeStep:
    name = "ledge"
    n_params = 2
    bounds = [(0.05, 1.0), (0.05, 1.0)]

    def default_theta(self, member, damage):
        a = _anchor(member, damage)
        dminx, dminy, dmaxx, dmaxy = a["d"]
        length = ((a["x1"] - dminx) if a["right"] else (dmaxx - a["x0"])) / a["W"] + 0.06
        depth = ((a["y1"] - dminy) if a["top"] else (dmaxy - a["y0"])) / a["H"] + 0.06
        return _clip([length, depth], self.bounds)

    def apply(self, member, damage, theta):
        length, depth = _clip(theta, self.bounds)
        a = _anchor(member, damage)
        bx0, bx1 = (a["x1"] - length * a["W"], a["x1"] + PAD) if a["right"] \
            else (a["x0"] - PAD, a["x0"] + length * a["W"])
        by0, by1 = (a["y1"] - depth * a["H"], a["y1"] + PAD) if a["top"] \
            else (a["y0"] - PAD, a["y0"] + depth * a["H"])
        return _finish(member, box(bx0, by0, bx1, by1))


# --------------------------------------------------------------------- scarf

class Scarf:
    name = "scarf"
    n_params = 2
    # slope is bounded away from 0: a zero-slope scarf is just a butt joint, not a scarf.
    bounds = [(0.1, 1.0), (0.28, 0.7)]  # length, slope (tan of tilt)

    def default_theta(self, member, damage):
        a = _anchor(member, damage)
        dminx, dminy, dmaxx, dmaxy = a["d"]
        length = ((a["x1"] - dminx) if a["right"] else (dmaxx - a["x0"])) / a["W"] + 0.12
        return _clip([length, 0.4], self.bounds)

    def apply(self, member, damage, theta):
        length, slope = _clip(theta, self.bounds)
        a = _anchor(member, damage)
        # Extend only slightly in y (eps) so the slope actually tilts the in-member
        # interface; a large PAD here would dilute the slant to near-vertical.
        eps = 0.02
        y_lo, y_hi = a["y0"] - eps, a["y1"] + eps
        dx = slope * a["H"]                       # horizontal run across the height
        if a["right"]:
            xc = a["x1"] - length * a["W"]
            poly = Polygon([(xc - dx, y_lo), (xc + dx, y_hi),
                            (a["x1"] + PAD, y_hi), (a["x1"] + PAD, y_lo)])
        else:
            xc = a["x0"] + length * a["W"]
            poly = Polygon([(xc + dx, y_lo), (xc - dx, y_hi),
                            (a["x0"] - PAD, y_hi), (a["x0"] - PAD, y_lo)])
        return _finish(member, poly)


# ----------------------------------------------------------------- dovetail

class DovetailLap:
    name = "dovetail"
    n_params = 2
    # A full-height end key with an interlocking tab that protrudes into the kept piece
    # (non-convex) -- mechanically resists pull-out, unlike a plain butt or scarf.
    bounds = [(0.1, 1.0), (0.12, 0.5)]  # length, key (tab protrusion as fraction of length)

    def default_theta(self, member, damage):
        a = _anchor(member, damage)
        dminx, dminy, dmaxx, dmaxy = a["d"]
        length = ((a["x1"] - dminx) if a["right"] else (dmaxx - a["x0"])) / a["W"] + 0.1
        return _clip([length, 0.3], self.bounds)

    def apply(self, member, damage, theta):
        length, key = _clip(theta, self.bounds)
        a = _anchor(member, damage)
        ymid = 0.5 * (a["y0"] + a["y1"])
        neck = 0.18 * a["H"]                      # half-height of the interlock tab
        tab = key * length * a["W"]              # how far the tab hooks into the kept piece
        if a["right"]:
            xend, xin = a["x1"] + PAD, a["x1"] - length * a["W"]
            xtab = xin - tab
            poly = Polygon([(xend, a["y0"] - PAD), (xend, a["y1"] + PAD),
                            (xin, a["y1"] + PAD), (xin, ymid + neck),
                            (xtab, ymid + neck), (xtab, ymid - neck),
                            (xin, ymid - neck), (xin, a["y0"] - PAD)])
        else:
            xend, xin = a["x0"] - PAD, a["x0"] + length * a["W"]
            xtab = xin + tab
            poly = Polygon([(xend, a["y0"] - PAD), (xin, a["y0"] - PAD),
                            (xin, ymid - neck), (xtab, ymid - neck),
                            (xtab, ymid + neck), (xin, ymid + neck),
                            (xin, a["y1"] + PAD), (xend, a["y1"] + PAD)])
        return _finish(member, poly)


# ----------------------------------------------------------------- dutchman

class DutchmanInlay:
    name = "dutchman"
    n_params = 3
    bounds = [(0.02, 0.4), (0.02, 0.4), (-0.4, 0.4)]  # margin_x, margin_y, draft

    def default_theta(self, member, damage):
        return np.array([0.08, 0.08, 0.0])

    def apply(self, member, damage, theta):
        mx, my, draft = _clip(theta, self.bounds)
        a = _anchor(member, damage)
        dminx, dminy, dmaxx, dmaxy = a["d"]
        px0, px1 = dminx - mx * a["W"], dmaxx + mx * a["W"]
        py0, py1 = dminy - my * a["H"], dmaxy + my * a["H"]
        # draft slants the side walls into a trapezoid
        s = draft * (px1 - px0)
        poly = Polygon([(px0 - s, py0), (px1 + s, py0), (px1 - s, py1), (px0 + s, py1)])
        cut = _finish(member, poly)
        # An inlay is only valid for INTERIOR damage: it must be surrounded by kept wood.
        # If the patch opens onto a free edge it's really a splice, not an inlay -> reject.
        if cut.feasible:
            shared = cut.insert.boundary.intersection(member.poly_shapely.boundary).length
            if shared > 0.05 * member.poly_shapely.boundary.length:
                cut.feasible = False
        return cut


TEMPLATES: dict[str, Template] = {
    t.name: t for t in (LedgeStep(), Scarf(), DovetailLap(), DutchmanInlay())
}
TEMPLATE_NAMES = list(TEMPLATES.keys())  # stable order = class-label index


def label_index(name: str) -> int:
    return TEMPLATE_NAMES.index(name)
