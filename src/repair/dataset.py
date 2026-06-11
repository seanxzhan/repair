"""Load MiGumi joints and extract a 2D longitudinal member profile.

This is the linchpin of the pipeline. A MiGumi joint is stored as a node graph,
but ``vis_files/<variant>_jwood.json`` exposes each part as a set of "linked
height fields" (LHFs): a 2D polygon (``polysets``) extruded along a ``plane_normal``.

Extraction recipe (verified against the data):
  * For the chosen part, pick the LHF whose polygon has the *most* vertices -- this
    is empirically the detailed cut profile (e.g. CJ_DT part0 -> lhf_2, 10 pts),
    rather than the plain cross-section square (lhf_0).
  * Each polyset is a *local* 2D polygon stored in columns [0, 1] (the third column is
    always 0); ``plane_normal``/``plane_origin``/``amount`` only describe how that profile
    is extruded and placed in 3D. So the 2D profile is always columns [0, 1].
  * Clean, orient CCW, normalize, and rotate so the member's long axis is horizontal.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely import affinity

from .config import DATASET_ROOT


@dataclass
class JointMeta:
    key: str
    name: str
    type: str
    n_parts: int


@dataclass
class Member:
    key: str
    part_name: str
    polygon: np.ndarray          # (N, 2) closed CCW ring (no duplicate last point)
    poly_shapely: Polygon        # cleaned, normalized polygon
    dropped_axis: int            # which world axis (0/1/2) was the plane normal
    plane_normal: np.ndarray
    meta: JointMeta

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = self.poly_shapely.bounds
        return x0, y0, x1, y1


# --------------------------------------------------------------------------- io

def list_joints(root: Path = DATASET_ROOT) -> list[str]:
    return sorted(p.name for p in root.iterdir() if (p / "info.json").exists())


def load_meta(key: str, root: Path = DATASET_ROOT) -> JointMeta:
    d = json.loads((root / key / "info.json").read_text())
    return JointMeta(key=key, name=d.get("name", key), type=d.get("type", "?"),
                     n_parts=int(d.get("n_parts", 1)))


def load_jwood(key: str, variant: str = "base", root: Path = DATASET_ROOT) -> dict:
    return json.loads((root / key / "vis_files" / f"{variant}_jwood.json").read_text())


# ------------------------------------------------------------------- profile

# The 2D profile always lives in the first two local coordinates.
KEEP_AXES = [0, 1]


def _richest_lhf(part: dict) -> dict:
    """The LHF whose largest polyset has the most vertices."""
    return max(part["lhfs"].values(),
               key=lambda lhf: max(len(p) for p in lhf["polysets"]))


def _clean_normalize(pts2d: np.ndarray) -> Polygon | None:
    """Make a valid, CCW, origin-centred polygon scaled so longest side -> 1,
    rotated so the long axis is horizontal (x)."""
    poly = Polygon(pts2d)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        return None
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)

    # Normalize: centroid -> origin, longest bbox side -> 1.0
    cx, cy = poly.centroid.coords[0]
    poly = affinity.translate(poly, -cx, -cy)
    x0, y0, x1, y1 = poly.bounds
    scale = max(x1 - x0, y1 - y0)
    poly = affinity.scale(poly, 1.0 / scale, 1.0 / scale, origin=(0, 0))

    # Long axis -> horizontal: swap x/y if taller than wide.
    x0, y0, x1, y1 = poly.bounds
    if (y1 - y0) > (x1 - x0):
        poly = affinity.affine_transform(poly, [0, 1, 1, 0, 0, 0])  # (x,y)->(y,x)

    if not poly.exterior.is_ccw:
        poly = Polygon(list(poly.exterior.coords)[::-1])
    return poly


def extract_profile(jwood: dict, meta: JointMeta, part_index: int = 0) -> Member:
    part = jwood["parts"][part_index]
    lhf = _richest_lhf(part)
    normal = np.asarray(lhf["plane_normal"], dtype=float)

    pts = np.asarray(max(lhf["polysets"], key=len), dtype=float)  # (N,3)
    pts2d = pts[:, KEEP_AXES]
    poly = _clean_normalize(pts2d)
    if poly is None:
        raise ValueError(f"{meta.key}: degenerate profile")

    ring = np.asarray(poly.exterior.coords)[:-1]  # drop duplicate closing point
    return Member(key=meta.key, part_name=part["name"], polygon=ring,
                  poly_shapely=poly, dropped_axis=2, plane_normal=normal, meta=meta)


def load_member(key: str, part_index: int = 0, variant: str = "base",
                root: Path = DATASET_ROOT) -> Member:
    meta = load_meta(key, root)
    jwood = load_jwood(key, variant, root)
    return extract_profile(jwood, meta, part_index)


def rectangular_member(w: float = 1.2, h: float = 0.5) -> Member:
    """A clean rectangular member -- a neutral canvas for showing template shapes without
    the visual noise of an irregular joint profile."""
    from shapely.geometry import box
    poly = box(-w / 2, -h / 2, w / 2, h / 2)
    ring = np.asarray(poly.exterior.coords)[:-1]
    return Member(key="RECT", part_name="test", polygon=ring, poly_shapely=poly,
                  dropped_axis=2, plane_normal=np.array([0., 0., 1.]),
                  meta=JointMeta("RECT", "Rectangle", "Test", 1))


def left_end_mate(member: Member) -> LineString:
    x0, y0, x1, y1 = member.bbox
    return LineString([(x0, y0), (x0, y1)])


def loadable_joints(root: Path = DATASET_ROOT) -> list[str]:
    """Joints whose part-0 profile loads under the v1 (axis-aligned) recipe."""
    keys = []
    for k in list_joints(root):
        try:
            load_member(k, root=root)
            keys.append(k)
        except Exception:
            pass
    return keys


# ------------------------------------------------------------- mate interface

def load_mate_interface(jwood: dict, member: Member, part_index: int = 0) -> LineString:
    """Frozen coupling face that any repair must preserve.

    v1 heuristic: the member's LEFT end face (min-x boundary). Repairs are driven from
    the right/interior (see ``damage``), so a normal repair leaves the mate intact while
    an over-long splice that reaches the left end is (correctly) infeasible. Illustrative,
    not a true contact analysis. The raw assembly transforms exist in ``jwood`` but live
    in a different frame than the normalized member, so we don't use them here."""
    x0, _, x1, _ = member.bbox
    coords = np.asarray(member.poly_shapely.exterior.coords)
    thresh = x0 + 0.05 * (x1 - x0)
    pts = coords[coords[:, 0] <= thresh]
    if len(pts) < 2:                       # widen until we capture the end face
        thresh = x0 + 0.15 * (x1 - x0)
        pts = coords[coords[:, 0] <= thresh]
    if len(pts) < 2:
        pts = coords[np.argsort(coords[:, 0])[:2]]
    pts = pts[np.argsort(pts[:, 1])]       # order along the face
    return LineString(pts)
