"""Turn a (member, damage) pair into a fixed-size tensor for the CNN prior."""
from __future__ import annotations

import numpy as np
import shapely
from shapely.ops import unary_union

from .config import RASTER_RES
from .dataset import Member
from .damage import Damage

# Fixed normalized frame: members are centred at origin and scaled to longest side 1,
# so [-0.75, 0.75]^2 comfortably contains every profile.
FRAME = (-0.75, -0.75, 0.75, 0.75)


def _mask(geom, res, frame):
    x0, y0, x1, y1 = frame
    xs = np.linspace(x0, x1, res)
    ys = np.linspace(y0, y1, res)
    if geom is None or geom.is_empty:
        return np.zeros((res, res), dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)         # (res,res), row j = y, col i = x
    inside = shapely.contains_xy(geom, gx.ravel(), gy.ravel())
    return inside.reshape(res, res).astype(np.float32)


def rasterize(member: Member, damage: Damage, mate=None, res: int = RASTER_RES) -> np.ndarray:
    """(C,H,W) float array; channels = [member, must_replace, features, mate]."""
    feats = unary_union(damage.features) if damage.features else None
    mate_geom = mate.buffer(0.02) if mate is not None else None
    chans = [
        _mask(member.poly_shapely, res, FRAME),
        _mask(damage.must_replace, res, FRAME),
        _mask(feats, res, FRAME),
        _mask(mate_geom, res, FRAME),
    ]
    return np.stack(chans, axis=0)


def features_vector(member: Member, damage: Damage) -> np.ndarray:
    """Hand features (used for the PCA scatter in script 07)."""
    x0, y0, x1, y1 = member.bbox
    W, H = x1 - x0, y1 - y0
    c = damage.centroid
    dminx, dminy, dmaxx, dmaxy = damage.must_replace.bounds
    return np.array([
        damage.must_replace.area / member.poly_shapely.area,
        (c[0] - x0) / W, (c[1] - y0) / H,
        (dmaxx - dminx) / W, (dmaxy - dminy) / H,
        member.poly_shapely.area, len(damage.features), H / W,
    ], dtype=np.float32)
