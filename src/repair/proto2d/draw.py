"""Drawing one configuration (damage field + interface + contacts) to a PNG.

Shared by the dataset report and the surrogate evaluation. No matplotlib:
PIL only, so it runs anywhere the package does.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import shapely
from PIL import Image, ImageDraw

from . import families as fm
from .model import Block

WOOD, ROT, NEW = np.array([236, 220, 190]), np.array([120, 20, 20]), np.array([250, 248, 244])


def draw(fam: fm.Family, p: dict, f, cs=None, b: Block = Block(), scale: int = 60, x0: float = 4.0) -> Image.Image:
    """Retained wood coloured by severity, new wood with the damage it replaces
    faded, the crit contour and the interface in black, contacts green (live)
    or red (dead). Draws x in [x0, length]."""
    W, H = int((b.length - x0) * scale), int(b.height * scale)
    xs = x0 + (np.arange(W) + 0.5) / scale; ys = b.height - (np.arange(H) + 0.5) / scale
    X, Y = np.meshgrid(xs, ys)
    sev = np.clip(f.severity(X, Y), 0, 1)[..., None]
    rgb = (WOOD * (1 - sev) + ROT * sev).astype(np.uint8)
    dead = sev[..., 0] >= f.crit
    edge = np.zeros_like(dead); edge[:, 1:] |= dead[:, 1:] != dead[:, :-1]; edge[1:, :] |= dead[1:, :] != dead[:-1, :]
    layers = fam.layers(p, b)
    for lay in layers:
        removed = shapely.contains_xy(lay.removed, X.ravel(), Y.ravel()).reshape(X.shape)
        rgb[removed] = (0.35 * rgb[removed] + 0.65 * NEW).astype(np.uint8)
    rgb[edge] = (0, 0, 0)
    img = Image.fromarray(rgb); dr = ImageDraw.Draw(img)
    to_px = lambda x, y: ((x - x0) * scale, (b.height - y) * scale)
    for lay in layers:
        for p0, p1, _ in fm.interface_segments(lay, b):
            dr.line([to_px(*p0), to_px(*p1)], fill=(0, 0, 0), width=2)
    for c in (cs or []):
        x, y = to_px(*c.point); r = 3
        dr.ellipse([x - r, y - r, x + r, y + r], fill=(20, 160, 20) if c.live else (210, 50, 50))
    dr.rectangle([0, 0, W - 1, H - 1], outline=(0, 0, 0))
    return img


def draw_field(f, b: Block = Block(), scale: int = 40, x0: float = 0.0) -> Image.Image:
    """A damage field alone: severity from wood to rot, the crit contour in black."""
    W, H = int((b.length - x0) * scale), int(b.height * scale)
    xs = x0 + (np.arange(W) + 0.5) / scale; ys = b.height - (np.arange(H) + 0.5) / scale
    X, Y = np.meshgrid(xs, ys)
    sev = np.clip(f.severity(X, Y), 0, 1)[..., None]
    rgb = (WOOD * (1 - sev) + ROT * sev).astype(np.uint8)
    dead = sev[..., 0] >= f.crit
    edge = np.zeros_like(dead); edge[:, 1:] |= dead[:, 1:] != dead[:, :-1]; edge[1:, :] |= dead[1:, :] != dead[:-1, :]
    rgb[edge] = (0, 0, 0)
    img = Image.fromarray(rgb); ImageDraw.Draw(img).rectangle([0, 0, W - 1, H - 1], outline=(0, 0, 0))
    return img


def png_base64(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
