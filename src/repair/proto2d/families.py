"""2D repair-interface families, drawn from three MiGumi splices, each in two orientations.

model.py has one family (a mortise-and-tenon splice) with its contact faces
written out by hand. This module makes the family the variable: an interface is
whatever region of the block the new wood replaces, given as a polygon, and the
contact faces are read off that polygon's edges inside the block. The LP,
the damage field and the sound-wood integral are unchanged.

The families, and the MiGumi joint each is a cross-section of:

    butt           a plain cut (the baseline; carries no moment)
    tenon          mortise and tenon, the Figure 1 family of docs/proposal.md
    dovetail       CJ_AT  Ari Tsugi       tenon whose cheeks flare toward the tip at a
                                          dovetail angle of 6 to 15 degrees
    hooked_scarf   CJ_DT  Daimochi Tsugi  long shallow scarf with a 45-degree hook
                                          step mid-chord and equal shoulders at both ends

Orientation is a separate family. `*_flip` reflects the interface left-to-right
and swaps which side is retained: the same joint, cut the other way round. For
the tenon and dovetail the tongue then belongs to the retained wood and the
pocket to the new wood; for the scarf the retained wedge is above instead of
below. In sound wood a flipped family carries exactly the moment of its
original (reflection reverses the couple, swapping the free body reverses it
back); under a damage front that leans, the two differ, which is the point.

A family may span the beam's width in *layers* (a width fraction with its own
removed region; each layer's force budget is its fraction of the cap). Every
family here uses one layer; the machinery stays for joints that need it.

Coordinates as in model.py: x along the block, y up, the damage front entering
from the right. Every family removes everything right of its interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shapely
import shapely.affinity
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from .model import EPS, Block, Contact, Damage, Grid, Statics, _span, solve_moment

TOL = 1e-9


@dataclass(frozen=True)
class Param:
    name: str
    lo: float
    hi: float
    default: float
    doc: str = ""


@dataclass(frozen=True)
class Layer:
    """A fraction `weight` of the beam's width, and the region the new wood fills."""
    weight: float
    removed: Polygon
    label: str = ""


@dataclass
class LayerContact(Contact):
    layer: int = 0


@dataclass(frozen=True)
class Family:
    key: str
    title: str
    source: str                          # the MiGumi joint, or "" for none
    params: tuple[Param, ...]
    build: object = field(repr=False)    # (p: dict, b: Block) -> list[Layer]
    clip: object = field(repr=False, default=None)   # (p: dict, b: Block) -> dict, feasibility

    def defaults(self) -> dict:
        return {q.name: q.default for q in self.params}

    def feasible(self, p: dict, b: Block) -> dict:
        p = {q.name: float(np.clip(p[q.name], q.lo, q.hi)) for q in self.params}
        return self.clip(p, b) if self.clip else p

    def layers(self, p: dict, b: Block = Block()) -> list[Layer]:
        p = self.feasible(p, b)
        out = []
        for lay in self.build(p, b):
            reg = lay.removed.intersection(box(0.0, 0.0, b.length, b.height))
            out.append(Layer(lay.weight, reg, lay.label))
        return out

    def vector(self, p: dict) -> np.ndarray:
        return np.array([p[q.name] for q in self.params], float)

    def from_vector(self, v) -> dict:
        return {q.name: float(x) for q, x in zip(self.params, v)}


# ------------------------------------------------------------------ geometry
def _far(b: Block) -> float:
    """Where 'everything to the right' stops. Far enough past the block that a
    reflection about any axis inside the block still covers the block's left end."""
    return 4.0 * b.length


def _end(a: float, b: Block) -> Polygon:
    """Everything right of x = a."""
    return box(a, 0.0, _far(b), b.height)


def _flip(layers: list[Layer], axis: float, b: Block) -> list[Layer]:
    """Reflect each removed region about x = axis and swap the roles of the two
    sides: the new removed region is the block minus the reflection."""
    blk = box(0.0, 0.0, b.length, b.height)
    out = []
    for lay in layers:
        refl = shapely.affinity.scale(lay.removed, xfact=-1.0, yfact=1.0, origin=(axis, 0.0))
        out.append(Layer(lay.weight, blk.difference(refl), lay.label))
    return out


def _flipped(build, axis):
    """A builder whose output is `build`'s, flipped about x = axis(p)."""
    return lambda p, b: _flip(build(p, b), axis(p), b)


def _on_boundary(p, q, b: Block) -> bool:
    """Both endpoints on the same side of the block: not an interface edge."""
    for k, val in ((0, 0.0), (0, b.length), (1, 0.0), (1, b.height)):
        if abs(p[k] - val) < TOL and abs(q[k] - val) < TOL:
            return True
    return False


def interface_segments(layer: Layer, b: Block = Block()):
    """Edges of the removed region that lie inside the block, as
    (p0, p1, normal) with the normal pointing into the new wood."""
    segs = []
    geoms = getattr(layer.removed, "geoms", [layer.removed])
    for g in geoms:
        if g.is_empty or g.area < TOL:
            continue
        g = orient(g, 1.0)                            # CCW: interior on the left
        rings = [g.exterior] + list(g.interiors)
        for ring in rings:
            pts = np.asarray(ring.coords, float)
            for p, q in zip(pts[:-1], pts[1:]):
                d = q - p
                L = float(np.hypot(*d))
                if L < TOL or _on_boundary(p, q, b):
                    continue
                n = np.array([-d[1], d[0]]) / L       # left normal of a CCW edge
                segs.append((p, q, n))
    return segs


def contact_set(fam: Family, p: dict, d: Damage, b: Block = Block(), n_samp: int = 7) -> list[LayerContact]:
    """Every contact sample of every layer, with its live flag. Face indices
    run through the layers in order."""
    out, face = [], 0
    for li, lay in enumerate(fam.layers(p, b)):
        for p0, p1, n in interface_segments(lay, b):
            for pt in _span(p0, p1, n_samp):
                q = pt - EPS * n                      # probe just inside the retained wood
                out.append(LayerContact(face, pt, n, bool(d.sound(q[0], q[1], b)), li))
            face += 1
    return out


def budgets(contacts: list[LayerContact], layers: list[Layer], st: Statics):
    """One force budget per layer: w * cap over that layer's contacts."""
    return [([i for i, c in enumerate(contacts) if c.layer == li], lay.weight * st.cap)
            for li, lay in enumerate(layers)]


def removed_masks(fam: Family, p: dict, grid: Grid) -> list[tuple[float, np.ndarray]]:
    """(weight, boolean mask over grid cells) per layer."""
    out = []
    for lay in fam.layers(p, grid.block):
        m = shapely.contains_xy(lay.removed, grid.X.ravel(), grid.Y.ravel()).reshape(grid.X.shape)
        out.append((lay.weight, m))
    return out


def sound_removed(fam: Family, p: dict, d: Damage, grid: Grid) -> float:
    """Width-weighted area of sound wood the interface throws away."""
    sound = d.sound(grid.X, grid.Y, grid.block)
    return float(sum(w * (sound & m).sum() for w, m in removed_masks(fam, p, grid))) * grid.cell


def evaluate(fam: Family, p: dict, d: Damage, st: Statics, grid: Grid, n_samp: int = 7):
    """(M, R, contacts, forces): capacity, sound wood removed, the contact set
    and the LP's forces on it."""
    b = grid.block
    layers = fam.layers(p, b)
    cs = contact_set(fam, p, d, b, n_samp)
    M, F = solve_moment(cs, st, budgets(cs, layers, st))
    return M, sound_removed(fam, p, d, grid), cs, F


# ------------------------------------------------------------------ families
def _clip_tenon(p, b):
    p["ell"] = min(p["ell"], p["a"] - 0.2)
    half = min(p["yc"], b.height - p["yc"]) - 0.01
    p["t"] = min(p["t"], 2 * half)
    return p


def _tenon(p, b):
    a, ell, y1, y2 = p["a"], p["ell"], p["yc"] - p["t"] / 2, p["yc"] + p["t"] / 2
    pocket = box(a - ell, y1, a, y2) if ell > TOL else Polygon()
    return [Layer(1.0, unary_union([_end(a, b), pocket]), "full width")]


def _clip_tenon_flip(p, b):
    p["ell"] = min(p["ell"], b.length - p["a"] - 0.05)
    half = min(p["yc"], b.height - p["yc"]) - 0.01
    p["t"] = min(p["t"], 2 * half)
    return p


def _tip(p):
    """Tip thickness of a dovetail: neck plus the flare its angle gives over its length."""
    return p["t"] + 2.0 * p["ell"] * np.tan(np.radians(p["angle"]))


def _clip_dovetail(p, b):
    p["ell"] = min(p["ell"], p["a"] - 0.2)
    half = min(p["yc"], b.height - p["yc"]) - 0.01
    p["t"] = min(p["t"], 2 * half)
    # the tip must stay inside the block: shorten the tenon, keep the angle
    p["ell"] = max(0.0, min(p["ell"], (2 * half - p["t"]) / (2.0 * np.tan(np.radians(p["angle"])))))
    return p


def _clip_dovetail_flip(p, b):
    p["ell"] = min(p["ell"], b.length - p["a"] - 0.05)
    half = min(p["yc"], b.height - p["yc"]) - 0.01
    p["t"] = min(p["t"], 2 * half)
    p["ell"] = max(0.0, min(p["ell"], (2 * half - p["t"]) / (2.0 * np.tan(np.radians(p["angle"])))))
    return p


def _trapezoid(a, ell, yc, t_neck, t_tip):
    """Tongue rooted at x = a, tip at x = a - ell, widening from t_neck to t_tip."""
    return Polygon([(a, yc - t_neck / 2), (a - ell, yc - t_tip / 2),
                    (a - ell, yc + t_tip / 2), (a, yc + t_neck / 2)])


def _dovetail(p, b):
    a, ell = p["a"], p["ell"]
    tongue = _trapezoid(a, ell, p["yc"], p["t"], _tip(p)) if ell > TOL else Polygon()
    return [Layer(1.0, unary_union([_end(a, b), tongue]), "full width")]


def _clip_scarf(p, b):
    p["L"] = min(p["L"], b.length - p["a"])
    p["d"] = min(p["d"], 0.5 * b.height - 0.05)
    # the hook riser must stay below the top and the chord must keep descending
    p["s"] = max(0.0, min(p["s"], 0.45 * (b.height - 2 * p["d"]), 0.45 * p["L"]))
    return p


def _hooked_scarf(p, b):
    a, L, H = p["a"], p["L"], b.height
    top, bot = np.array([a, H - p["d"]]), np.array([a + L, p["d"]])
    h0 = 0.5 * (top + bot)                            # the hook sits mid-chord
    h1 = h0 + np.array([p["s"], p["s"]])              # a 45-degree riser, as in CJ_DT
    ring = [(a, H), tuple(top), tuple(h0), tuple(h1), tuple(bot), (a + L, 0.0),
            (_far(b), 0.0), (_far(b), H)]
    return [Layer(1.0, Polygon(ring).buffer(0), "full width")]


def _butt(p, b):
    return [Layer(1.0, _end(p["a"], b), "full width")]


P = Param
_TENON = (
    P("a", 6.0, 11.8, 9.48, "shoulder position"),
    P("ell", 0.0, 4.0, 0.76, "tenon length"),
    P("yc", 0.1, 1.9, 1.0, "cheek centre"),
    P("t", 0.05, 1.9, 0.7, "tenon thickness"))
_DOVETAIL = (
    P("a", 6.0, 11.8, 9.48, "shoulder position"),
    P("ell", 0.0, 4.0, 1.2, "tenon length"),
    P("yc", 0.1, 1.9, 1.0, "tenon centre"),
    P("t", 0.05, 1.9, 0.67, "thickness at the neck"),
    P("angle", 6.0, 15.0, 12.0, "dovetail angle in degrees (1:8 is 7, 1:6 is 9.5); never zero, so no dovetail is a tenon"))
_SCARF = (
    P("a", 6.0, 11.0, 8.0, "start of the scarf"),
    P("L", 0.5, 5.0, 2.0, "run of the scarf"),
    P("d", 0.0, 0.9, 0.3, "shoulder depth at both ends"),
    P("s", 0.0, 0.5, 0.25, "hook riser, mid-chord"))


def _with_defaults(params, **kw):
    return tuple(P(q.name, q.lo, q.hi, kw.get(q.name, q.default), q.doc) for q in params)


FAMILIES: dict[str, Family] = {f.key: f for f in [
    Family("butt", "plain cut", "", (
        P("a", 6.0, 11.8, 9.48, "cut position"),), _butt),
    Family("tenon", "mortise and tenon, tenon on the new wood", "", _TENON, _tenon, _clip_tenon),
    Family("tenon_flip", "mortise and tenon, tenon on the retained wood", "",
           _with_defaults(_TENON, a=8.6, ell=0.5), _flipped(_tenon, lambda p: p["a"]), _clip_tenon_flip),
    Family("dovetail", "dovetail, tongue on the new wood", "CJ_AT", _DOVETAIL, _dovetail, _clip_dovetail),
    Family("dovetail_flip", "dovetail, tongue on the retained wood", "CJ_AT",
           _with_defaults(_DOVETAIL, a=8.4, ell=0.8), _flipped(_dovetail, lambda p: p["a"]), _clip_dovetail_flip),
    Family("hooked_scarf", "hooked scarf, retained wedge below", "CJ_DT", _SCARF, _hooked_scarf, _clip_scarf),
    Family("hooked_scarf_flip", "hooked scarf, retained wedge above", "CJ_DT",
           _with_defaults(_SCARF, L=1.5, s=0.15), _flipped(_hooked_scarf, lambda p: p["a"] + p["L"] / 2), _clip_scarf),
]}
FLIP_OF = {"tenon": "tenon_flip", "dovetail": "dovetail_flip", "hooked_scarf": "hooked_scarf_flip"}


# ------------------------------------------------------------------ text views
def ascii_layers(fam: Family, p: dict, d: Damage, cs=None, b: Block = Block(),
                 nx: int = 96, ny: int = 16, x0: float = 6.0) -> str:
    """Each layer as a character grid over x in [x0, length]: '#' retained sound,
    'x' retained rotten, '.' new wood; contacts drawn as 'O' live and 'X' dead."""
    xs = x0 + (np.arange(nx) + 0.5) * (b.length - x0) / nx
    ys = (np.arange(ny) + 0.5) * b.height / ny
    X, Y = np.meshgrid(xs, ys)
    sound = d.sound(X, Y, b)
    blocks = []
    for li, lay in enumerate(fam.layers(p, b)):
        rem = shapely.contains_xy(lay.removed, X.ravel(), Y.ravel()).reshape(X.shape)
        img = np.where(rem, ".", np.where(sound, "#", "x")).astype("<U1")
        for c in (cs or []):
            if c.layer != li:
                continue
            j = int((c.point[0] - x0) / (b.length - x0) * nx); i = int(c.point[1] / b.height * ny)
            if 0 <= i < ny and 0 <= j < nx:
                img[i, j] = "O" if c.live else "X"
        rows = ["".join(r) for r in img[::-1]]
        blocks.append(f"layer {li}: {lay.label} (w = {lay.weight:.2f})\n" + "\n".join(rows))
    return "\n".join(blocks)


def summary(fam: Family, p: dict, d: Damage, st: Statics, grid: Grid, n_samp: int = 7) -> str:
    M, R, cs, F = evaluate(fam, p, d, st, grid, n_samp)
    p = fam.feasible(p, grid.block)
    live = sum(c.live for c in cs)
    lines = [f"{fam.key:13s} {fam.title}" + (f"  [{fam.source}]" if fam.source else ""),
             "  " + "  ".join(f"{q.name}={p[q.name]:.2f}" for q in fam.params),
             f"  M_max = {M:7.2f}   sound removed = {R:.3f}   {live}/{len(cs)} contacts live"]
    faces = sorted({(c.layer, c.face) for c in cs})
    per = []
    for li, fi in faces:
        cc = [c for c in cs if c.face == fi]
        n = cc[0].normal
        per.append(f"L{li} n=({n[0]:+.2f},{n[1]:+.2f}) {sum(c.live for c in cc)}/{len(cc)}")
    lines.append("  " + "   ".join(per))
    return "\n".join(lines)


# ------------------------------------------------------------------ checks
def sanity_checks(seed: int = 0):
    """The generic machinery must reproduce model.py exactly, and the families
    must nest the way the geometry says they do."""
    from dataclasses import replace
    from . import model as sp
    rng = np.random.default_rng(seed)
    grid = Grid()
    st = Statics()
    ten = FAMILIES["tenon"]
    # 1. tenon == model.Splice, at Figure 1 and at random interfaces / damage fronts
    trials = [(sp.Splice(), sp.Damage())]
    for _ in range(12):
        a = rng.uniform(7.0, 11.0); ell = rng.uniform(0.0, min(3.0, a - 0.3))
        yc = rng.uniform(0.4, 1.6); t = rng.uniform(0.1, 2 * min(yc, 2 - yc) - 0.05)
        s = sp.Splice(a=a, ell=ell).with_cheeks(yc, t)
        dmg = sp.Damage(x0=rng.uniform(7.5, 10.5), slope=rng.uniform(-1, 2), width=rng.uniform(0.05, 0.6))
        trials.append((s, dmg))
    for s, dmg in trials:
        p = dict(a=s.a, ell=s.ell, yc=0.5 * (s.y1 + s.y2), t=s.y2 - s.y1)
        M0 = sp.max_moment(sp.contact_set(s, dmg), st); R0 = grid.sound_removed(s, dmg)
        M1, R1, cs, _ = evaluate(ten, p, dmg, st, grid)
        n0 = len(sp.contact_set(s, dmg))
        assert abs(M0 - M1) < 1e-6 * max(1.0, M0), (M0, M1, p)
        assert abs(R0 - R1) < 1e-9, (R0, R1, p)
        assert s.ell < 1e-9 or len(cs) == n0, (len(cs), n0)   # ell = 0 merges the three collinear faces
    # 2. a plain cut carries nothing, whatever the damage
    for fam_key in ("butt",):
        for _ in range(5):
            dmg = sp.Damage(x0=rng.uniform(7.5, 10.5))
            M, *_ = evaluate(FAMILIES[fam_key], {"a": rng.uniform(7, 11)}, dmg, st, grid)
            assert M < 1e-6
    # 3. a dovetail's tongue has the area its angle says, and its tip stays in the block
    dv = FAMILIES["dovetail"]
    for _ in range(20):
        p = dv.feasible({q.name: rng.uniform(q.lo, q.hi) for q in dv.params}, grid.block)
        area = dv.layers(p, grid.block)[0].removed.area - (grid.block.length - p["a"]) * grid.block.height
        assert abs(area - p["ell"] * (p["t"] + _tip(p)) / 2) < 1e-9, (p, area)
        assert p["yc"] + _tip(p) / 2 <= grid.block.height + 1e-9 and p["yc"] - _tip(p) / 2 >= -1e-9
    # 4. sound wood everywhere: every contact live, and every family but the
    #    butt carries a positive moment at its defaults
    far = sp.Damage(x0=30.0)
    for fam in FAMILIES.values():
        M, R, cs, _ = evaluate(fam, fam.defaults(), far, st, grid)
        assert all(c.live for c in cs), fam.key
        assert (M > 1e-6) == (fam.key != "butt"), (fam.key, M)
    # 5. a flip is the same joint cut the other way round: in sound wood it
    #    carries the same moment, and the two removed areas sum to twice the
    #    block area right of the reflection axis
    for key, fkey in FLIP_OF.items():
        fam, flip = FAMILIES[key], FAMILIES[fkey]
        for _ in range(6):
            p = {q.name: rng.uniform(q.lo, q.hi) for q in fam.params}
            p["a"] = rng.uniform(7.0, 9.0)
            if "ell" in p:
                p["ell"] = rng.uniform(0.2, 1.5)
            if "L" in p:
                p["L"] = rng.uniform(1.0, 3.0)
            p = flip.feasible(fam.feasible(p, grid.block), grid.block)
            M0, *_ = evaluate(fam, p, far, st, grid)
            M1, *_ = evaluate(flip, p, far, st, grid)
            assert abs(M0 - M1) < 1e-6 * max(1.0, M0), (key, p, M0, M1)
            axis = p["a"] + (p["L"] / 2 if "L" in p else 0.0)
            A0 = sum(l.removed.area for l in fam.layers(p, grid.block))
            A1 = sum(l.removed.area for l in flip.layers(p, grid.block))
            assert abs(A0 + A1 - 2 * (grid.block.length - axis) * grid.block.height) < 1e-9, (key, A0, A1)
    return True
