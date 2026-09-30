"""A fixed-size, shared-layout parametrization of a two-part joint.

The premise, measured in this repo before it was written down:

  * The two members **partition** their stock overlap. `solid_A` and `solid_B`
    fill `stock_A & stock_B` to 0.985-1.000 with no interpenetration, so the
    joint is one dividing surface, not two independent cut sets.
  * Therefore **B is derived**: `solid_B = stock_B \\ dilate(solid_A, c)`, exact
    to 0.14% median (2.41% worst) across the 25 two-part joints. The residual is
    strictly one-directional -- `solid_B` is always *contained* in
    `stock_B \\ solid_A`, so B removes more than A fills, never less.
    It is NOT uniform clearance: sweeping `c` over [0, 0.008] improves exactly
    one joint (CJ_IT, 1.46% -> 0.80%) and the other 24 fit best at c=0. The
    extra removal is localized relief, so `c` is a real parameter to expose
    (zero clearance means coincident faces, which wrecks contact simulation)
    but not a model of the dataset's residual. CJ_KT's 2.41% is unexplained.

So only A is free, and A is already a stack of extrusions. This module writes
that stack at *fixed size*, in a frame normalized away from any particular
joint, so that one parameter vector means the same thing for every joint:

    Program = K slots x (active, normal, plane offset, depth, R rings x n_ctrl pts)

What "shared layout" means, and what it does not (earlier notes call this
"template-free", which is easy to misread as "joint types are gone"):

  * **Every joint still has its own cuts.** A dovetail and a lap are different
    LHF stacks in the dataset and different vectors here. The joint *type* is
    implicit in the values -- which slots are used, where their planes sit, what
    the rings look like. There is no type label and no per-type parameter list.
  * **Only the layout is shared.** Every part is padded to K slots and every
    ring resampled to n_ctrl points, so all joints give a vector of the same
    length whose positions mean the same thing. That makes this a common input
    format -- one surrogate can read any joint -- and nothing more.
  * **No named knobs.** Depth is explicit per slot, but a "width" or a "flare
    angle" is buried in ring control-point coordinates. A *template* is exactly
    what would supply those: hand-named parameters for one joint type, or
    offsets from one dataset joint. This module has neither -- each joint is
    stored absolutely, so the 25 dataset joints are 25 points in the space, not
    25 origins. Per-type parameters would be defined on top of this, per type.
  * **Not a design space to search freely.** Nothing here makes an arbitrary
    vector a valid joint. The fitted joints are known-good points; what lies
    between them is unmeasured, so picking a joint type remains a discrete
    choice made outside this module.

What the fixed size costs in fidelity is what `examples/fit_interface.py`
measures, by fitting every joint and sweeping K, R and n_ctrl.

Canonical frame: taken from A's stock LHF -- its plane frame gives the axes, its
origin the translation, and the longest stock extent the length scale. Geometry
is built and compared *in* that frame, so a joint's parameters are invariant to
where it happened to sit in the dataset's world coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from repair.authoring import canonicalize
from repair.jwood import LHF, lhf_to_mesh, orientation_frame

# What the dataset's two-part joints actually require (measured, not guessed).
MAX_SLOTS = 4
MAX_RINGS = 4
MAX_VERTS = 28


# --------------------------------------------------------------------------- #
# Two-part joint identification.
# --------------------------------------------------------------------------- #
def list_two_part_joints() -> list[str]:
    """Joints that actually reconstruct to two parts.

    `info.json` is not authoritative: RJM_SHS claims `n_parts: 2` and ships
    three, so the count is taken from the reconstruction instead.
    """
    from repair.config import DATASET_ROOT
    from repair.relational import reconstruct_parts

    out = []
    for p in sorted(DATASET_ROOT.iterdir()):
        if not (p / "info.json").exists():
            continue
        try:
            if len(reconstruct_parts(p.name, "base")) == 2:
                out.append(p.name)
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------- #
# Canonical frame.
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    """World -> canonical: rotate by `rot.T`, subtract `origin`, divide by `scale`."""

    rot: np.ndarray        # 3x3, columns are the canonical axes in world coords
    origin: np.ndarray     # (3,) world
    scale: float

    def point_to(self, p: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(p) - self.origin) @ self.rot / self.scale

    def vec_to(self, v: np.ndarray) -> np.ndarray:
        return np.atleast_2d(v) @ self.rot

    def matrix(self) -> np.ndarray:
        """4x4 world -> canonical, for transforming meshes."""
        T = np.eye(4)
        T[:3, :3] = self.rot.T / self.scale
        T[:3, 3] = -(self.rot.T @ self.origin) / self.scale
        return T


def canonical_frame(recon) -> Frame:
    """The frame implied by part A's stock: its plane axes, origin and extent."""
    stock = recon.spec.stock
    rot = orientation_frame(stock.normal, recon.stock_ori)
    scale = float(np.ptp(recon.stock_mesh.bounds, axis=0).max())
    return Frame(rot=rot, origin=np.asarray(stock.origin, float), scale=scale)


# --------------------------------------------------------------------------- #
# Ring resampling: variable vertex counts -> one fixed count.
# --------------------------------------------------------------------------- #
def resample_ring(ring: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polygon to exactly `n` points.

    Corner-preserving when it can be: if `n` is at least the polygon's vertex
    count, every original corner is kept and the spare points are distributed
    along the longest edges, so the shape is reproduced *exactly*. Only when
    `n` is smaller than the corner count does this approximate, by uniform
    arc-length sampling.

    The distinction matters for the sweep. Plain arc-length resampling cuts every
    corner a sample happens to straddle, which would charge a 28-gon a ~2% error
    even at 64 control points -- measuring the resampler, not the parametrization.
    """
    p = np.asarray(ring, float)[:, :2]
    if len(p) > 1 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    m = len(p)
    if m == 0:
        return np.zeros((n + 1, 2))
    loop = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    if seg.sum() <= 0:
        return np.vstack([np.repeat(p[:1], n, axis=0), p[:1]])

    if n >= m:
        # Keep every corner; spread the surplus over edges by length
        # (largest-remainder, so the counts sum exactly).
        extra = n - m
        share = seg / seg.sum() * extra
        k = np.floor(share).astype(int)
        for i in np.argsort(-(share - k))[:extra - k.sum()]:
            k[i] += 1
        pts = []
        for i in range(m):
            pts.append(p[i])
            for j in range(1, k[i] + 1):
                t = j / (k[i] + 1)
                pts.append(loop[i] * (1 - t) + loop[i + 1] * t)
        out = np.asarray(pts)
    else:
        d = np.concatenate([[0.0], np.cumsum(seg)])
        t = np.linspace(0.0, d[-1], n, endpoint=False)
        out = np.column_stack([np.interp(t, d, loop[:, i]) for i in (0, 1)])
    return np.vstack([out, out[:1]])


# --------------------------------------------------------------------------- #
# The program.
# --------------------------------------------------------------------------- #
@dataclass
class Slot:
    """One extrusion, in canonical coordinates. `signs[i] == 0` = unused ring."""

    normal: np.ndarray     # (3,) unit, canonical
    origin: np.ndarray     # (3,) canonical
    depth: float           # canonical
    rings: np.ndarray      # (R, n_ctrl + 1, 2) canonical sketch coords
    signs: np.ndarray      # (R,) +1 outer, -1 hole, 0 unused


@dataclass
class Program:
    key: str
    frame: Frame
    slots: list[Slot]
    clearance: float = 0.0

    @property
    def n_slots(self) -> int:
        return len(self.slots)

    def vector(self) -> np.ndarray:
        """Flatten to the fixed-length vector a surrogate or sampler would use."""
        parts = []
        for s in self.slots:
            parts += [s.normal, s.origin, [s.depth], s.signs, s.rings.ravel()]
        return np.concatenate([np.asarray(p, float).ravel() for p in parts])


def _reframe_sketch(lhf: LHF, frame: Frame) -> tuple[np.ndarray, np.ndarray, list]:
    """Express a (canonicalized, ori=0) LHF's plane and sketch in `frame`.

    Rotating the world changes which seed basis the sketch lives in, so the 2D
    coordinates need the same change of basis `authoring.canonicalize` applies
    for orientation -- here for an arbitrary rotation rather than a 90deg one.
    """
    n_c = (frame.rot.T @ lhf.normal)
    n_c /= np.linalg.norm(n_c)
    o_c = frame.point_to(lhf.origin)[0]
    F_old = frame.rot.T @ orientation_frame(lhf.normal, 0)[:, :2]   # in canonical coords
    F_new = orientation_frame(n_c, 0)[:, :2]
    M = F_new.T @ F_old                                             # 2x2, old -> new
    rings = [(np.asarray(r, float)[:, :2] @ M.T) / frame.scale for r in lhf.rings]
    return n_c, o_c, rings


def fit_program(recon, n_slots: int = MAX_SLOTS, n_rings: int = MAX_RINGS,
                n_ctrl: int = 16) -> Program:
    """Fit a fixed-size program to part A of a reconstruction.

    Cuts are taken largest-volume first (so truncating to `n_slots` drops the
    least material), canonicalized to ori=0, moved into the canonical frame, and
    every ring resampled to `n_ctrl` points.
    """
    frame = canonical_frame(recon)
    cuts = [canonicalize(c, o) for c, o in zip(recon.spec.cuts, recon.cut_oris)]
    order = np.argsort([-lhf_to_mesh(c, 0).volume for c in cuts])
    slots = []
    for i in order[:n_slots]:
        lhf = cuts[i]
        n_c, o_c, rings = _reframe_sketch(lhf, frame)
        keep = list(zip(rings, lhf.signs))[:n_rings]
        R = np.zeros((n_rings, n_ctrl + 1, 2))
        S = np.zeros(n_rings)
        for j, (r, s) in enumerate(keep):
            R[j] = resample_ring(r, n_ctrl)
            S[j] = s
        slots.append(Slot(normal=n_c, origin=o_c,
                          depth=float(abs(lhf.amount)) / frame.scale,
                          rings=R, signs=S))
    # Pad to exactly `n_slots`. A part with two cuts and one with four must give
    # the same vector length, or "fixed-size" is not true and nothing downstream
    # -- sampler, surrogate, optimizer -- can treat them as the same space.
    # An empty slot has every ring sign 0, which `slot_to_lhfs` skips.
    while len(slots) < n_slots:
        slots.append(Slot(normal=np.array([0.0, 0.0, 1.0]), origin=np.zeros(3),
                          depth=0.0, rings=np.zeros((n_rings, n_ctrl + 1, 2)),
                          signs=np.zeros(n_rings)))
    return Program(key=recon.name, frame=frame, slots=slots)


def _region(rings, signs, shrink: float = 0.0):
    """rings/signs -> a shapely region, eroded by `shrink`, free of collinear points.

    The simplify is not cosmetic. `resample_ring` puts `n_ctrl` points around the
    polygon, so an 8-gon resampled to 64 gets 8 collinear points per edge; those
    triangulate into degenerate faces, the extruded solid comes out non-watertight,
    and manifold then refuses every boolean against it. Dropping exactly-collinear
    vertices at build time costs nothing geometrically and keeps the parameter
    vector fixed-size, which is the whole point.
    """
    pos = [Polygon(r) for r, s in zip(rings, signs) if s > 0]
    g = unary_union(pos) if pos else Polygon()
    for r, s in zip(rings, signs):
        if s < 0:
            g = g.difference(Polygon(r))
    if shrink > 0:
        g = g.buffer(-shrink)
    if g.is_empty or g.area <= 1e-12:
        return None
    g = g.simplify(1e-10)
    return g if not g.is_empty else None


def slot_to_lhfs(slot: Slot, name: str, shrink: float = 0.0) -> list[LHF]:
    """A slot as LHFs in canonical space -- one per DISJOINT lobe.

    Splitting matters: a slot whose region is a MultiPolygon (RJ_KJAT's cut is
    two separate lobes) would otherwise extrude into one concatenated mesh with
    two components, which trimesh will not certify as a volume and manifold then
    refuses. One solid per lobe keeps every boolean well-defined.

    `shrink` erodes the region, which is how clearance is applied: eroding A's
    cuts by `c` grows A by `c`, so B is cut back from it by that much.
    """
    rings = [r for r, g in zip(slot.rings, slot.signs) if g != 0]
    signs = [int(g) for g in slot.signs if g != 0]
    if not rings or slot.depth - shrink <= 0:
        return []
    g = _region(rings, signs, shrink)
    if g is None:
        return []
    polys = list(g.geoms) if isinstance(g, MultiPolygon) else [g]
    out = []
    for j, poly in enumerate(polys):
        if poly.area <= 1e-12:
            continue
        r = [np.asarray(poly.exterior.coords)[:, :2]]
        sg = [1]
        for h in poly.interiors:
            r.append(np.asarray(h.coords)[:, :2])
            sg.append(-1)
        out.append(LHF(name=f"{name}_{j}", rings=r, signs=sg, origin=slot.origin,
                       normal=slot.normal, amount=max(slot.depth - shrink, 1e-6)))
    return out


def _cut_meshes(program: "Program", shrink: float = 0.0):
    return [lhf_to_mesh(l, 0)
            for i, s in enumerate(program.slots)
            for l in slot_to_lhfs(s, f"slot_{i}", shrink=shrink)]


def build_A(program: "Program", stock_A: trimesh.Trimesh) -> trimesh.Trimesh:
    """solid_A = stock_A - union(slots), in canonical space.

    Raises on a failed boolean rather than returning the uncut stock: a silent
    failure here reads as a 0% reconstruction error on a part that was never cut.
    """
    solid = stock_A
    for m in _cut_meshes(program, 0.0):
        solid = solid.difference(m)
    return solid


def derive_B(program: Program, stock_A: trimesh.Trimesh,
             stock_B: trimesh.Trimesh, clearance: float = 0.0) -> trimesh.Trimesh:
    """solid_B = stock_B \\ dilate(solid_A, c), with the dilation done in the
    cut parametrization: eroding each cut by `c` grows A by `c`."""
    grown = stock_A
    for m in _cut_meshes(program, clearance):
        grown = grown.difference(m)
    return stock_B.difference(grown)
