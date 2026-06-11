"""Reverse-engineer a dataset joint into (type signature, editable parameters).

This is the representation behind a classify-then-regress predictor for *editing existing
joints*:

  * :func:`signature` -> the joint **type** (classification target): part count + each
    part's CSG topology + the canonical-direction class of each cut. Verified on the data:
    30 joints collapse to 20 signatures, and the CJ_SAT scarf family shares one signature.

  * :func:`extract` -> the **parameters** (regression / edit target): per cut a plane
    ``offset`` (along the normal), in-plane slide, extrusion ``amount`` and 2D ``profile``,
    plus per-part stock size. A network would predict adjustments to these after picking
    the type.

:meth:`JointParams.rebuild` round-trips back through the verified ``jwood`` evaluator, so
editing is just: mutate the parameters and rebuild. Validated to reproduce the shipped
parts exactly (it repackages the same numbers the evaluator already trusts).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import trimesh

from . import jwood
from .design import CANONICAL_NORMALS

_KEYS = list(CANONICAL_NORMALS)
_VECS = np.array([CANONICAL_NORMALS[k] for k in _KEYS], float)


def normal_class(normal) -> str:
    """Nearest canonical direction -- the discrete label a classifier head predicts."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    return _KEYS[int(np.argmax(_VECS @ n))]


# --------------------------------------------------------------- type signature

def _part_signature(part: dict) -> tuple:
    topo = re.sub(r"lhf_\d+", "L", part["expression"])          # CSG shape, names abstracted
    stock = re.search(r"lhf_\d+", part["expression"]).group(0)
    cuts = tuple(sorted(normal_class(l["plane_normal"])
                        for k, l in part["lhfs"].items() if k != stock))
    return (topo, cuts)


def signature(joint: jwood.Joint) -> tuple:
    """A hashable joint **type**: (n_parts, sorted per-part (CSG topology, cut classes))."""
    return (len(joint.parts), tuple(sorted(_part_signature(p) for p in joint.parts)))


# ----------------------------------------------------------------- parameters

@dataclass
class CutParams:
    """Editable parameters of one Linked Height Field (stock or a milling cut)."""
    name: str
    is_stock: bool
    normal: np.ndarray            # plane normal (its class is part of the type)
    offset: float                 # signed plane position along the normal
    in_plane: np.ndarray          # (2,) origin slide within the plane (u, v)
    amount: float                 # extrusion depth
    polysets: list                # list of (Ni, 2) polygons in the derived (u, v) frame
    poly_signs: list

    @property
    def profile(self) -> np.ndarray:
        """The richest polyset -- the detailed cut profile (for display/descriptors)."""
        return max(self.polysets, key=len)

    def scale_profile(self, factor: float) -> None:
        """Uniformly scale every polyset about the main profile's centroid."""
        c = self.profile.mean(axis=0)
        self.polysets = [c + factor * (np.asarray(p) - c) for p in self.polysets]

    def to_lhf(self) -> dict:
        """Back to a jwood LHF dict (so the verified evaluator can rebuild it)."""
        u, v, n = jwood.get_frame_from_normal(self.normal)
        origin = self.offset * n + self.in_plane[0] * u + self.in_plane[1] * v
        polys = [np.c_[p, np.zeros(len(p))].tolist() for p in self.polysets]
        return {"plane_normal": n.tolist(), "plane_origin": origin.tolist(),
                "amount": [float(self.amount)], "polysets": polys,
                "poly_signs": list(self.poly_signs)}


@dataclass
class JointParams:
    key: str
    signature: tuple
    part_names: list[str]
    expressions: list[str]
    parts: list[list[CutParams]]          # per part: [stock, cut, cut, ...]
    state_map: dict

    def rebuild(self) -> list[trimesh.Trimesh]:
        meshes = []
        for cuts, expr in zip(self.parts, self.expressions):
            lhfs = {cp.name: cp.to_lhf() for cp in cuts}
            meshes.append(jwood.evaluate({"expression": expr, "lhfs": lhfs}))
        return meshes

    def rebuild_part(self, pi: int) -> trimesh.Trimesh:
        lhfs = {cp.name: cp.to_lhf() for cp in self.parts[pi]}
        return jwood.evaluate({"expression": self.expressions[pi], "lhfs": lhfs})

    def stock_solid(self, pi: int) -> trimesh.Trimesh:
        """The uncut block (the stock LHF) of part ``pi``."""
        cp = next(c for c in self.parts[pi] if c.is_stock)
        return jwood.evaluate({"expression": cp.name, "lhfs": {cp.name: cp.to_lhf()}})

    def partition(self, primary: int = 0) -> list[trimesh.Trimesh]:
        """Cut ONE block into mating pieces from a single set of cut params: the primary
        part's cuts carve the block (piece 0), and the rest of the block is the mating
        complement -- so the pieces never overlap, by construction. The complement is
        split into connected components so multi-piece joints still come apart."""
        from .jwood import _BOOL, _clean
        piece0 = self.rebuild_part(primary)
        block = self.stock_solid(primary)
        rest = _clean(trimesh.boolean.difference([block, piece0], engine=_BOOL))
        comps = [c for c in rest.split(only_watertight=False) if c.volume > 1e-4]
        return [piece0, *(comps or [rest])]

    def param_vector(self) -> dict:
        """Flat, fixed-size *scalar* parameters per cut (the regression target). The
        variable-length ``profile`` is returned separately by :meth:`profiles`."""
        out = {}
        for pi, cuts in enumerate(self.parts):
            for cp in cuts:
                tag = f"p{pi}.{cp.name}"
                out[f"{tag}.offset"] = round(cp.offset, 4)
                out[f"{tag}.amount"] = round(cp.amount, 4)
                out[f"{tag}.slide_u"] = round(float(cp.in_plane[0]), 4)
                out[f"{tag}.slide_v"] = round(float(cp.in_plane[1]), 4)
                if not cp.is_stock:
                    out[f"{tag}.normal_class"] = normal_class(cp.normal)
        return out


def extract(joint: jwood.Joint) -> JointParams:
    parts = []
    for part in joint.parts:
        stock = re.search(r"lhf_\d+", part["expression"]).group(0)
        cps = []
        for name, lhf in part["lhfs"].items():
            n = np.asarray(lhf["plane_normal"], float)
            n = n / np.linalg.norm(n)
            o = np.asarray(lhf["plane_origin"], float)
            u, v, _ = jwood.get_frame_from_normal(n)
            offset = float(o @ n)
            in_plane = np.array([o @ u, o @ v])
            polysets = [np.asarray(ps, float)[:, :2] for ps in lhf["polysets"]]
            cps.append(CutParams(
                name=name, is_stock=(name == stock), normal=n, offset=offset,
                in_plane=in_plane, amount=float(lhf["amount"][0]),
                polysets=polysets, poly_signs=list(lhf["poly_signs"])))
        parts.append(cps)
    return JointParams(key=joint.key, signature=signature(joint),
                       part_names=[p["name"] for p in joint.parts],
                       expressions=[p["expression"] for p in joint.parts],
                       parts=parts, state_map=joint.state_map)


def load(key: str) -> JointParams:
    return extract(jwood.load(key, "base"))
