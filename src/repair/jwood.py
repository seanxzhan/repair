"""Evaluate MiGumi JWood (MXG) parameters into 3D solids -- and re-evaluate under
edited parameters to synthesize new joint variations.

Each part of a joint is, per the MiGumi paper (Ganeshan et al., SIGGRAPH Asia 2025),
a material stock minus a set of millable extrusions:  ``P = M - U_i E_i``.  In the
``vis_files/<variant>_jwood.json`` files this is stored as a CSG ``expression`` over
**Linked Height Fields (LHFs)**.  Each LHF is a 2D ``polyset`` (a polygon in local
coords, columns [0,1]) extruded ``amount`` along ``plane_normal`` from ``plane_origin``.

The one piece not stored is the in-plane basis (u, v) of each plane -- it is derived
from the normal alone.  We reproduce the upstream convention exactly (verified against
``migumi/torch_compute/polyline_utils.py:get_frame_from_normal_np`` and validated by
reconstructing all 65 base parts to <2% symmetric-difference vs. the shipped STLs)::

    arbitrary = [0,0,1] if |n_z| < 0.999 else [1,0,0]
    u = normalize(arbitrary x n);  v = normalize(n x u);  world = origin + x*u + y*v + z*n

Edit an LHF (amount, 2D scale, position along the normal, on/off) via :class:`LHFEdit`
and pass a map to :func:`evaluate` to get a freshly re-CSG'd mesh for a new variation.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from shapely.affinity import scale as shp_scale
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .config import DATASET_ROOT

_BOOL = "manifold"          # trimesh boolean backend (pip install manifold3d)


# --------------------------------------------------------------- plane frame

def get_frame_from_normal(normal) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """In-plane basis (u, v, n) for a plane normal -- exact upstream convention."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    arbitrary = np.array([0., 0., 1.]) if abs(n[2]) < 0.999 else np.array([1., 0., 0.])
    u = np.cross(arbitrary, n); u /= np.linalg.norm(u)
    v = np.cross(n, u);         v /= np.linalg.norm(v)
    return u, v, n


# ------------------------------------------------------------------- editing

@dataclass
class LHFEdit:
    """A parameter override for one Linked Height Field. Defaults are the identity."""
    amount_mul: float = 1.0      # scale the extrusion depth
    poly_scale: float = 1.0      # uniform 2D scale of the profile about its centroid
    normal_shift: float = 0.0    # slide the plane along its normal (cut depth/position)
    enabled: bool = True         # drop the LHF entirely when False


# ---------------------------------------------------------------- mesh utils

def _clean(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Weld duplicate/coincident vertices so the mesh is watertight for booleans
    (pymeshlab; the shipped STLs and a few extrusions are otherwise non-manifold)."""
    import pymeshlab as ml
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(vertex_matrix=np.asarray(m.vertices, np.float64),
                        face_matrix=np.asarray(m.faces, np.int32)))
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_merge_close_vertices()
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_duplicate_faces()
    cm = ms.current_mesh()
    return trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)


def _lhf_region(lhf: dict, edit: LHFEdit) -> Polygon:
    """The LHF's 2D profile (polysets combined by sign), with the edit's scale applied."""
    pos, neg = [], []
    for ps, sgn in zip(lhf["polysets"], lhf["poly_signs"]):
        p = Polygon(np.asarray(ps, float)[:, :2])
        if not p.is_valid:
            p = p.buffer(0)
        (pos if sgn > 0 else neg).append(p)
    g = unary_union(pos)
    if neg:
        g = g.difference(unary_union(neg))
    if edit.poly_scale != 1.0:
        g = shp_scale(g, edit.poly_scale, edit.poly_scale, origin="centroid")
    return g


def _lhf_solid(lhf: dict, edit: LHFEdit) -> trimesh.Trimesh:
    g = _lhf_region(lhf, edit)
    amount = float(lhf["amount"][0]) * edit.amount_mul
    u, v, n = get_frame_from_normal(lhf["plane_normal"])
    origin = np.asarray(lhf["plane_origin"], float) + edit.normal_shift * n
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = u, v, n, origin
    parts = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
    meshes = []
    for p in parts:
        if p.is_empty or p.area <= 0:
            continue
        m = trimesh.creation.extrude_polygon(p, height=amount)
        m.apply_transform(T)
        meshes.append(m)
    if not meshes:
        raise ValueError("empty LHF region")
    return _clean(trimesh.util.concatenate(meshes))


# ------------------------------------------------------------------ evaluate

def evaluate(part: dict, edits: dict[str, LHFEdit] | None = None) -> trimesh.Trimesh:
    """CSG-evaluate one part's ``expression`` over its (optionally edited) LHFs."""
    edits = edits or {}
    solids = {}
    for name, lhf in part["lhfs"].items():
        edit = edits.get(name, LHFEdit())
        if edit.enabled:
            solids[name] = _lhf_solid(lhf, edit)

    def union(*ms):
        ms = [m for m in ms if m is not None]
        return functools.reduce(
            lambda a, b: trimesh.boolean.union([a, b], engine=_BOOL), ms)

    def difference(a, b):
        if b is None:
            return a
        return trimesh.boolean.difference([a, b], engine=_BOOL)

    # Disabled LHFs evaluate to None so Union/Difference can skip them.
    ns = {"Union": union, "Difference": difference,
          **{k: solids.get(k) for k in part["lhfs"]}}
    return _clean(eval(part["expression"], ns))


# ----------------------------------------------------------------------- io

@dataclass
class Joint:
    key: str
    variant: str
    parts: list[dict]            # raw jwood part dicts (name, expression, lhfs)
    state_map: dict              # part name -> {state: 4x4 assembly transform}

    def evaluate(self, edits: dict[int, dict[str, LHFEdit]] | None = None
                 ) -> list[trimesh.Trimesh]:
        edits = edits or {}
        return [evaluate(p, edits.get(i)) for i, p in enumerate(self.parts)]

    def assemble(self, meshes: list[trimesh.Trimesh], state: str = "1"
                 ) -> list[trimesh.Trimesh]:
        """Place evaluated parts into their assembled pose (state '1') via state_map."""
        out = []
        for part, m in zip(self.parts, meshes):
            T = np.array(self.state_map[part["name"]][state], float)
            out.append(m.copy().apply_transform(T))
        return out


def load(key: str, variant: str = "base", root: Path = DATASET_ROOT) -> Joint:
    jw = json.loads((root / key / "vis_files" / f"{variant}_jwood.json").read_text())
    return Joint(key=key, variant=variant, parts=jw["parts"],
                 state_map=jw.get("state_map", {}))


def mating_overlap(assembled: list[trimesh.Trimesh]) -> float:
    """Total volume where assembled parts interpenetrate -- a hard mating violation.

    A correctly coupled joint has ~0 (surfaces touch but solids don't overlap). This is
    a necessary, not sufficient, mating check: it catches collisions, not gaps. Mating
    is a *cross-part* property, so editing parts independently generally breaks it."""
    total = 0.0
    for i in range(len(assembled)):
        for k in range(i + 1, len(assembled)):
            inter = trimesh.boolean.intersection(
                [assembled[i], assembled[k]], engine=_BOOL)
            if inter is not None and len(inter.vertices):
                total += inter.volume
    return total
