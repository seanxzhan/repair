"""Geometry visualization with polyscope (primary), matplotlib as a headless fallback.

Profiles are triangulated (mapbox-earcut) and extruded to thin 3D timber prisms, which
plays to polyscope's strength and reads like real wood members. Every structure uses a
fixed color convention so the storyboard reads as one narrative:

    member = light gray   kept = green   insert = orange
    rot    = red          features = brown   mate = blue
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ensure_out
from .dataset import Member
from .damage import Damage

COLORS = dict(member=(0.78, 0.76, 0.72), kept=(0.36, 0.66, 0.38),
              insert=(0.92, 0.58, 0.18), rot=(0.82, 0.20, 0.18),
              feature=(0.45, 0.30, 0.16), mate=(0.16, 0.42, 0.82))

_PS = None          # polyscope module once initialized
_PS_OK = None       # tri-state: None=untried, True/False=availability


# ----------------------------------------------------------- triangulation

def _triangulate(poly):
    import mapbox_earcut as earcut
    ext = np.asarray(poly.exterior.coords)[:-1]
    rings, parts = [], [ext]
    for hole in poly.interiors:
        parts.append(np.asarray(hole.coords)[:-1])
    acc = 0
    for p in parts:
        acc += len(p)
        rings.append(acc)
    V = np.concatenate(parts).astype(np.float64)
    idx = earcut.triangulate_float64(V, np.array(rings, dtype=np.uint32))
    F = np.asarray(idx, dtype=np.int64).reshape(-1, 3)
    return V, F, len(ext)


def to_prism_mesh(geom, thickness: float = 0.12, z0: float = 0.0):
    """Extrude a (possibly multi-) polygon to a solid 3D prism mesh (verts, faces)."""
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    allV, allF, off = [], [], 0
    for p in polys:
        if p.is_empty or p.area <= 0:
            continue
        V2, F, m = _triangulate(p)
        n = len(V2)
        bottom = np.c_[V2, np.full(n, z0)]
        top = np.c_[V2, np.full(n, z0 + thickness)]
        Vp = np.vstack([bottom, top])
        sides = []
        for i in range(m):
            a, b = i, (i + 1) % m
            sides.append([a, b, b + n])
            sides.append([a, b + n, a + n])
        Fp = np.vstack([F[:, ::-1], F + n, np.array(sides, dtype=np.int64)])
        allV.append(Vp)
        allF.append(Fp + off)
        off += len(Vp)
    if not allV:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.vstack(allV), np.vstack(allF)


# --------------------------------------------------------------- polyscope

def _init_ps():
    global _PS, _PS_OK
    if _PS_OK is not None:
        return _PS_OK
    try:
        import polyscope as ps
        ps.init()
        ps.set_ground_plane_mode("none")
        ps.set_up_dir("z_up")
        ps.set_navigation_style("free")
        ps.set_transparency_mode("pretty")
        _PS, _PS_OK = ps, True
    except Exception as e:        # no GL context (headless) -> matplotlib fallback
        print(f"  [viz] polyscope unavailable ({type(e).__name__}); using matplotlib")
        _PS_OK = False
    return _PS_OK


def _camera():
    _PS.look_at((0.05, -1.05, 1.35), (0.0, 0.0, 0.05))


def _add(name, geom, color, thickness=0.12, z0=0.0, enabled=True, transparency=1.0):
    if geom is None or geom.is_empty:
        return
    V, F = to_prism_mesh(geom, thickness, z0)
    if len(V) == 0:
        return
    m = _PS.register_surface_mesh(name, V, F, smooth_shade=False)
    m.set_color(color)
    m.set_enabled(enabled)
    if transparency < 1.0:
        m.set_transparency(transparency)


def _add_mate(mate):
    if mate is None or mate.is_empty:
        return
    nodes = np.c_[np.asarray(mate.coords), np.full(len(mate.coords), 0.14)]
    edges = np.array([[i, i + 1] for i in range(len(nodes) - 1)])
    if len(edges) == 0:
        return
    cn = _PS.register_curve_network("mate", nodes, edges)
    cn.set_color(COLORS["mate"])
    cn.set_radius(0.012)


# ------------------------------------------------------------- public API

def render_member(member: Member, path, mate=None):
    if not _init_ps():
        return _mpl_member(member, path, mate)
    _PS.remove_all_structures()
    _add("member", member.poly_shapely, COLORS["member"])
    if mate is not None:
        _add_mate(mate)
    _camera()
    return _shot(path)


def render_repair(member: Member, damage: Damage, cut, mate, path, title=None,
                  explode: float = 0.0):
    """kept + insert + rot + features + mate in one scene.

    explode>0 lifts the insert out of its cavity (along +z, nudged in +x) so the joint
    interface shape -- dovetail flare, scarf slope, ledge step -- becomes visible."""
    if not _init_ps():
        return _mpl_repair(member, damage, cut, mate, path, title)
    from shapely import affinity
    _PS.remove_all_structures()
    _add("kept", cut.kept, COLORS["kept"], thickness=0.10)
    insert_geom = cut.insert
    iz = 0.0
    if explode > 0:                       # pull the insert out of the cavity
        insert_geom = affinity.translate(cut.insert, explode * 0.5, 0.0)
        iz = explode
    _add("insert", insert_geom, COLORS["insert"], thickness=0.17, z0=iz)  # stands proud
    _add("rot", damage.must_replace, COLORS["rot"], thickness=0.04, z0=0.20,
         transparency=0.5)
    for i, f in enumerate(damage.features):
        _add(f"feat{i}", f, COLORS["feature"], thickness=0.04, z0=0.24)
    _add_mate(mate)
    _camera()
    return _shot(path)


def render_damage(member: Member, damage: Damage, mate, path):
    if not _init_ps():
        return _mpl_repair(member, damage, None, mate, path, None)
    _PS.remove_all_structures()
    _add("member", member.poly_shapely, COLORS["member"])
    _add("rot", damage.must_replace, COLORS["rot"], thickness=0.06, z0=0.12)
    for i, f in enumerate(damage.features):
        _add(f"feat{i}", f, COLORS["feature"], thickness=0.05, z0=0.18)
    _add_mate(mate)
    _camera()
    return _shot(path)


def _shot(path):
    path = str(path)
    _PS.set_screenshot_extension(".png")
    _PS.screenshot(path, transparent_bg=False)
    return path


def show():
    """Open the interactive polyscope window (call after a render_* in a script)."""
    if _init_ps():
        _PS.show()


# --------------------------------------------------- matplotlib fallback

def _poly_patches(ax, geom, color, alpha=1.0, z=1):
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath
    if geom is None or geom.is_empty:
        return
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for p in polys:
        if p.is_empty:
            continue
        verts = list(p.exterior.coords)
        codes = [MPath.MOVETO] + [MPath.LINETO] * (len(verts) - 2) + [MPath.CLOSEPOLY]
        for hole in p.interiors:
            hv = list(hole.coords)
            verts += hv
            codes += [MPath.MOVETO] + [MPath.LINETO] * (len(hv) - 2) + [MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="k",
                               lw=0.6, alpha=alpha, zorder=z))


def _mpl_axes(path, title=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_aspect("equal"); ax.set_xlim(-0.75, 0.75); ax.set_ylim(-0.6, 0.6)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    return fig, ax


def _save(fig, path):
    import matplotlib.pyplot as plt
    ensure_out()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return str(path)


def _mpl_member(member, path, mate=None):
    fig, ax = _mpl_axes(path, member.key)
    _poly_patches(ax, member.poly_shapely, COLORS["member"])
    if mate is not None and not mate.is_empty:
        mx, my = np.asarray(mate.coords).T
        ax.plot(mx, my, color=COLORS["mate"], lw=4)
    return _save(fig, path)


def _mpl_repair(member, damage, cut, mate, path, title):
    fig, ax = _mpl_axes(path, title)
    if cut is not None and cut.feasible:
        _poly_patches(ax, cut.kept, COLORS["kept"], z=1)
        _poly_patches(ax, cut.insert, COLORS["insert"], z=2)
    else:
        _poly_patches(ax, member.poly_shapely, COLORS["member"], z=1)
    _poly_patches(ax, damage.must_replace, COLORS["rot"], alpha=0.75, z=3)
    for f in damage.features:
        _poly_patches(ax, f, COLORS["feature"], alpha=0.8, z=4)
    if mate is not None and not mate.is_empty:
        mx, my = np.asarray(mate.coords).T
        ax.plot(mx, my, color=COLORS["mate"], lw=4, zorder=5)
    return _save(fig, path)
