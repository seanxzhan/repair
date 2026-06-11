"""Inspect the classify-then-regress representation: ONE set of cut params -> mating pieces.

    pick a joint  ->  its TYPE (reverse.signature) is shown
                  ->  ONE set of cut-plane params is reverse-engineered (the primary's cuts)
                  ->  you edit that single set of params
                  ->  the params cut the block into perfectly-mating pieces

Each cut is drawn in the 3D view as a coloured outline + a normal arrow, named like its
GUI control (lhf_1, lhf_2, ...). Open a cut's controls and its plane is highlighted, so
you can see which plane each parameter moves. Hover any slider for what it does.

    python examples/edit_joint.py            # opens on CJ_DT
    python examples/edit_joint.py CJ_SAT
"""
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from repair import reverse
from repair.config import DATASET_ROOT, ensure_out
from repair.design import CANONICAL_NORMALS
from repair.jwood import get_frame_from_normal, mating_overlap

PIECE_COLORS = [(0.85, 0.60, 0.32), (0.32, 0.55, 0.85), (0.48, 0.74, 0.42),
                (0.82, 0.42, 0.42), (0.62, 0.45, 0.80)]
CUT_COLORS = [(0.95, 0.85, 0.20), (0.20, 0.85, 0.85), (0.95, 0.30, 0.65),
              (0.55, 0.95, 0.30), (0.60, 0.40, 0.80), (0.20, 0.65, 0.65)]
NORMAL_KEYS = list(CANONICAL_NORMALS)
SEPARATION = 0.7           # apart copies spread along Y by this fraction of span
X_GAP = 0.5                # extra gap (fraction of span) between the apart set and the mains

# Plain-language help for each parameter (shown on hover).
TIPS = dict(
    normal="The direction the cut plane FACES. Changing it re-orients the whole cut.",
    offset="Slides the cut plane back/forth ALONG its normal -- how deep into the block "
           "the cut sits (follow the arrow).",
    depth="Multiplies how far the cut extrudes along its normal -- the THICKNESS of "
          "material it removes.",
    profile="Scales the 2D cut SHAPE (the outline) bigger/smaller, in place.",
    slide="Slides the cut SIDEWAYS within its own plane (perpendicular to the normal); "
          "depth is unchanged.",
)


def list_joints() -> list[str]:
    return sorted(p.name for p in DATASET_ROOT.iterdir() if (p / "info.json").exists())


@dataclass
class Adjust:
    d_offset: float = 0.0
    d_slide_u: float = 0.0
    d_slide_v: float = 0.0
    amount_mul: float = 1.0
    profile_scale: float = 1.0
    normal_key: str = ""


class Editor:
    def __init__(self, key: str):
        self.joints = list_joints()
        self.primary = 0
        self.show_planes = True
        self.show_tips = True
        self.set_joint(key if key in self.joints else self.joints[0])

    def _tip(self, key: str):
        if self.show_tips and psim.IsItemHovered():
            psim.SetTooltip(TIPS[key])

    def set_joint(self, key: str):
        self.key = key
        self.primary = 0
        self.nominal = reverse.load(key)
        self.adj = {cp.name: Adjust() for cp in self.nominal.parts[self.primary]}
        self.gizmos = {}            # lhf name -> (outline, arrow) polyscope handles
        self.cut_color = {}
        self.regen()

    def edited(self) -> reverse.JointParams:
        jp = copy.deepcopy(self.nominal)
        for cp in jp.parts[self.primary]:
            a = self.adj[cp.name]
            cp.offset += a.d_offset
            cp.in_plane = cp.in_plane + np.array([a.d_slide_u, a.d_slide_v])
            cp.amount *= a.amount_mul
            if a.normal_key:
                cp.normal = np.array(CANONICAL_NORMALS[a.normal_key], float)
            if a.profile_scale != 1.0:
                cp.scale_profile(a.profile_scale)
        return jp

    def regen(self):
        try:
            jp = self.edited()
            self.pieces = jp.partition(self.primary)
            self.overlap = mating_overlap([p for p in self.pieces if len(p.vertices)])
        except Exception as e:
            print(f"  [edit] rebuild failed: {type(e).__name__}: {e}")
            return
        valid_idx = [i for i, p in enumerate(self.pieces) if len(p.vertices)]
        allpts = np.vstack([self.pieces[i].vertices for i in valid_idx])
        span = float(np.ptp(allpts, axis=0).max())
        x_off = -(float(np.ptp(allpts[:, 0])) + X_GAP * span)   # apart set sits to -X
        gap = SEPARATION * span
        K = len(valid_idx)
        ps.remove_all_structures()
        self.gizmos = {}
        for rank, i in enumerate(valid_idx):
            m = self.pieces[i]
            col = PIECE_COLORS[i % len(PIECE_COLORS)]
            # in-place, transparent -- shows how the pieces mate (with the cut planes)
            asm = ps.register_surface_mesh(f"piece{i}", np.asarray(m.vertices),
                                           np.asarray(m.faces), smooth_shade=False)
            asm.set_color(col); asm.set_transparency(0.35)
            # opaque copy: offset to -X of the mains, pulled apart along Y by piece order
            shift = np.array([x_off, (rank - (K - 1) / 2) * gap, 0.0])
            sep = ps.register_surface_mesh(f"piece{i}_apart",
                                           np.asarray(m.vertices, float) + shift,
                                           np.asarray(m.faces), smooth_shade=False)
            sep.set_color(col)
        if self.show_planes:
            self._draw_planes(jp)
        self._draw_axes()

    def _draw_axes(self):
        """XYZ gnomon (X=red, Y=green, Z=blue) at the scene corner for orientation."""
        verts = [p.vertices for p in self.pieces if len(p.vertices)]
        if not verts:
            return
        pts = np.vstack(verts)
        lo, span = pts.min(0), float((pts.max(0) - pts.min(0)).max())
        origin = lo - 0.15 * span
        for ax, (col, nm) in enumerate([((0.9, 0.2, 0.2), "X"), ((0.3, 0.75, 0.3), "Y"),
                                        ((0.2, 0.45, 0.95), "Z")]):
            end = origin.copy(); end[ax] += 0.4 * span
            cn = ps.register_curve_network(f"axis_{nm}", np.array([origin, end]),
                                           np.array([[0, 1]]))
            cn.set_color(col); cn.set_radius(0.006)

    def _draw_planes(self, jp: reverse.JointParams):
        """Draw each cut as a labelled outline (its profile) + a normal arrow, in place
        on the fixed (transparent) assembled pieces they bound."""
        ci = 0
        for cp in jp.parts[self.primary]:
            if cp.is_stock:
                continue
            color = CUT_COLORS[ci % len(CUT_COLORS)]
            self.cut_color[cp.name] = color
            ci += 1
            u, v, n = get_frame_from_normal(cp.normal)
            origin = cp.offset * n + cp.in_plane[0] * u + cp.in_plane[1] * v
            poly = cp.profile
            nodes = origin + poly[:, :1] * u + poly[:, 1:2] * v
            edges = np.array([[k, (k + 1) % len(nodes)] for k in range(len(nodes))])
            out = ps.register_curve_network(f"{cp.name}", nodes, edges)
            out.set_color(color); out.set_radius(0.006)
            arr = ps.register_curve_network(f"{cp.name}_normal",
                                            np.array([origin, origin + 0.4 * n]),
                                            np.array([[0, 1]]))
            arr.set_color(color); arr.set_radius(0.010)
            self.gizmos[cp.name] = (out, arr)

    def _highlight(self, active: str | None):
        for name, (out, arr) in self.gizmos.items():
            hot = name == active
            out.set_radius(0.014 if hot else 0.006)
            arr.set_radius(0.020 if hot else 0.010)

    # ------------------------------------------------------------------ gui
    def gui(self):
        changed, idx = psim.Combo("joint", self.joints.index(self.key), self.joints)
        if changed:
            self.set_joint(self.joints[idx]); return

        nparts, partsigs = self.nominal.signature
        topo = "; ".join(f"{t}|{list(c)}" for t, c in partsigs[:1])
        psim.TextUnformatted(f"TYPE: {nparts}p  {topo}")

        c0, self.show_planes = psim.Checkbox("show cut planes", self.show_planes)
        psim.SameLine()
        _, self.show_tips = psim.Checkbox("hover help", self.show_tips)
        dirty = c0
        if len(self.nominal.parts) > 1:
            cp, pr = psim.Combo("primary (defines cuts)", self.primary,
                                [str(i) for i in range(len(self.nominal.parts))])
            if cp:
                self.primary = pr
                self.adj = {x.name: Adjust() for x in self.nominal.parts[self.primary]}
                self.regen(); return
        if psim.Button("reset edits"):
            self.set_joint(self.key); return
        psim.SameLine()
        if psim.Button("export STL"):
            self.export()
        ov = getattr(self, "overlap", 0.0)
        psim.TextUnformatted(f"pieces={len(self.pieces)}  "
                             f"mating: {'OK' if ov < 1e-3 else f'overlap={ov:.3f}'}")
        psim.Separator()
        psim.TextUnformatted("one cut-param set -> cuts the block into mating pieces.")
        psim.TextDisabled("(hover a slider for help; open a cut to highlight its plane)")

        active = None
        for cp in self.nominal.parts[self.primary]:
            a = self.adj[cp.name]
            if cp.is_stock:
                label = f"{cp.name} (STOCK = the block)"
            else:
                label = f"{cp.name} (cut [{reverse.normal_class(cp.normal)}])"
            if not psim.TreeNode(label):
                continue
            active = cp.name
            if not cp.is_stock:
                cur = a.normal_key or reverse.normal_class(cp.normal)
                cc, ni = psim.Combo(f"normal##{cp.name}", NORMAL_KEYS.index(cur), NORMAL_KEYS)
                self._tip("normal")
                if cc:
                    a.normal_key = NORMAL_KEYS[ni]; dirty = True
            c1, a.d_offset = psim.SliderFloat(f"offset {cp.offset:+.2f}##{cp.name}",
                                              a.d_offset, -1.5, 1.5); self._tip("offset")
            c2, a.amount_mul = psim.SliderFloat(f"depth x##{cp.name}",
                                                a.amount_mul, 0.2, 2.0); self._tip("depth")
            c3, a.profile_scale = psim.SliderFloat(f"profile size x##{cp.name}",
                                                   a.profile_scale, 0.3, 2.0); self._tip("profile")
            c4, a.d_slide_u = psim.SliderFloat(f"slide u##{cp.name}",
                                               a.d_slide_u, -1.0, 1.0); self._tip("slide")
            c5, a.d_slide_v = psim.SliderFloat(f"slide v##{cp.name}",
                                               a.d_slide_v, -1.0, 1.0); self._tip("slide")
            dirty = dirty or c1 or c2 or c3 or c4 or c5
            psim.TreePop()

        if dirty:
            self.regen()
        elif self.show_planes:
            self._highlight(active)        # cheap: just thickens the open cut's plane

    def export(self):
        out = ensure_out()
        for i, m in enumerate(self.pieces):
            if len(m.vertices):
                m.export(out / f"edit_{self.key}_{i}.stl")
        print(f"  -> wrote {len(self.pieces)} pieces to {out}/edit_{self.key}_*.stl")


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "CJ_DT"
    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_up_dir("z_up")
    ps.set_navigation_style("turntable")
    ps.set_transparency_mode("pretty")
    editor = Editor(key)
    ps.set_user_callback(editor.gui)
    ps.show()


if __name__ == "__main__":
    main()
