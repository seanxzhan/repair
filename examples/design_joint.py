"""Design a joint the way a network would: start from a blank timber column and add
*cuts* (the predicted output); the mating pieces are computed from the cuts.

Each cut is one MXG millable extrusion in the exact parametrization a model would emit:
    normal   a class in design.CANONICAL_NORMALS (the in-plane frame is derived, not predicted)
    offset   plane position along the normal, from the stock centre
    uc,vc    2D profile-rectangle centre, in the plane's derived (u,v) frame
    uw,vh    2D profile-rectangle size
    depth    extrusion length along +normal  (large => a through cut)
    piece    which piece this carved region becomes (>=1; piece 0 is the kept base)

The pieces are a PARTITION of the column, so they always mate (overlap stays 0). The panel
prints each cut as the "prediction vector" so you can see exactly what the net must output.

    python examples/design_joint.py
"""
from __future__ import annotations

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from repair import design as dz
from repair.config import ensure_out
from repair.jwood import mating_overlap

PIECE_COLORS = [(0.78, 0.76, 0.72), (0.92, 0.58, 0.18), (0.16, 0.42, 0.82),
                (0.36, 0.66, 0.38), (0.82, 0.20, 0.18)]
NORMAL_KEYS = list(dz.CANONICAL_NORMALS)


def normal_index(normal) -> int:
    for i, k in enumerate(NORMAL_KEYS):
        if np.allclose(dz.CANONICAL_NORMALS[k], normal, atol=1e-3):
            return i
    return 0


class Designer:
    def __init__(self):
        self.stock = dz.Stock(length=3.0, width=1.0, height=1.0)
        # Start with one cut -> a half-lap, so two mating pieces show immediately.
        self.cuts = [dz.Cut(normal=(0, 0, 1), offset=0.0, vc=-0.75,
                            uw=2.0, vh=1.6, depth=1.0, piece=1)]
        self.regen()

    def regen(self):
        try:
            self.pieces = dz.pieces_from_cuts(self.stock, self.cuts)
            self.overlap = mating_overlap([p for p in self.pieces if len(p.vertices)])
        except Exception as e:
            print(f"  [design] failed: {type(e).__name__}: {e}")
            return
        ps.remove_all_structures()
        for i, m in enumerate(self.pieces):
            if not len(m.vertices):
                continue
            reg = ps.register_surface_mesh(f"piece{i}", np.asarray(m.vertices),
                                           np.asarray(m.faces), smooth_shade=False)
            reg.set_color(PIECE_COLORS[i % len(PIECE_COLORS)])

    # ------------------------------------------------------------------ gui
    def gui(self):
        dirty = False
        psim.TextUnformatted("Blank stock (timber column)")
        for fld, lo, hi in [("length", 1.0, 6.0), ("width", 0.3, 2.0),
                            ("height", 0.3, 2.0)]:
            c, val = psim.SliderFloat(fld, getattr(self.stock, fld), lo, hi)
            if c:
                setattr(self.stock, fld, val); dirty = True

        if psim.Button("+ add cut"):
            self.cuts.append(dz.Cut()); dirty = True
        psim.SameLine()
        if psim.Button("export STL"):
            self.export()
        ov = getattr(self, "overlap", 0.0)
        vols = [round(p.volume, 3) for p in self.pieces]
        psim.TextUnformatted(f"pieces={len(self.pieces)} vols={vols}")
        psim.TextUnformatted(f"mating: {'OK (overlap 0)' if ov < 1e-3 else f'overlap={ov:.3f}'}")
        psim.Separator()

        remove = None
        for i, cut in enumerate(self.cuts):
            if not psim.TreeNode(f"cut {i}  ->  piece {cut.piece}"):
                continue
            ni = normal_index(cut.normal)
            c, ni = psim.Combo(f"normal##{i}", ni, NORMAL_KEYS)
            if c:
                cut.normal = dz.CANONICAL_NORMALS[NORMAL_KEYS[ni]]; dirty = True
            for fld, lo, hi in [("offset", -1.5, 1.5), ("uc", -1.5, 1.5),
                                ("vc", -1.5, 1.5), ("uw", 0.05, 3.0),
                                ("vh", 0.05, 3.0), ("depth", 0.05, 6.0)]:
                c, val = psim.SliderFloat(f"{fld}##{i}", getattr(cut, fld), lo, hi)
                if c:
                    setattr(cut, fld, val); dirty = True
            c, pj = psim.Combo(f"piece##{i}", cut.piece - 1, ["1", "2", "3", "4"])
            if c:
                cut.piece = pj + 1; dirty = True
            # The prediction vector a network would emit for this cut:
            psim.TextUnformatted(
                f"net output: normal=#{ni}({NORMAL_KEYS[ni]}) offset={cut.offset:+.2f} "
                f"rect=({cut.uc:+.2f},{cut.vc:+.2f},{cut.uw:.2f},{cut.vh:.2f}) "
                f"depth={cut.depth:.2f} piece={cut.piece}")
            if psim.Button(f"remove##{i}"):
                remove = i
            psim.TreePop()
        if remove is not None:
            self.cuts.pop(remove); dirty = True

        if dirty:
            self.regen()

    def export(self):
        out = ensure_out()
        for i, m in enumerate(self.pieces):
            if len(m.vertices):
                m.export(out / f"design_piece{i}.stl")
        print(f"  -> wrote pieces to {out}/design_piece*.stl")


def main():
    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_up_dir("z_up")
    ps.set_navigation_style("turntable")
    designer = Designer()
    ps.set_user_callback(designer.gui)
    ps.show()


if __name__ == "__main__":
    main()
