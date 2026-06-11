"""Interactively alter a joint's MXG parameters and re-CSG it into a new variation.

This drives the verified JWood evaluator (``repair.jwood``), which reconstructs every
shipped ``base`` part to <2% of the original STL, so edits here are faithful re-CSG --
not an approximation. Per Linked Height Field (the stock and each milled cut) you can:

    amount   scale the extrusion depth          scale   resize the 2D profile
    shift    slide the plane along its normal    on/off  drop the cut entirely

    python examples/edit_joint.py            # opens on CJ_DT
    python examples/edit_joint.py RJ_TGWA

Edits recompute live; "Export STL" writes the current variation to out/.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from repair import jwood
from repair.config import DATASET_ROOT, ensure_out

PART_COLORS = [(0.78, 0.76, 0.72), (0.92, 0.58, 0.18), (0.16, 0.42, 0.82),
               (0.36, 0.66, 0.38), (0.82, 0.20, 0.18)]


def list_joints() -> list[str]:
    return sorted(p.name for p in DATASET_ROOT.iterdir() if (p / "info.json").exists())


def stock_name(part: dict) -> str:
    return re.search(r"lhf_\d+", part["expression"]).group(0)   # first token = stock M


class Editor:
    def __init__(self, key: str):
        self.joints = list_joints()
        self.explode = 0.0
        self.set_joint(key if key in self.joints else self.joints[0])

    def set_joint(self, key: str):
        self.key = key
        self.joint = jwood.load(key, "base")
        # edits[part_idx][lhf_name] -> LHFEdit (identity to start)
        self.edits = {i: {name: jwood.LHFEdit() for name in p["lhfs"]}
                      for i, p in enumerate(self.joint.parts)}
        self.regen()

    def regen(self):
        try:
            self.meshes = self.joint.evaluate(self.edits)
        except Exception as e:                 # an edit can momentarily empty a part
            print(f"  [edit] eval failed: {type(e).__name__}: {e}")
            return
        # Mating check: do the parts still interlock without colliding when assembled?
        try:
            self.overlap = jwood.mating_overlap(self.joint.assemble(self.meshes))
        except Exception:
            self.overlap = float("nan")
        ps.remove_all_structures()
        for i, m in enumerate(self.meshes):
            V = np.asarray(m.vertices, float).copy()
            V[:, 0] += self.explode * i
            reg = ps.register_surface_mesh(f"part{i}", V, np.asarray(m.faces),
                                           smooth_shade=False)
            reg.set_color(PART_COLORS[i % len(PART_COLORS)])

    # ------------------------------------------------------------------ gui
    def gui(self):
        changed, idx = psim.Combo("joint", self.joints.index(self.key), self.joints)
        if changed:
            self.set_joint(self.joints[idx]); return

        ch, self.explode = psim.SliderFloat("explode", self.explode, 0.0, 3.0)
        dirty = ch
        if psim.Button("reset edits"):
            self.set_joint(self.key); return
        psim.SameLine()
        if psim.Button("export STL"):
            self.export()
        # Live mating readout: 0 = parts interlock cleanly; >0 = they collide.
        ov = getattr(self, "overlap", 0.0)
        verdict = "MATES (overlap 0)" if ov < 1e-3 else f"COLLISION overlap={ov:.3f}"
        psim.TextUnformatted(f"assembled: {verdict}")
        psim.Separator()

        for i, part in enumerate(self.joint.parts):
            stock = stock_name(part)
            if not psim.TreeNode(f"part{i}: {part['name']}"):
                continue
            psim.TextUnformatted(part["expression"])
            for name in part["lhfs"]:
                e = self.edits[i][name]
                role = "STOCK" if name == stock else "cut"
                if psim.TreeNode(f"{name} ({role})"):
                    c1, e.enabled = psim.Checkbox(f"on##{i}{name}", e.enabled)
                    c2, e.amount_mul = psim.SliderFloat(
                        f"amount##{i}{name}", e.amount_mul, 0.1, 2.0)
                    c3, e.poly_scale = psim.SliderFloat(
                        f"scale##{i}{name}", e.poly_scale, 0.3, 2.0)
                    c4, e.normal_shift = psim.SliderFloat(
                        f"shift##{i}{name}", e.normal_shift, -1.0, 1.0)
                    dirty = dirty or c1 or c2 or c3 or c4
                    psim.TreePop()
            psim.TreePop()

        if dirty:
            self.regen()

    def export(self):
        out = ensure_out()
        for i, m in enumerate(self.meshes):
            p = out / f"edit_{self.key}_{i}.stl"
            m.export(p)
        print(f"  -> wrote {len(self.meshes)} parts to {out}/edit_{self.key}_*.stl")


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "CJ_DT"
    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_up_dir("z_up")
    ps.set_navigation_style("turntable")
    editor = Editor(key)
    ps.set_user_callback(editor.gui)
    ps.show()


if __name__ == "__main__":
    main()
