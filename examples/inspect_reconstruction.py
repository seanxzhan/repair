"""Side-by-side viewer for the reverse-engineering results.

Left = ground-truth STL parts. Right = parts reconstructed from the jwood
parameters (stock - cuts, with in-plane orientations resolved against the STL).
An ImGui panel lists the parameters used per part and the recovered relational
model (members + interfaces + fillers), so you can see where reconstruction is
exact and where it drifts.

    python examples/inspect_reconstruction.py            # CJ_DT
    python examples/inspect_reconstruction.py CJ_AKT     # 3-part keyed joint

Controls: pick joint/variant, slide GROUND-TRUTH<->RECON apart (set to 0 to
overlay them), explode parts along Z, and toggle the parametric cut volumes.
"""
from __future__ import annotations

import sys

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import trimesh

from repair.config import DATASET_ROOT
from repair.relational import import_joint, reconstruct_parts

VARIANTS = ["base", "mill", "odf", "ours"]
PART_COLORS = [(0.78, 0.76, 0.72), (0.92, 0.58, 0.18), (0.16, 0.42, 0.82),
               (0.36, 0.66, 0.38), (0.82, 0.20, 0.18)]


def list_joints() -> list[str]:
    return sorted(p.name for p in DATASET_ROOT.iterdir() if (p / "info.json").exists())


def _show(name, mesh, color, dx, dz, transparency=1.0):
    """Register a trimesh as a polyscope surface, translated by (dx, 0, dz)."""
    if mesh is None or mesh.is_empty or len(mesh.faces) == 0:
        return
    V = np.asarray(mesh.vertices, np.float64).copy()
    V[:, 0] += dx
    V[:, 2] += dz
    m = ps.register_surface_mesh(name, V, np.asarray(mesh.faces, np.int64),
                                 smooth_shade=False, transparency=transparency)
    m.set_color(color)


class Viewer:
    def __init__(self, key: str):
        self.joints = list_joints()
        self.key = key if key in self.joints else self.joints[0]
        self.variant_idx = 0
        self.separation = -1.0     # -1 => auto from joint width
        self.explode = 0.0
        self.show_cuts = False
        self.summary = ""
        self.load()

    def load(self):
        ps.remove_all_structures()
        variant = VARIANTS[self.variant_idx]
        try:
            recons = reconstruct_parts(self.key, variant)
            joint = import_joint(self.key, variant)
        except Exception as e:  # missing variant / bad data
            self.summary = f"{self.key} ({variant}): cannot load\n{e!r}"
            print(self.summary)
            return

        # Auto separation = 1.5x the GT bounding width in X.
        gt = trimesh.util.concatenate([r.stl for r in recons])
        width = float(gt.bounds[1, 0] - gt.bounds[0, 0])
        dx = width * 1.5 if self.separation < 0 else self.separation

        lines = [joint.summary(), "", "parameters used for reconstruction:"]
        for i, r in enumerate(recons):
            color = PART_COLORS[i % len(PART_COLORS)]
            dz = self.explode * i
            _show(f"GT/{r.name}", r.stl, color, 0.0, dz)
            _show(f"REC/{r.name}", r.solid, color, dx, dz)
            if self.show_cuts:
                for j, cm in enumerate(r.cut_meshes):
                    _show(f"REC/{r.name}/cut{j}", cm, (0.85, 0.15, 0.15), dx, dz,
                          transparency=0.35)
            lines.append(r.param_text())

        self.summary = "\n".join(lines)
        print(self.summary + "\n" + "-" * 60)

    def step_joint(self, d: int):
        i = (self.joints.index(self.key) + d) % len(self.joints)
        self.key = self.joints[i]
        self.load()

    def gui(self):
        psim.TextUnformatted(f"GROUND TRUTH (left)   |   RECONSTRUCTED (right)")
        psim.TextUnformatted(f"Joint {self.joints.index(self.key) + 1}/{len(self.joints)}")
        if psim.Button("<- prev"):
            self.step_joint(-1)
        psim.SameLine()
        if psim.Button("next ->"):
            self.step_joint(+1)

        changed, idx = psim.Combo("joint", self.joints.index(self.key), self.joints)
        if changed:
            self.key = self.joints[idx]
            self.load()
        changed, self.variant_idx = psim.Combo("variant", self.variant_idx, VARIANTS)
        if changed:
            self.load()

        changed, sep = psim.SliderFloat("separation (0=overlay)", max(self.separation, 0.0),
                                        0.0, 6.0)
        if changed:
            self.separation = sep
            self.load()
        changed, self.explode = psim.SliderFloat("explode (Z)", self.explode, 0.0, 3.0)
        if changed:
            self.load()
        changed, self.show_cuts = psim.Checkbox("show cut volumes", self.show_cuts)
        if changed:
            self.load()

        psim.Separator()
        psim.TextUnformatted(self.summary)


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "CJ_DT"
    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_up_dir("z_up")
    ps.set_navigation_style("turntable")
    viewer = Viewer(key)
    ps.set_user_callback(viewer.gui)
    ps.show()


if __name__ == "__main__":
    main()
