"""Interactive polyscope browser for the MiGumi joint dataset.

Loads the triangle meshes (the unambiguous 3D ground truth) for any joint and shows
each part as its own colored surface, with an ImGui panel to switch joint + variant
and read off how the joint is parametrized.

    python examples/inspect_dataset.py            # opens on CJ_DT
    python examples/inspect_dataset.py RJ_TGWA    # opens on a given joint

How a joint is parametrized (four nested representations per joint folder):
  base.json                  symbolic MXG node-graph DSL (Rectangle2D/PolyLine2D ->
                             LinkedHeightField3D/ApplyHeight -> Union/Difference)
  vis_files/<v>_jwood.json   evaluated form: each part = a CSG `expression` over LHFs
                             (Linked Height Fields). An LHF is a 2D `polyset` extruded
                             `amount` along `plane_normal` from `plane_origin`.
                             `state_map` holds per-part 4x4 assembly transforms.
  polyline_files/<v>.json    the editor node-graph (nodes/connections/positions)
  meshes/<v>/<i>.stl         triangulated solid, one STL per part, shared assembled
                             frame (long axis = Y, unit-square XZ cross-section)

Variants <v>: base (ideal input) | mill (open-only) | odf (open+diff-flip) | ours.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import trimesh

from repair.config import DATASET_ROOT

VARIANTS = ["base", "mill", "odf", "ours"]
# Distinct per-part colors (cycled if a joint has more parts than entries).
PART_COLORS = [(0.78, 0.76, 0.72), (0.92, 0.58, 0.18), (0.16, 0.42, 0.82),
               (0.36, 0.66, 0.38), (0.82, 0.20, 0.18)]


def list_joints() -> list[str]:
    return sorted(p.name for p in DATASET_ROOT.iterdir() if (p / "info.json").exists())


def part_paths(key: str, variant: str) -> list[Path]:
    """STL files for a joint+variant, ordered by part index (globbed, not from n_parts:
    some joints, e.g. RJM_SHS, ship more STLs than info.json claims)."""
    d = DATASET_ROOT / key / "meshes" / variant
    return sorted(d.glob("*.stl"), key=lambda p: int(p.stem)) if d.is_dir() else []


def joint_summary(key: str, variant: str) -> str:
    """One text block describing the parametrization of the current joint."""
    info = json.loads((DATASET_ROOT / key / "info.json").read_text())
    lines = [f"{key} - {info.get('name', '?')}",
             f"type: {info.get('type', '?')}   n_parts: {info.get('n_parts', '?')}"
             f"   assembly_steps: {info.get('assembly_steps', '?')}", ""]
    jw_path = DATASET_ROOT / key / "vis_files" / f"{variant}_jwood.json"
    if jw_path.exists():
        jw = json.loads(jw_path.read_text())
        lines.append(f"jwood ({variant}) - CSG over Linked Height Fields:")
        for part in jw["parts"]:
            lhfs = part.get("lhfs", {})
            amounts = [round(float(l["amount"][0]), 2) for l in lhfs.values()]
            lines.append(f"  {part['name']}: {part.get('expression', '?')}")
            lines.append(f"    {len(lhfs)} LHFs  extrude amounts={amounts}")
    else:
        lines.append(f"(no jwood for variant '{variant}')")
    return "\n".join(lines)


class Viewer:
    def __init__(self, key: str):
        self.joints = list_joints()
        self.key = key if key in self.joints else self.joints[0]
        self.variant_idx = 0
        self.explode = 0.0
        self.summary = ""
        self.load()

    def load(self):
        ps.remove_all_structures()
        variant = VARIANTS[self.variant_idx]
        paths = part_paths(self.key, variant)
        for i, p in enumerate(paths):
            mesh = trimesh.load(p, force="mesh")
            V = np.asarray(mesh.vertices, dtype=np.float64).copy()
            V[:, 0] += self.explode * i      # fan parts apart along X
            F = np.asarray(mesh.faces, dtype=np.int64)
            m = ps.register_surface_mesh(f"part{i} ({p.stem})", V, F,
                                         smooth_shade=False)
            m.set_color(PART_COLORS[i % len(PART_COLORS)])
        self.summary = joint_summary(self.key, variant)
        print(self.summary + "\n" + "-" * 60)

    def step_joint(self, d: int):
        i = (self.joints.index(self.key) + d) % len(self.joints)
        self.key = self.joints[i]
        self.load()

    def gui(self):
        psim.TextUnformatted(f"Joint {self.joints.index(self.key) + 1}"
                             f"/{len(self.joints)}")
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

        changed, self.explode = psim.SliderFloat("explode", self.explode, 0.0, 3.0)
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
