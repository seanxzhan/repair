"""Polyscope viewer for OUR OWN exported LHF parametrizations.

Reads nothing but `out/lhf_base/*.json` -- no `migumi-dataset`, no STL, no
orientation cache, no orientation search. That is the whole point: the exports
written by `export_lhf.py` are canonical (every LHF's sketch is in
`jwood.seed_basis(plane_normal)`, i.e. orientation 0), so a part is just

    solid = stock - union(cuts)

evaluated directly. If this viewer renders a joint correctly, the export really
does stand on its own. (Checking the result against the ground-truth STL is
`export_lhf.py`'s job, not this one's -- it would defeat the demonstration.)

    python examples/inspect_lhf.py                 # first joint in the dir
    python examples/inspect_lhf.py CJ_AKT          # a given joint
    python examples/inspect_lhf.py --dir some/dir  # a different export directory

Controls: pick a joint, explode the parts, show the stock blocks the parts were
cut from, show the parametric cut volumes. Fillers (keys / pegs / wedges) are
named as such, from the `relational` block in the file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from repair.authoring import export_part_solids, load_export
from repair.config import OUT_DIR

DEFAULT_DIR = OUT_DIR / "lhf_base"
PART_COLORS = [(0.78, 0.76, 0.72), (0.92, 0.58, 0.18), (0.16, 0.42, 0.82),
               (0.36, 0.66, 0.38), (0.82, 0.20, 0.18)]
FILLER_COLOR = (0.90, 0.72, 0.30)
CUT_COLOR = (0.85, 0.15, 0.15)


def _show(name, mesh, color, dz=0.0, transparency=1.0):
    if mesh is None or mesh.is_empty or len(mesh.faces) == 0:
        return
    V = np.asarray(mesh.vertices, np.float64).copy()
    V[:, 2] += dz
    m = ps.register_surface_mesh(name, V, np.asarray(mesh.faces, np.int64),
                                 smooth_shade=False, transparency=transparency)
    m.set_color(color)


def _lhf_text(tag: str, lhf) -> str:
    rings = (f"{len(lhf.rings)} rings {lhf.signs}" if len(lhf.rings) > 1
             else f"{len(lhf.rings[0])}-gon")
    vec = lambda v: "[" + ",".join(f"{x:+.2f}" for x in v) + "]"
    return (f"  {tag:6s} {lhf.name}: n={vec(lhf.normal)} o={vec(lhf.origin)} "
            f"amount={lhf.amount:+.3f} ({rings})")


class Viewer:
    def __init__(self, export_dir: Path, key: str | None):
        self.dir = Path(export_dir)
        self.joints = sorted(p.stem for p in self.dir.glob("*.json"))
        if not self.joints:
            raise SystemExit(f"no exports in {self.dir} -- run examples/export_lhf.py first")
        self.key = key if key in self.joints else self.joints[0]
        self.explode = 0.0
        self.show_cuts = False
        self.show_stock = False
        self.summary = ""
        self.load()

    def load(self):
        ps.remove_all_structures()
        try:
            self.doc = load_export(self.dir / f"{self.key}.json")
            self.parts = export_part_solids(self.doc)
        except Exception as e:
            self.summary = f"{self.key}: cannot load\n{e!r}"
            print(self.summary)
            return

        rel = self.doc.get("relational", {})
        fillers = {m["name"] for m in rel.get("members", []) if m.get("is_filler")}

        for i, p in enumerate(self.parts):
            dz = self.explode * i
            is_filler = p["name"] in fillers
            color = FILLER_COLOR if is_filler else PART_COLORS[i % len(PART_COLORS)]
            _show(f"{p['name']}{' [filler]' if is_filler else ''}", p["solid"], color, dz)
            if self.show_stock:
                _show(f"{p['name']}/stock", p["stock"], (0.55, 0.55, 0.55), dz,
                      transparency=0.15)
            if self.show_cuts:
                for j, cm in enumerate(p["cut_meshes"]):
                    _show(f"{p['name']}/cut{j}", cm, CUT_COLOR, dz, transparency=0.35)

        self.summary = self.build_summary(rel, fillers)
        print(self.summary + "\n" + "-" * 60)

    def build_summary(self, rel, fillers) -> str:
        d, prov = self.doc, self.doc.get("provenance", {})
        lines = [f"{d['key']} ({d['variant']})   schema {d['schema']}",
                 f"source: {d.get('source', '?')}",
                 f"evaluated at ori=0 from the file alone -- no STL, no resolve.", ""]
        for p in self.parts:
            spec, pv = p["spec"], prov.get(p["name"], {})
            tag = "  [FILLER]" if p["name"] in fillers else ""
            lines.append(f"{p['name']}{tag}: {spec.expression}")
            lines.append(f"  solid_vol={p['solid'].volume:.4f} "
                         f"stock_vol={p['stock'].volume:.4f}")
            lines.append(_lhf_text("stock", spec.stock))
            for c in spec.cuts:
                lines.append(_lhf_text("cut", c))
            if pv:
                lines.append(f"  provenance: dataset oris stock={pv['resolved_stock_ori']} "
                             f"cuts={pv['resolved_cut_oris']} -> canonicalized to 0; "
                             f"agreement={pv['stl_agreement'] * 100:.2f}% "
                             f"vol_error={pv['vol_error'] * 100:.3f}%")
        if rel.get("interfaces"):
            lines += ["", f"relational model: {len(rel['members'])} members, "
                          f"{len(rel['interfaces'])} interfaces"]
            for itf in rel["interfaces"]:
                roles = ", ".join(f"{i['part']}:{i['role']}({i['volume']:.2f})"
                                  for i in itf["incidences"])
                lines.append(f"  {itf['id']} (vol={itf['volume']:.2f}): {roles}")
        elif "error" in rel:
            lines += ["", f"relational model unavailable: {rel['error']}"]
        return "\n".join(lines)

    def step(self, d: int):
        self.key = self.joints[(self.joints.index(self.key) + d) % len(self.joints)]
        self.load()

    def gui(self):
        psim.TextUnformatted(f"OUR OWN LHF EXPORTS -- {self.dir}")
        psim.TextUnformatted(f"Joint {self.joints.index(self.key) + 1}/{len(self.joints)}")
        if psim.Button("<- prev"):
            self.step(-1)
        psim.SameLine()
        if psim.Button("next ->"):
            self.step(+1)

        changed, idx = psim.Combo("joint", self.joints.index(self.key), self.joints)
        if changed:
            self.key = self.joints[idx]
            self.load()
        changed, self.explode = psim.SliderFloat("explode (Z)", self.explode, 0.0, 3.0)
        if changed:
            self.load()
        changed, self.show_cuts = psim.Checkbox("show cut volumes", self.show_cuts)
        if changed:
            self.load()
        psim.SameLine()
        changed, self.show_stock = psim.Checkbox("show stock", self.show_stock)
        if changed:
            self.load()

        psim.Separator()
        psim.TextUnformatted(self.summary)


def main():
    args = sys.argv[1:]
    export_dir = DEFAULT_DIR
    if "--dir" in args:
        i = args.index("--dir")
        export_dir = Path(args[i + 1])
        del args[i:i + 2]
    key = args[0] if args else None

    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_up_dir("z_up")
    ps.set_navigation_style("turntable")
    viewer = Viewer(export_dir, key)
    ps.set_user_callback(viewer.gui)
    ps.show()


if __name__ == "__main__":
    main()
