"""Export our own canonical LHF parametrization, one JSON per joint.

The dataset's `vis_files/*_jwood.json` cannot be evaluated on their own: they
store a 2D sketch and a plane but not the sketch's in-plane orientation, so a
reader has to recover it by searching against the ground-truth STL. These
exports fix that. Every LHF is canonicalized to orientation 0 (sketch coords in
`jwood.seed_basis(plane_normal)`), so the file evaluates with no STL and no
search:

    spec  = PartSpec.from_json(d["parts"][i])
    solid, _, _ = build_solid(spec.stock, 0, spec.cuts, [0] * len(spec.cuts))

`parts` keeps the dataset's own shape, so existing readers work unchanged. The
extra top-level keys are additive: `provenance` records what the resolver
recovered (so the export is auditable against the dataset), and `relational`
carries the recovered members / interfaces / fillers.

    python examples/export_lhf.py                  # all 30 joints, base variant
    python examples/export_lhf.py CJ_DT CJ_AKT     # just these
    python examples/export_lhf.py --out some/dir   # somewhere other than out/lhf_base
    python examples/export_lhf.py --no-relational  # geometry only, skip the joint graph

Every part is verified after writing by rebuilding it from the file it just
wrote and comparing against the STL.
"""
from __future__ import annotations

import sys

from repair.authoring import export_joint
from repair.config import DATASET_ROOT

VARIANT = "base"          # the only variant these exports cover, by request


def list_joints() -> list[str]:
    return sorted(p.name for p in DATASET_ROOT.iterdir()
                  if (p / "vis_files" / f"{VARIANT}_jwood.json").exists())


def main():
    args = sys.argv[1:]
    out_dir, relational = None, True
    if "--no-relational" in args:
        args.remove("--no-relational")
        relational = False
    if "--out" in args:
        i = args.index("--out")
        out_dir = args[i + 1]
        del args[i:i + 2]

    keys = args or list_joints()
    worst, failures, n_parts = (0.0, ""), [], 0
    print(f"exporting {len(keys)} joints (variant={VARIANT}, relational={relational})\n")
    for key in keys:
        try:
            path, rows = export_joint(key, VARIANT, out_dir=out_dir,
                                      relational=relational)
        except Exception as e:
            failures.append((key, repr(e)))
            print(f"  {key:10s} FAILED  {e!r}")
            continue
        n_parts += len(rows)
        err = max((r["vol_error"] for r in rows), default=0.0)
        if err > worst[0]:
            worst = (err, key)
        flag = "" if err < 1e-3 else "   <-- CHECK"
        print(f"  {key:10s} {len(rows)} parts  max vol_error {err * 100:6.3f}%"
              f"  -> {path}{flag}")

    print(f"\n{len(keys) - len(failures)}/{len(keys)} joints, {n_parts} parts written")
    print(f"worst volume error vs STL: {worst[0] * 100:.4f}%"
          + (f" ({worst[1]})" if worst[1] else ""))
    if failures:
        print(f"failures: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
