"""Reverse-engineer every dataset joint into (type, parameters) -- the training scaffold
for a classify-then-regress editor.

Stage 1 (classify): predict the joint **type** = reverse.signature(joint). 30 joints -> 20
types; some types (e.g. the CJ_SAT scarf family) have several instances.
Stage 2 (regress): predict the per-cut **parameters** within that type.

    python examples/catalog_types.py        # prints the catalog, writes out/joint_types.json
"""
from __future__ import annotations

import json

from repair import jwood, reverse
from repair.config import ensure_out


def main():
    joints = [reverse.load(k) for k in
              sorted(p.name for p in jwood.DATASET_ROOT.iterdir()
                     if (p / "info.json").exists())]

    # Assign an integer type id per distinct signature (the classification label).
    sigs = sorted({jp.signature for jp in joints}, key=str)
    type_id = {s: i for i, s in enumerate(sigs)}

    by_type: dict[int, list] = {}
    for jp in joints:
        by_type.setdefault(type_id[jp.signature], []).append(jp)

    print(f"{len(joints)} joints -> {len(sigs)} types (classification classes)\n")
    table = {}
    for tid in sorted(by_type):
        members = by_type[tid]
        sig = members[0].signature
        topo = "; ".join(f"{t}|{list(c)}" for t, c in sig[1][:1])
        print(f"TYPE {tid:2d}  ({len(members)} instance{'s' if len(members)>1 else ''})"
              f"  {sig[0]}p  {topo}")
        for jp in members:
            pv = jp.param_vector()
            nprof = sum(len(cp.profile) for cuts in jp.parts for cp in cuts if not cp.is_stock)
            scal = {k: v for k, v in pv.items() if not k.endswith("normal_class")}
            print(f"    {jp.key:10s} {len(scal):2d} scalar params, "
                  f"{nprof:3d} profile verts to predict")
            table[jp.key] = {"type_id": tid, "params": pv}

    out = ensure_out() / "joint_types.json"
    out.write_text(json.dumps(table, indent=2, default=str))
    print(f"\n-> wrote {out}  (per-joint type_id + parameter vector)")


if __name__ == "__main__":
    main()
