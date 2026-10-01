"""Step 2: generate the labelled dataset (fields x interfaces, per family).

    python examples/proto2d/generate_dataset.py                       # 2000 fields x 20 per family -> out/proto2d
    python examples/proto2d/generate_dataset.py --fields 200 --per-field 5 --out out/proto2d_small
    python examples/proto2d/generate_dataset.py --families tenon dovetail

See repair.proto2d.dataset for the row layout; examples/proto2d/report_dataset.py
renders a sample of rows to an HTML file for inspection.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    from repair.proto2d import dataset as ds, families as fm
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from repair.proto2d import dataset as ds, families as fm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/proto2d")
    ap.add_argument("--fields", type=int, default=2000)
    ap.add_argument("--per-field", type=int, default=20)
    ap.add_argument("--families", nargs="*", default=None, choices=list(fm.FAMILIES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    keys = a.families or list(fm.FAMILIES)
    print(f"{a.fields} fields x {a.per_field} interfaces x {len(keys)} families = {a.fields * a.per_field * len(keys)} rows -> {a.out}")
    t0 = time.time()
    ds.generate(a.out, a.fields, a.per_field, keys, a.seed, a.test_frac, a.workers)
    fields, fams = ds.load(a.out)
    print(f"done in {time.time() - t0:.0f} s; {fields['test'].sum()} of {len(fields['test'])} fields held out")
    for k, r in fams.items():
        M = r["M"]
        print(f"  {k:18s} {len(M):7d} rows  M: zero {100 * (M <= 1e-6).mean():4.1f}%  mean {M.mean():6.1f}  max {M.max():6.1f}"
              f"  |  R mean {r['R'].mean():.2f}  |  live {r['live'].mean():.1f}/{r['n_contacts'].mean():.1f}")


if __name__ == "__main__":
    main()
