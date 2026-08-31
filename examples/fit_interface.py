"""How small can the fixed-size interface program get and still cover the dataset?

`repair.interface` writes part A as a fixed-size stack of extrusions in a
canonical frame, and derives part B as `stock_B \\ dilate(solid_A, c)`. That is
only worth building on if it can actually represent real joints, so this fits
every two-part joint and measures what each budget costs.

Three questions, kept separate rather than as one grid:

  1. how many control points per ring does a profile need?
  2. how many extrusion slots does a part need?
  3. what clearance `c` reproduces the dataset's own tolerance?

    python examples/fit_interface.py            # all three sweeps
    python examples/fit_interface.py --quick    # fewer settings

Error is symmetric-difference volume against the ground-truth solid, in the
canonical frame, as a percentage of that solid's volume.
"""
from __future__ import annotations

import sys

import numpy as np

from repair.interface import (MAX_RINGS, MAX_SLOTS, build_A, canonical_frame,
                              derive_B, fit_program, list_two_part_joints)
from repair.relational import reconstruct_parts


def _sd(a, b) -> float:
    try:
        return abs(a.difference(b).volume) + abs(b.difference(a).volume)
    except Exception:
        return float("nan")


def load(key):
    """Ground truth for one joint, already in its canonical frame."""
    A, B = reconstruct_parts(key, "base")
    T = canonical_frame(A).matrix()
    out = {}
    for name, m in [("stockA", A.stock_mesh), ("stockB", B.stock_mesh),
                    ("trueA", A.solid), ("trueB", B.solid)]:
        c = m.copy()
        c.apply_transform(T)
        out[name] = c
    out["recon"] = A
    return out


def errors(gt, n_slots, n_rings, n_ctrl, clearance=0.0):
    prog = fit_program(gt["recon"], n_slots, n_rings, n_ctrl)
    ea = eb = float("nan")
    try:
        ea = _sd(build_A(prog, gt["stockA"]), gt["trueA"]) / gt["trueA"].volume * 100
    except Exception:
        pass
    try:
        fb = derive_B(prog, gt["stockA"], gt["stockB"], clearance)
        eb = _sd(fb, gt["trueB"]) / gt["trueB"].volume * 100
    except Exception:
        pass
    return ea, eb


def summarize(label, vals):
    v = np.array([x for x in vals if np.isfinite(x)])
    if not len(v):
        return f"  {label:>10s}   all failed"
    return (f"  {label:>10s}   median {np.median(v):7.3f}%   max {v.max():7.3f}%   "
            f"exact(<0.1%) {int((v < 0.1).sum()):2d}/{len(vals)}   "
            f"good(<1%) {int((v < 1.0).sum()):2d}/{len(vals)}")


def main():
    quick = "--quick" in sys.argv
    keys = list_two_part_joints()
    print(f"{len(keys)} two-part joints\n")
    gts = {k: load(k) for k in keys}

    ctrls = [4, 8, 16, 32] if quick else [3, 4, 6, 8, 12, 16, 24, 32]
    print(f"[1] control points per ring  (slots={MAX_SLOTS}, rings={MAX_RINGS})")
    for n in ctrls:
        print(summarize(f"n_ctrl={n}", [errors(g, MAX_SLOTS, MAX_RINGS, n)[0]
                                        for g in gts.values()]))

    print(f"\n[2] extrusion slots  (n_ctrl=32, rings={MAX_RINGS})")
    for k in range(1, MAX_SLOTS + 1):
        print(summarize(f"slots={k}", [errors(g, k, MAX_RINGS, 32)[0]
                                       for g in gts.values()]))

    print("\n[3] clearance c, fitted per joint at full size (error is on part B)")
    grid = [0.0, 0.0005, 0.001, 0.002, 0.004, 0.008]
    print(f"  {'joint':10s} {'best c':>8s} {'errB@c':>9s} {'errB@0':>9s}")
    best = []
    for k, g in gts.items():
        scores = [(errors(g, MAX_SLOTS, MAX_RINGS, 32, c)[1], c) for c in grid]
        scores = [(e, c) for e, c in scores if np.isfinite(e)]
        if not scores:
            print(f"  {k:10s}   all failed")
            continue
        e0 = dict((c, e) for e, c in scores).get(0.0, float("nan"))
        e, c = min(scores)
        best.append(c)
        print(f"  {k:10s} {c:8.4f} {e:8.3f}% {e0:8.3f}%")
    if best:
        print(f"\n  fitted clearance: median {np.median(best):.4f}  "
              f"range [{min(best):.4f}, {max(best):.4f}]  (canonical units)")


if __name__ == "__main__":
    main()
