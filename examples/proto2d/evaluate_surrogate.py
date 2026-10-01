"""Evaluate trained surrogates on held-out fields; writes an HTML report.

    python examples/proto2d/evaluate_surrogate.py                              # -> out/proto2d/models/report.html
    python examples/proto2d/evaluate_surrogate.py families=[tenon] sweep.n_fields=8

Config: configs/evaluate.yaml. The metrics and the sweep figure live in
repair.proto2d.evaluation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

try:
    from repair.proto2d import evaluation as ev
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from repair.proto2d import evaluation as ev


@hydra.main(version_base=None, config_path="../../configs", config_name="evaluate")
def main(cfg: DictConfig):
    print(ev.report(cfg))


if __name__ == "__main__":
    main()
