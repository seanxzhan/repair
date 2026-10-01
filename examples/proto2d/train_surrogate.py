"""Train the capacity surrogate (role A) for one family.

    python examples/proto2d/train_surrogate.py family=tenon
    python examples/proto2d/train_surrogate.py family=dovetail train.epochs=400 train.lr=3e-4
    python examples/proto2d/train_surrogate.py -m family=tenon,tenon_flip,dovetail     # one run per family
    python examples/proto2d/train_surrogate.py --config-name my_experiment             # configs/my_experiment.yaml

Config: configs/surrogate.yaml. Outputs: out/proto2d/models/<family>/ (model.pt,
metrics.json, log.jsonl, .hydra/config.yaml). The loop and the loggers live in
repair.proto2d.train.
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

try:
    from repair.proto2d import train as tr
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from repair.proto2d import train as tr


@hydra.main(version_base=None, config_path="../../configs", config_name="surrogate")
def main(cfg: DictConfig):
    if cfg.family == "butt":
        raise SystemExit("the plain cut's capacity is identically zero; nothing to learn")
    hc = HydraConfig.get()
    run_dir = Path(hc.run.dir) if hc.mode.name == "RUN" else Path(hc.sweep.dir) / hc.sweep.subdir
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = tr.LOGGERS[cfg.log.backend](cfg, run_dir)
    try:
        tr.train(cfg, run_dir, logger)
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
