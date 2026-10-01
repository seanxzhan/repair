"""Train the capacity surrogate for one family.

    python -m repair.proto2d.train family=tenon
    python -m repair.proto2d.train family=dovetail train.epochs=60
    python -m repair.proto2d.train -m family=tenon,tenon_flip,dovetail     # one run per family

Config: configs/surrogate.yaml (hydra). Outputs go to out/proto2d/models/<family>/:
model.pt (weights + normalization), metrics.json, log.jsonl, and hydra's
.hydra/config.yaml with the resolved config.

Logging goes through `Logger`, a two-method interface (log, finish). The
console backend prints and appends to log.jsonl; the wandb backend is a
drop-in that forwards the same calls, to be enabled with log.backend=wandb.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from . import dataset as ds
from . import families as fm
from . import net as nets


# ------------------------------------------------------------------ logging
class Logger:
    """Minimal metrics sink. Subclasses: ConsoleLogger, WandbLogger."""

    def __init__(self, cfg: DictConfig, run_dir: Path):
        self.cfg, self.run_dir = cfg, run_dir

    def log(self, metrics: dict, step: int):
        raise NotImplementedError

    def finish(self):
        pass


class ConsoleLogger(Logger):
    def __init__(self, cfg, run_dir):
        super().__init__(cfg, run_dir)
        self.f = open(run_dir / "log.jsonl", "w")

    def log(self, metrics, step):
        rec = dict(step=step, **{k: float(v) for k, v in metrics.items()})
        self.f.write(json.dumps(rec) + "\n"); self.f.flush()
        if "val/rmse" in metrics:
            print(f"  step {step:6d}  " + "  ".join(f"{k} {float(v):.4f}" for k, v in metrics.items()))

    def finish(self):
        self.f.close()


class WandbLogger(ConsoleLogger):
    """Same calls forwarded to Weights & Biases. `pip install wandb` first."""

    def __init__(self, cfg, run_dir):
        super().__init__(cfg, run_dir)
        import wandb
        self.run = wandb.init(project=cfg.log.project, name=cfg.log.run_name or cfg.family,
                              tags=list(cfg.log.tags), config=OmegaConf.to_container(cfg, resolve=True),
                              dir=str(run_dir))

    def log(self, metrics, step):
        super().log(metrics, step)
        self.run.log(metrics, step=step)

    def finish(self):
        super().finish()
        self.run.finish()


LOGGERS = {"console": ConsoleLogger, "wandb": WandbLogger}


# ------------------------------------------------------------------ data
class FamilyData:
    """One family's rows and the field rasters, as tensors on the device.
    Parameters are normalized to [0, 1] by the family's bounds; the target is
    capacity over the training peak."""

    def __init__(self, data_dir, family: str, device):
        fields, fams = ds.load(data_dir)
        r = fams[family]
        self.fam = fm.FAMILIES[family]
        self.lo = np.array([q.lo for q in self.fam.params], np.float32)
        self.hi = np.array([q.hi for q in self.fam.params], np.float32)
        self.raster = torch.tensor(fields["raster"].astype(np.float32), device=device).unsqueeze(1)   # (F, 1, H, W)
        P = (r["params"] - self.lo) / (self.hi - self.lo)
        self.params = torch.tensor(P, dtype=torch.float32, device=device)
        self.field = torch.tensor(r["field"].astype(np.int64), device=device)
        self.test = torch.tensor(r["test"], device=device)
        self.peak = float(r["M"][~r["test"]].max())
        self.target = torch.tensor(r["M"] / self.peak, dtype=torch.float32, device=device)
        self.train_idx = torch.nonzero(~self.test).squeeze(1)
        self.test_idx = torch.nonzero(self.test).squeeze(1)

    def batch(self, idx):
        return self.raster[self.field[idx]], self.params[idx], self.target[idx]

    def normalization(self) -> dict:
        return dict(lo=self.lo.tolist(), hi=self.hi.tolist(), peak=self.peak,
                    param_names=[q.name for q in self.fam.params])


@torch.no_grad()
def evaluate(model, data: FamilyData, idx, batch=8192) -> dict:
    model.eval()
    preds = torch.cat([model(*data.batch(idx[i:i + batch])[:2]) for i in range(0, len(idx), batch)])
    y = data.target[idx]
    err = preds - y
    rmse = err.pow(2).mean().sqrt()
    r2 = 1.0 - err.pow(2).sum() / (y - y.mean()).pow(2).sum().clamp_min(1e-12)
    zero = y <= 0
    thr = 0.02                                   # 2% of peak: the surrogate's "dead" call
    zero_acc = ((preds <= thr) == zero).float().mean()
    nz = ~zero
    rmse_nz = err[nz].pow(2).mean().sqrt() if nz.any() else torch.tensor(0.0)
    model.train()
    return {"rmse": rmse.item(), "r2": r2.item(), "zero_acc": zero_acc.item(), "rmse_nonzero": rmse_nz.item(),
            "rmse_capacity": rmse.item() * data.peak}


def train(cfg: DictConfig, run_dir: Path, logger: Logger) -> dict:
    torch.manual_seed(cfg.train.seed); np.random.seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    data = FamilyData(cfg.data_dir, cfg.family, device)
    model = nets.build(cfg.model, len(data.fam.params)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    n_train = len(data.train_idx)
    steps_per_epoch = max(1, n_train // cfg.train.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.epochs * steps_per_epoch)
    print(f"{cfg.family}: {n_train} train rows, {len(data.test_idx)} test rows, peak capacity {data.peak:.1f}, "
          f"{sum(p.numel() for p in model.parameters())} parameters, device {device}")
    best, step, t0 = None, 0, time.time()
    for epoch in range(cfg.train.epochs):
        perm = data.train_idx[torch.randperm(n_train, device=device)]
        for i in range(steps_per_epoch):
            idx = perm[i * cfg.train.batch:(i + 1) * cfg.train.batch]
            raster, params, y = data.batch(idx)
            pred = model(raster, params)
            w = torch.where(y <= 0, torch.full_like(y, cfg.train.zero_weight), torch.ones_like(y))
            loss = (w * (pred - y).pow(2)).sum() / w.sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step(); sched.step(); step += 1
            if step % cfg.log.every == 0:
                logger.log({"train/loss": loss.item(), "lr": sched.get_last_lr()[0]}, step)
        val = evaluate(model, data, data.test_idx)
        logger.log({f"val/{k}": v for k, v in val.items()} | {"epoch": epoch}, step)
        if best is None or val["rmse"] < best["rmse"]:
            best = dict(val, epoch=epoch, step=step)
            torch.save({"state_dict": model.state_dict(), "model_cfg": OmegaConf.to_container(cfg.model),
                        "family": cfg.family, "normalization": data.normalization(), "val": best}, run_dir / "model.pt")
    metrics = dict(best=best, final=val, train_rows=n_train, test_rows=len(data.test_idx),
                   seconds=time.time() - t0, steps=step)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"best epoch {best['epoch']}: val rmse {best['rmse']:.4f} (= {best['rmse_capacity']:.2f} in capacity units), "
          f"r2 {best['r2']:.3f}, zero/nonzero acc {best['zero_acc']:.3f}  [{metrics['seconds']:.0f} s]")
    return metrics


def load_model(path, device="cpu"):
    """(model, checkpoint dict) from a model.pt written by `train`."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ck["model_cfg"])
    model = nets.build(cfg, len(ck["normalization"]["param_names"])).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model, ck


@hydra.main(version_base=None, config_path="../../../configs", config_name="surrogate")
def main(cfg: DictConfig):
    if cfg.family == "butt":
        raise SystemExit("the plain cut's capacity is identically zero; nothing to learn")
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir
                   if hydra.core.hydra_config.HydraConfig.get().mode.name == "RUN"
                   else Path(hydra.core.hydra_config.HydraConfig.get().sweep.dir) / hydra.core.hydra_config.HydraConfig.get().sweep.subdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = LOGGERS[cfg.log.backend](cfg, run_dir)
    try:
        train(cfg, run_dir, logger)
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
