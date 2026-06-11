"""Build a labeled dataset by running the oracle over joints x damage samples.

Each sample stores the rasterized (member, damage) input, the oracle's winning template
(the label), its optimal parameters, and the oracle energy. This is what the prior is
trained on and evaluated against. Splits expose the two generalization axes: held-out
*damage* (same joints) and held-out *topology* (whole joint families unseen in training).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import dataset as ds
from .damage import sample_damage
from .optimizer import oracle
from .rasterize import features_vector, rasterize
from .templates import TEMPLATE_NAMES, label_index


@dataclass
class Sample:
    joint_key: str
    joint_type: str
    damage_kind: str
    raster: np.ndarray          # (C,H,W)
    feats: np.ndarray           # (F,)
    label: int                  # index into TEMPLATE_NAMES
    theta_star: np.ndarray
    energy_star: float
    all_energies: dict          # template name -> oracle energy (for regret/baselines)

    @property
    def label_name(self) -> str:
        return TEMPLATE_NAMES[self.label]


def generate_dataset(n_per_joint: int = 8, joints: list[str] | None = None,
                     seed: int = 0, raster_res: int | None = None,
                     verbose: bool = True) -> list[Sample]:
    rng = np.random.default_rng(seed)
    keys = joints or ds.loadable_joints()
    samples: list[Sample] = []
    for ki, key in enumerate(keys):
        member = ds.load_member(key)
        jw = ds.load_jwood(key)
        mate = ds.load_mate_interface(jw, member)
        for _ in range(n_per_joint):
            dmg = sample_damage(member, rng)
            name, best, fits = oracle(member, dmg, mate, rng=rng)
            r = rasterize(member, dmg, mate) if raster_res is None \
                else rasterize(member, dmg, mate, res=raster_res)
            samples.append(Sample(
                joint_key=key, joint_type=member.meta.type, damage_kind=dmg.kind,
                raster=r, feats=features_vector(member, dmg),
                label=label_index(name), theta_star=best.theta,
                energy_star=float(best.energy.total),
                all_energies={n: float(f.energy.total) for n, f in fits.items()},
            ))
        if verbose:
            print(f"  [{ki + 1}/{len(keys)}] {key}: {n_per_joint} samples")
    return samples


def save_dataset(samples: list[Sample], path) -> None:
    np.savez_compressed(
        path,
        raster=np.stack([s.raster for s in samples]),
        feats=np.stack([s.feats for s in samples]),
        label=np.array([s.label for s in samples]),
        theta=np.array([np.pad(s.theta_star, (0, 4 - len(s.theta_star)))
                        for s in samples]),
        energy_star=np.array([s.energy_star for s in samples]),
        joint_key=np.array([s.joint_key for s in samples]),
        joint_type=np.array([s.joint_type for s in samples]),
        damage_kind=np.array([s.damage_kind for s in samples]),
        all_energies=np.array([[s.all_energies[n] for n in TEMPLATE_NAMES]
                               for s in samples]),
    )


def load_dataset(path) -> list[Sample]:
    d = np.load(path, allow_pickle=True)
    out = []
    for i in range(len(d["label"])):
        out.append(Sample(
            joint_key=str(d["joint_key"][i]), joint_type=str(d["joint_type"][i]),
            damage_kind=str(d["damage_kind"][i]), raster=d["raster"][i],
            feats=d["feats"][i], label=int(d["label"][i]),
            theta_star=d["theta"][i], energy_star=float(d["energy_star"][i]),
            all_energies={n: float(d["all_energies"][i][j])
                          for j, n in enumerate(TEMPLATE_NAMES)},
        ))
    return out


def make_splits(samples: list[Sample], mode: str = "random",
                test_frac: float = 0.25, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    if mode == "random" or mode == "held_out_damage":
        # held_out_damage == random over instances: same joints appear in train & test,
        # but the specific damage instances differ.
        rng.shuffle(idx)
        cut = int(len(idx) * (1 - test_frac))
        tr, te = idx[:cut], idx[cut:]
    elif mode == "held_out_topology":
        keys = sorted({s.joint_key for s in samples})
        rng.shuffle(np.array(keys))
        keys = list(keys)
        n_test = max(1, int(len(keys) * test_frac))
        test_keys = set(keys[:n_test])
        tr = np.array([i for i in idx if samples[i].joint_key not in test_keys])
        te = np.array([i for i in idx if samples[i].joint_key in test_keys])
    else:
        raise ValueError(mode)
    return [samples[i] for i in tr], [samples[i] for i in te]
