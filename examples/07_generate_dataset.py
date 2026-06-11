"""07 - Build the labeled dataset by running the oracle over joints x damage samples.

Each sample = rasterized (member, damage) -> oracle's winning template + all template
energies. Saves out/dataset.npz and plots the label distribution and a feature-space PCA.

    python examples/07_generate_dataset.py [--n-per-joint N] [--quick]
"""
import sys

from repair import dataset as ds
from repair.charts import hist_labels, scatter_pca
from repair.config import ensure_out
from repair.datagen import generate_dataset, save_dataset

out = ensure_out()


def _arg(flag, default):
    return int(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def main():
    n_per = _arg("--n-per-joint", 8)
    joints = ds.loadable_joints()
    if "--quick" in sys.argv:
        joints = joints[:8]
        n_per = 4
    print(f"generating {len(joints)} joints x {n_per} = {len(joints) * n_per} samples "
          f"(oracle is ~1s/sample, please wait)...")
    samples = generate_dataset(n_per_joint=n_per, joints=joints, seed=0)
    save_dataset(samples, out / "dataset.npz")
    print(f"  saved {len(samples)} samples -> out/dataset.npz")

    hist_labels(samples, out / "07_label_dist.png")
    scatter_pca(samples, out / "07_pca.png")
    print("  -> out/07_label_dist.png, out/07_pca.png")


if __name__ == "__main__":
    main()
