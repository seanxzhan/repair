"""08 - Train the CNN prior to predict the oracle's template from the raster.

Loads out/dataset.npz, trains a small CNN, saves out/prior.pt, and plots the confusion
matrix + training loss on a held-out (random) split.

    python examples/08_train_prior.py
"""
import matplotlib.pyplot as plt

from repair.charts import confusion, save
from repair.config import ensure_out
from repair.datagen import load_dataset, make_splits
from repair.evaluate import confusion_matrix, template_accuracy
from repair.prior import TorchCNNPrior

out = ensure_out()


def main():
    path = out / "dataset.npz"
    if not path.exists():
        print("  run examples/07_generate_dataset.py first (need out/dataset.npz)")
        return
    samples = load_dataset(path)
    train, test = make_splits(samples, mode="random", seed=0)
    print(f"  {len(train)} train / {len(test)} test")

    prior = TorchCNNPrior(in_ch=samples[0].raster.shape[0])
    prior.fit(train, epochs=40)
    prior.save(out / "prior.pt")
    acc = template_accuracy(prior, test)
    print(f"  held-out template accuracy: {acc:.2f}")

    confusion(confusion_matrix(prior, test), out / "08_confusion.png",
              title=f"Template confusion (held-out acc={acc:.2f})")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(prior.history)
    ax.set_xlabel("epoch"); ax.set_ylabel("train loss"); ax.set_title("CNN prior training")
    save(fig, out / "08_loss.png")
    print("  -> out/prior.pt, out/08_confusion.png, out/08_loss.png")


if __name__ == "__main__":
    main()
