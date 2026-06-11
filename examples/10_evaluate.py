"""10 - Evaluate the prior: energy regret vs baselines across generalization splits.

Energy regret = E(chosen template) - E(oracle), in oracle-energy units (0 = matches oracle).
Two honest generalization axes:
  * held_out_damage   - same joints, unseen damage instances (expect strong)
  * held_out_topology - whole joints unseen in training       (expect weaker)

    python examples/10_evaluate.py
"""
from repair.charts import bar_regret
from repair.config import ensure_out
from repair.datagen import load_dataset, make_splits
from repair.evaluate import evaluate_split
from repair.prior import TorchCNNPrior

out = ensure_out()


def main():
    path = out / "dataset.npz"
    if not path.exists():
        print("  run examples/07_generate_dataset.py first (need out/dataset.npz)")
        return
    samples = load_dataset(path)

    results = {}
    for mode in ["held_out_damage", "held_out_topology"]:
        train, test = make_splits(samples, mode=mode, seed=0)
        prior = TorchCNNPrior(in_ch=samples[0].raster.shape[0])
        prior.fit(train, epochs=40, verbose=False)
        res = evaluate_split(prior, train, test)
        results[mode] = res
        print(f"  {mode:18s} n_test={res['n_test']:3d} acc={res['accuracy']:.2f} "
              f"regret: prior={res['regret_prior']:.3f} "
              f"mf={res['regret_most_frequent']:.3f} rand={res['regret_random']:.3f}")

    bar_regret(results, out / "10_regret.png")
    print("  -> out/10_regret.png")


if __name__ == "__main__":
    main()
