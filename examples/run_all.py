"""Run the whole storyboard 01 -> 10 in order, regenerating out/*.png.

    python examples/run_all.py [--quick]

--quick shrinks the dataset (script 07) so the full run finishes in a couple of minutes.
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).parent
SCRIPTS = ["01_load_joint", "02_apply_damage", "03_templates", "04_energy",
           "05_optimize", "06_oracle", "07_generate_dataset", "08_train_prior",
           "09_inference", "10_evaluate"]


def main():
    quick = "--quick" in sys.argv
    for name in SCRIPTS:
        print(f"\n=== {name} ===")
        argv = sys.argv[:1]
        if name == "07_generate_dataset" and quick:
            argv = argv + ["--quick"]
        sys.argv = argv
        runpy.run_path(str(HERE / f"{name}.py"), run_name="__main__")


if __name__ == "__main__":
    main()
