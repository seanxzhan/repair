# repair — learning wood-joint repair strategies

A machine-learning + architecture project: timber building members rot over time, and
repairing them means cutting a joint into the sound wood and splicing in a new piece. This
repo illustrates a full pipeline for **repair planning** on the
[MiGumi](https://bardofcodes.github.io/migumi/) dataset of traditional Japanese joints.

## Approach — Direction A: template prior + parametric fit

We reduce a member to a **2D longitudinal profile** (the side view), procedurally damage it,
and choose a repair from a small vocabulary of parametric **templates** (ledge, scarf,
dovetail lap, dutchman inlay). A proxy **energy** scores each candidate repair, and a classical
per-template **optimizer** acts as an *oracle* that produces the best repair. A small **CNN
prior** is trained to predict which template the oracle would pick — amortizing the oracle: at
inference the prior picks the template and the optimizer only fits that one template's
parameters.

The energy encodes the field's central tension:

> minimum intervention (remove the least sound wood) **vs.** a longer, cleaner, stronger
> structural interface.

Hard constraints: cover all the rot, preserve the coupling (mate) face, stay machinable.

## Pipeline (the `examples/` storyboard)

Each script imports from `src/repair/`, renders geometry with **polyscope** (extruded to 3D
timber prisms) and charts with **matplotlib**, and writes PNGs to `out/`.

| # | script | what it shows |
|---|--------|---------------|
| 01 | `01_load_joint.py` | load a joint → 2D member profile (+ frozen mate face) |
| 02 | `02_apply_damage.py` | procedural rot modes (central / end / ground-contact) |
| 03 | `03_templates.py` | the 4 repair templates at default parameters |
| 04 | `04_energy.py` | the 5 energy terms and the removal-vs-interface tension |
| 05 | `05_optimize.py` | fit one template's parameters (before/after) |
| 06 | `06_oracle.py` | fit all templates → pick the label; energy landscape |
| 07 | `07_generate_dataset.py` | run the oracle over joints × damage → `dataset.npz` |
| 08 | `08_train_prior.py` | train the CNN prior; confusion matrix |
| 09 | `09_inference.py` | end-to-end: prior → fit, vs the oracle |
| 10 | `10_evaluate.py` | energy regret vs baselines across generalization splits |
| — | `test_templates.py` | diagnostic/test: exploded template renders + algorithm invariant checks |

## Run it

```bash
conda activate repair          # see activate.sh
pip install -e .
python examples/01_load_joint.py        # ... through 06 (fast)
python examples/07_generate_dataset.py  # slow: runs the oracle (~a few minutes)
python examples/08_train_prior.py
python examples/09_inference.py
python examples/10_evaluate.py
# or the whole storyboard at once (use --quick for a small fast dataset):
python examples/run_all.py --quick
```

Add `--show` to the geometry scripts (01, 02, 03, 09) to open the interactive polyscope viewer.

## What generalizes (the honest result)

- **Damage generalization** (same joints, unseen rot): strong — sampled richly.
- **Topology generalization** (whole joints unseen): weaker — only 30 base joints exist.

Script 10 reports both splits separately. The defensible role of ML here is **amortized
optimization**: matching the slow oracle quickly, not zero-shot extrapolation to new joint types.

## Layout

```
src/repair/
  dataset.py     load MiGumi, extract the 2D member profile (the linchpin)
  damage.py      procedural rot + knots/checks
  templates.py   parametric repair vocabulary (ledge/scarf/dovetail/dutchman)
  energy.py      proxy energy + hard constraints
  optimizer.py   per-template fit + oracle (label source)
  rasterize.py   (member, damage) → CNN tensor
  datagen.py     build the labeled dataset + generalization splits
  prior.py       the CNN prior
  inference.py   prior → optimizer composition
  evaluate.py    energy regret + baselines
  viz.py         polyscope geometry rendering (matplotlib fallback)
  charts.py      matplotlib charts
```

## v1 simplifications

Structural/grain terms are geometric proxies (not FEA); the mate face is a heuristic
(left-end), not contact analysis; the optimizer uses penalty-folded Nelder-Mead; multi-part
joints repair one part. See `docs/` for the motivating timber-repair paper (Kloiber et al. 2023).
