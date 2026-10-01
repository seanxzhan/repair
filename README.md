# repair — learning wood-joint repair strategies

## Install

Set up environment:
```bash
conda env create -f env.yml
source activate.sh 
pip install -e .
```

Install CUDA if testing out `proto2d`:
```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

Grab the dataset:
```bash
# run the command below outside of this repo under ~/
git lfs install

# run the command below inside this repo
git clone https://huggingface.co/datasets/bardofcodes/migumi-dataset
```

## Inspect Dataset

```bash
python examples/inspect_dataset.py
```

## Inspect Reverse-Engineering

Reconstruct each part from its jwood parameters and view ground-truth vs. reconstructed side by side, with the parameters used and the recovered relational model (members + interfaces + fillers):

```bash
python examples/inspect_reconstruction.py            # CJ_DT
python examples/inspect_reconstruction.py CJ_AKT     # 3-part keyed joint
```

## Inspect Cuts

Every part is `Difference(stock, Union(cuts...))`, and each cut is an LHF: a 2D sketch on a plane, swept `amount` along that plane's normal. This takes those cuts apart one at a time, with the sketch and plane frame drawn where they sit:

```bash
python examples/inspect_cuts.py              # CJ_DT
python examples/inspect_cuts.py CJ_AKT       # 3-part keyed joint (holes)
python examples/inspect_cuts.py --headless   # cut listing + verification only
```

**CUTS** steps through a part's cuts -- selected one solid, the rest ghosted -- drawing its sketch rings at depth 0 and at `amount` (the cut is the volume between them), its plane origin, and the `(u, v, n)` frame. **ORIENT** shows all 8 in-plane orientations of the selected cut, grouped by which are the same solid, and which one the importer resolved to. **VERIFY** is a check rather than a workshop: it constructs cuts from scratch and confirms each survives a JSON round trip through the unmodified reader and comes back at `ori=0` with no STL to lean on -- which is what licenses the canonical exports below.

## Inspect the Repair-Interface Model

The 2D model behind `docs/proposal.md` (Figure 1): one block with a rotten end is
the input, and a mortise-and-tenon splice -- shoulder position, tenon length,
cheeks -- is the output. Every parameter of the interface, the damage front and
the statics is a slider; each change rebuilds the contact set and re-solves the
frictional-equilibrium LP, showing live/dead contacts, the LP's contact forces,
the moment the interface can transfer and the sound wood it throws away.
`compute landscape` sweeps (shoulder x tenon length) and draws capacity, the
objective or removed wood below the block, with the active-set cliffs overlaid.

```bash
python examples/proto2d/inspect_splice.py              # open the viewer
python examples/proto2d/inspect_splice.py --headless   # readouts only
```

The model itself is `src/repair/proto2d/model.py`; `docs/figures/proposal.py` renders
it once at the viewer's default settings.

## Inspect the 2D Interface Families

`src/repair/proto2d/families.py` makes the interface family the variable: a family
turns its parameters into the region of the block the new wood replaces (as a
polygon, per width layer), the contact faces are read off that polygon's edges,
and the same LP, damage field and sound-wood integral apply. Seven families: a
plain cut, the mortise-and-tenon of Figure 1, 2D cross-sections of two MiGumi
splices -- `CJ_AT` (dovetail) and `CJ_DT` (hooked scarf) -- and, as families of
their own, the flipped orientation of each of those three: the same joint cut
the other way round, so the tongue belongs to the retained wood or the scarf's
retained wedge sits above. In sound wood a flip carries exactly its original's
moment; under a leaning damage front the two differ. The tenon family reproduces
`model.py` to machine precision; `sanity_checks()` says how the families nest
and that every flip is a true mirror.

```bash
python examples/proto2d/inspect_families.py              # viewer: pick a family, every parameter a slider
python examples/proto2d/inspect_families.py --headless   # every family at its defaults, as text
```

Damage is either the parametric front of Figure 1 or a random severity field
from `src/repair/proto2d/damage.py` (erosion from seeds on the boundary: a
logistic of the anisotropic distance, with the front displaced by smooth noise;
every knob a slider, or drawn at random from the data-generation ranges). **compute landscape** sweeps any
two of the family's parameters and draws capacity, the objective or the sound
wood removed with the active-set cliffs overlaid; **go to landscape optimum**
moves the two swept sliders onto the orange marker.

## Generate the 2D Dataset

Fields times interfaces: each random damage field gets, for every family, a
batch of interfaces drawn within the family's bounds; each row stores the
clipped parameters, the field index, capacity, sound wood removed and per-face
live counts. The split is by field. `fields.npz` carries each field's seed and
knobs (it is rebuilt exactly from them) and a 96 x 32 severity raster of the
interface window, the network's damage input.

```bash
python examples/proto2d/generate_dataset.py                 # 2000 fields x 20 per family -> out/proto2d
python examples/proto2d/report_dataset.py --n 4             # sample rows, re-solve, draw -> out/proto2d/report.html
python examples/proto2d/report_dataset.py --test-only       # only held-out rows
```

The report is one self-contained HTML file (images embedded as base64); each
example prints the stored label next to the re-solved one.

## Train and Evaluate the Surrogate

The surrogate (`src/repair/proto2d/net.py`) reads the damage raster through a
small convolutional encoder and the normalized interface parameters through an
MLP, and predicts capacity. Training (`train.py`) and evaluation (`evaluate.py`)
are hydra entry points; their configs are `configs/surrogate.yaml` and
`configs/evaluate.yaml`, and any field can be overridden on the command line or
by writing a new config. Logging goes through a two-method `Logger`; the console
backend writes `log.jsonl`, and `log.backend=wandb` switches to Weights & Biases
once `wandb` is installed.

```bash
python -m repair.proto2d.train family=tenon                       # -> out/proto2d/models/tenon/
python -m repair.proto2d.train -m family=tenon,dovetail,hooked_scarf   # one run per family
python -m repair.proto2d.train family=dovetail train.epochs=400 train.lr=3e-4
python -m repair.proto2d.evaluate                                 # -> out/proto2d/models/report.html
```

The evaluation report gives test metrics per family on held-out fields and,
for a few held-out fields each, one parameter swept with the LP's staircase
against the surrogate's curve, drawn as inline SVG.

## Export / Inspect Our Own Parametrization

The dataset's `vis_files/*_jwood.json` cannot be evaluated on their own: they store a 2D sketch and a plane but not the sketch's in-plane orientation, so a reader must recover it by searching against the ground-truth STL. Export a canonical form once (every LHF at orientation 0) and that dependency is gone:

```bash
python examples/export_lhf.py        # 30 joints -> out/lhf_base/<KEY>.json
python examples/inspect_lhf.py       # view them, dataset not required
```

Each export keeps the dataset's `parts` shape (so `jwood.PartSpec.from_json` reads it unchanged) and adds `provenance` (the orientations the resolver recovered, plus agreement / volume error, so the file is auditable) and `relational` (recovered members, interfaces, fillers). `export_lhf.py` verifies each part by rebuilding it from the file it just wrote and comparing to the STL.

`inspect_lhf.py` reads only `out/lhf_base/*.json` -- audited: rendering all 30 joints opens 30 files, no dataset, no STL, no orientation cache.
