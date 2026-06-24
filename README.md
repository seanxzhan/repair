# repair — learning wood-joint repair strategies

## Install

Set up environment:
```bash
conda env create -f env.yml
source activate.sh 
pip install -e .
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
