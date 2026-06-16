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
