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
git lfs install
git clone https://huggingface.co/datasets/bardofcodes/migumi-dataset
```

## Inspect Dataset

```bash
python examples/inspect_dataset.py
```
