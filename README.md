# Homework 1: Music Era and Release-Market Classification

### Student
B12502028 吳以路

## Preparation
### Environment
Python 3.12.15.

Install the required packages:
```bash
pip install -r requirements.txt
```

## Data and Checkpoints
### Dataset structure

Keep Dataset A and Dataset B under the same parent directory. Each dataset must contain a `manifest.csv` file and an `audio/` folder:

```text
datasets/
├── dataset_A/
│   ├── manifest.csv
│   └── audio/
│       ├── A_....wav
│       └── ...
└── dataset_B/
    ├── manifest.csv
    └── audio/
        ├── B_....wav
        └── ...
```


## Inference
Generate Top-3 predictions for both datasets:

```bash
python predict.py \
  --dataset both \
  --dataset-dir /path/to/datasets \
  --output /path/to/predictions.json
```

| Argument | Required | Input / choices | Description |
|---|---|---|---|
| `--dataset` | No | `A`, `B`, or `A+B` | Select the dataset to predict. Default=`both`. |
| `--dataset-dir` | Yes | Directory Path | Path to the parent directory containing `dataset_A/` and `dataset_B/`. |
| `--output` | No | Prediction JSON File Path | Path for the output predictions JSON file. If omitted, the file is saved as `predictions.json` in the project directory. |