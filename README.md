# Lab2: Convolutional Neural Networks

This project implements the four assignments in `assignments.ipynb`.

## Data layout

Place the MNIST IDX files under `data/MNIST`. Put VinaFood21 in ImageFolder format:

```text
data/VinaFood21/
|-- train/
|   |-- class_1/
|   `-- class_2/
|-- val/
|   |-- class_1/
|   `-- class_2/
`-- test/
    |-- class_1/
    `-- class_2/
```

The VinaFood loader also accepts a single ImageFolder directory and makes a reproducible 80/20 train/validation split. Use `val` when present, otherwise the loader uses `test` for evaluation.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run one assignment

Run commands from the `Lab2` directory:

```powershell
python -m scripts.train_assignment1
python -m scripts.train_assignment2
python -m scripts.train_assignment3
python -m scripts.train_assignment4
```

Each script reads its YAML file from `configs/`. Edit the dataset paths, batch size, learning rate, epoch count, and model options there. The code selects CUDA when available and otherwise uses CPU.

## Outputs

The best checkpoint is saved under `checkpoints/assignmentN/`. Per-epoch loss and evaluation metrics are appended to `results/assignmentN_metrics.csv`. Checkpoints, datasets, and generated results are ignored by Git by default; include a small results table in the submitted README or attach the metrics file separately if requested.

## Architectures

- Assignment 1: LeNet trained on MNIST.
- Assignment 2: GoogLeNet trained on VinaFood21. The first convolution uses padding 3 and every max-pooling layer uses `ceil_mode=True`.
- Assignment 3: ResNet-18 trained on VinaFood21. A 3 by 3 max-pooling layer with stride 2 and padding 0 is placed between residual stages.
- Assignment 4: pretrained ResNet-50 loaded from HuggingFace and fine-tuned on VinaFood21.
