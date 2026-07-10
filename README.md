# Baggage Detection

A PyTorch project for classifying illegal/prohibited items in X-ray baggage images.

## What it does

Loads X-ray images with YOLO-format label files (class id + bounding box), and fine-tunes a ResNet18 image classifier (via transfer learning) to predict the item class present in each image. Includes a data ingestion pipeline, a training/validation/testing loop, and a small utility for visualizing a sample image with its bounding boxes drawn on top.

## Tech stack

- Python
- PyTorch / torchvision (ResNet18 with `ResNet18_Weights.DEFAULT`)
- Pillow, matplotlib
- pandas, numpy, seaborn, scikit-learn (listed as dependencies, not yet used in the current code)
- tqdm for training progress bars

## Setup

```bash
pip install -r requirements.txt
```

## Usage

The dataset is expected in YOLO layout (`images/` + `labels/` folders per split) under a `data/` directory with `train`, `valid`, and `test` subfolders. The path is currently hardcoded in `src/components/data_ingestion.py`:

```python
data_dir = "D:/Code/ML_projects/baggage_detection/data"
```

You'll need to edit this path to point at your own dataset before running anything.

Once the path is set, run data ingestion directly to kick off training (this module builds the data loaders and immediately starts training a `ModelTrain` instance for 20 epochs):

```bash
python src/components/data_ingestion.py
```

There's also a Jupyter notebook, `src/notebooks/data_ingestion.ipynb`, used for exploring data ingestion interactively.

## Status: Work in progress

This is an early-stage/experimental project, not a finished product:

- `src/pipeline/train_pipeline.py`, `src/pipeline/eval_pipeline.py`, and `src/pipeline/inference_pipeline.py` are all empty placeholder files — no standalone train/eval/inference entry points exist yet. Training is currently only invoked as a side effect of importing `data_ingestion.py`.
- The dataset path in `data_ingestion.py` is hardcoded to a local machine path and will need to be changed by anyone else running the code.
- `stepsfile.txt` (the author's own working notes) lists open TODOs: "build a model or check how many classes in the labels" and "building the train function," confirming the project is still under active development.
- No saved model weights, evaluation metrics, or results are included in the repo.
