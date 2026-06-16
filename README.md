# SPPIFN

SPPIFN is a computer vision project for produce freshness classification. Models take an image as input and predict whether the produce is fresh or rotten. The training pipeline is optimized for lightweight deployment targets, including older-generation Apple iPads via CoreML.

The repo includes utilities for:
- converting trained PyTorch checkpoints to CoreML
- augmenting image datasets
- evaluating pretrained and fine-tuned models
- running inference on single images or image folders
- restructuring the Kaggle dataset so freshness becomes the top-level class

## Project Layout

- `runner.py`: main entry point for training and evaluation
- `utils/restructure_by_freshness.py`: converts dataset layout from `produce/freshness` to `freshness/produce`
- `utils/data_augment.py`: creates augmented training images
- `utils/eval_dataset.py`: compares fine-tuned checkpoints against pretrained baselines
- `utils/inference.py`: runs prediction on sample images
- `utils/convert_coreML.py`: exports checkpoints to `.mlmodel`

## Quick Start

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate SPPIFN
```

### 2. Download the Kaggle dataset

If you have Kaggle API access configured locally, you can download the dataset with:

```bash
mkdir -p data/raw/kaggle
curl -L -o data/raw/kaggle/fruitquality1.zip \
  https://www.kaggle.com/api/v1/datasets/download/zlatan599/fruitquality1
unzip data/raw/kaggle/fruitquality1.zip -d data/raw/kaggle/fruitquality1
```

If the direct `curl` request fails, use the Kaggle CLI instead:

```bash
kaggle datasets download -d zlatan599/fruitquality1 -p data/raw/kaggle
unzip data/raw/kaggle/fruitquality1.zip -d data/raw/kaggle/fruitquality1
```

### 3. Restructure the dataset for freshness classification

The original dataset layout groups images by produce type first. PyTorch `ImageFolder` uses the top-level folders as class labels, so the dataset needs to be restructured before training.

```bash
python utils/restructure_by_freshness.py \
  --input_dir data/raw/kaggle/fruitquality1 \
  --output_dir data/processed/kaggle_freshness_structured
```

By default this creates symlinks to save disk space. Use `--copy` if you want physical copies instead.

### 4. Optional: augment the dataset

```bash
python utils/data_augment.py \
  --input_dir data/raw/kaggle/fruitquality1 \
  --output_dir data/processed/kaggle_augmented \
  --multiplier 2
```

If you augment first, restructure the augmented dataset instead of the raw one:

```bash
python utils/restructure_by_freshness.py \
  --input_dir data/processed/kaggle_augmented \
  --output_dir data/processed/kaggle_freshness_structured
```

## Training

Train a baseline model:

```bash
python runner.py train \
  --data_dir data/processed/kaggle_freshness_structured \
  --model mobilenet_v3_large \
  --epochs 10 \
  --batch_size 32 \
  --lr 0.001 \
  --image_size 224 \
  --num_workers 4 \
  --val_split 0.2 \
  --seed 42
```

Train a stronger production-oriented configuration:

```bash
python runner.py train \
  --data_dir data/processed/kaggle_freshness_structured \
  --model efficientnet_b2 \
  --epochs 40 \
  --batch_size 32 \
  --lr 0.001 \
  --backbone_lr 0.0001 \
  --optimizer sgd \
  --freeze_backbone False \
  --use_advanced_aug True \
  --use_mixup True \
  --use_cutmix True \
  --use_lr_scheduler True \
  --run_name efficientnet_b2_production \
  --num_workers 6
```

Training outputs:
- metrics logs are written to `logs/`
- best checkpoints are written to `checkpoints/<model>_<timestamp>/best.pt`
- checkpoint metadata is saved as `best.json`

For more tuning guidance, see `PRODUCTION_TRAINING_GUIDE.md`.

## Evaluation

Evaluate a saved checkpoint:

```bash
python runner.py eval \
  --checkpoint checkpoints/efficientnet_b2_YYYYMMDD_HHMMSS/best.pt \
  --data_dir data/processed/kaggle_freshness_structured
```

Compare a fine-tuned model against a pretrained baseline:

```bash
python utils/eval_dataset.py \
  --data_dir data/processed/kaggle_freshness_structured \
  --checkpoint checkpoints/efficientnet_b2_YYYYMMDD_HHMMSS/best.pt \
  --baseline
```

## Inference

Run inference on random images from a directory:

```bash
python utils/inference.py \
  --checkpoint checkpoints/efficientnet_b2_YYYYMMDD_HHMMSS/best.pt \
  --image_dir data/processed/kaggle_freshness_structured \
  --num_random 5
```

Run inference on specific images:

```bash
python utils/inference.py \
  --checkpoint checkpoints/efficientnet_b2_YYYYMMDD_HHMMSS/best.pt \
  --image_paths path/to/image1.jpg path/to/image2.jpg
```

## CoreML Export

Convert one checkpoint to CoreML:

```bash
python utils/convert_coreML.py \
  --checkpoint checkpoints/efficientnet_b2_YYYYMMDD_HHMMSS/best.pt
```

Convert all checkpoints in the `checkpoints/` directory:

```bash
python utils/convert_coreML.py \
  --checkpoint_dir checkpoints \
  --convert_all \
  --output_dir coreml_models
```

## Notes

- `runner.py` automatically uses CUDA, Apple MPS, or CPU depending on availability.
- The training and evaluation scripts expect a PyTorch `ImageFolder`-compatible directory.
- Existing converted models and checkpoints in this repo can be used as references for naming and output structure.

### Copyright Cade Stocker 2026