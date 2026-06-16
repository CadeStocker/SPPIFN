# Copyright 2026 Cade Stocker

"""
Data augmentation functions for the SPPIFN project.

This file can be run on a dataset (directory) to perform data augmentation and save the augmented images to a new directory (which can be specified)

Augmented data should have same directory structure as original data
Types of augmentations to perform:
- horizontal flip
- vertical flip
- rotation
- brightness adjustment
- contrast adjustment
- random crop and resize

Input looks like:

python data_augment.py --input_dir /path/to/original/images --output_dir /path/to/augmented/images --multiplier 3 --seed 42 --save_original True --workers 4

Used GPT-5.2-Codex to refactor and include parallel processing. (significantly faster)
"""

import argparse
from pathlib import Path
import random
import os
from concurrent.futures import ProcessPoolExecutor
import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm


# global vars for process workers
_PIPELINE = None
_INPUT_ROOT = None
_OUTPUT_DIR = None
_MULTIPLIER = None
_SAVE_ORIGINAL = None

def get_augmentation_pipeline():

    """
    Function to define the augmentation pipeline using albumentations library.

    compose uses probabilities to determine which augmentations to apply to each image,
    so we can have some randomness in the augmentations applied to each image
    (e.g., some images might only be flipped, while others might be flipped and rotated, etc.)
    """

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.Perspective(scale=(0.05, 0.1), p=0.5)
    ], bbox_params=None)

def _init_worker(input_root, output_dir, multiplier, save_original, seed):

    global _PIPELINE, _INPUT_ROOT, _OUTPUT_DIR, _MULTIPLIER, _SAVE_ORIGINAL

    _INPUT_ROOT = input_root
    _OUTPUT_DIR = output_dir
    _MULTIPLIER = multiplier
    _SAVE_ORIGINAL = save_original

    # keep per-process randomness independent
    if seed is not None:
        worker_seed = seed + (os.getpid() % 100000)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    # avoid OpenCV internal threading per process
    cv2.setNumThreads(0)

    _PIPELINE = get_augmentation_pipeline()

def _augment_and_save_worker(image_path):

    """
    Apply augmentations and save copies
    """

    # read image (BGR) and create RGB copy for albumentations
    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # create parent directories in output dir
    relative_path = image_path.relative_to(_INPUT_ROOT)
    output_image_dir = Path(_OUTPUT_DIR) / relative_path.parent
    output_image_dir.mkdir(parents=True, exist_ok=True)

    # save original image to output dir
    if _SAVE_ORIGINAL:
        output_path = output_image_dir / relative_path.name
        cv2.imwrite(str(output_path), image_bgr)

    # generate augmented copies
    for i in range(_MULTIPLIER):
        augmented = _PIPELINE(image=image_rgb)
        augmented_image = augmented["image"]

        # save with suffix
        stem = relative_path.stem
        suffix = relative_path.suffix
        augmented_name = f"{stem}_aug_{i + 1}{suffix}"
        augmented_path = output_image_dir / augmented_name

        augmented_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(augmented_path), augmented_bgr)

if __name__ == "__main__":

    """
    User can pick which augmentations/how many augmentations to perform on each image (e.g., perform 3 random augmentations on each image)

    User's input will look like:
    python data_augment.py --input_dir /path/to/original/images --output_dir /path/to/augmented/images --multiplier 3
    """

    # define command line arguments
    parser = argparse.ArgumentParser(description="Data augmentation for SPPIFN project")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing original images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save augmented images")
    parser.add_argument("--multiplier", type=int, default=2, help="Number of augmented copies of each image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--save_original",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the original image in the output directory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes to use (0 for single-process)",
    )

    args = parser.parse_args()

    # set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # get augmentation pipeline
    pipeline = get_augmentation_pipeline()

    input_path = Path(args.input_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    print(f"Starting data augmentation with multiplier {args.multiplier}...")
    print(f"="*50)

    # process all images, making sure to maintain directory structure in output dir
    image_paths = [
        image_path
        for image_path in Path(args.input_dir).rglob("*")
        if image_path.suffix.lower() in image_extensions
    ]
    if args.workers and args.workers > 0:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(input_path, args.output_dir, args.multiplier, args.save_original, args.seed),
        ) as executor:
            for _ in tqdm(
                executor.map(_augment_and_save_worker, image_paths, chunksize=8),
                total=len(image_paths),
                desc="Augmenting images",
                unit="image",
            ):
                pass
    else:
        _init_worker(input_path, args.output_dir, args.multiplier, args.save_original, args.seed)
        for image_path in tqdm(image_paths, desc="Augmenting images", unit="image"):
            _augment_and_save_worker(image_path)

    print("Data augmentation completed successfully!")
    print(f"="*50)