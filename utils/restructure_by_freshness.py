# Copyright 2026 Cade Stocker

"""
I got claude to write this to restructure the dataset. I was accidentally training models to predict type of produce
rather than freshness because of the dataset's structure.

Restructure the dataset from produce_type/freshness to freshness/produce_type layout.

This script reorganizes the data so that freshness (fresh/rotten) becomes the top-level 
classification category, allowing ImageFolder to treat those as the model's classes.

Current structure:
  apple/fresh/
  apple/rotten/
  banana/fresh/
  banana/rotten/
  ...

New structure:
  fresh/apple/
  fresh/banana/
  ...
  rotten/apple/
  rotten/banana/
  ...

Usage:
  python utils/restructure_by_freshness.py --input_dir data/processed/kaggle_augmented_may192026_1 --output_dir data/processed/kaggle_freshness_structured
"""

import argparse
from pathlib import Path
import shutil
from tqdm import tqdm

def restructure_dataset(input_dir, output_dir, use_symlinks=True):
    """
    Restructure dataset from produce_type/freshness to freshness/produce_type.
    
    Args:
        input_dir: Path to the input dataset (e.g., data/processed/kaggle_augmented_may192026_1)
        output_dir: Path where the restructured dataset will be created
        use_symlinks: If True, creates symlinks instead of copying files (saves disk space)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_path}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create fresh and rotten top-level directories
    fresh_dir = output_path / "fresh"
    rotten_dir = output_path / "rotten"
    fresh_dir.mkdir(exist_ok=True)
    rotten_dir.mkdir(exist_ok=True)
    
    # Iterate through each produce type
    produce_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    
    for produce_dir in tqdm(produce_dirs, desc="Restructuring dataset"):
        produce_name = produce_dir.name
        
        # Create subdirectories for this produce type in both fresh and rotten
        fresh_produce_dir = fresh_dir / produce_name
        rotten_produce_dir = rotten_dir / produce_name
        fresh_produce_dir.mkdir(exist_ok=True)
        rotten_produce_dir.mkdir(exist_ok=True)
        
        # Move fresh images
        fresh_src = produce_dir / "fresh"
        if fresh_src.exists():
            for image_file in tqdm(list(fresh_src.iterdir()), desc=f"  {produce_name}/fresh", leave=False):
                if image_file.is_file():
                    dest = fresh_produce_dir / image_file.name
                    if use_symlinks:
                        # Remove existing symlink if it exists
                        if dest.exists() or dest.is_symlink():
                            dest.unlink()
                        dest.symlink_to(image_file.resolve())
                    else:
                        shutil.copy2(image_file, dest)
        
        # Move rotten images
        rotten_src = produce_dir / "rotten"
        if rotten_src.exists():
            for image_file in tqdm(list(rotten_src.iterdir()), desc=f"  {produce_name}/rotten", leave=False):
                if image_file.is_file():
                    dest = rotten_produce_dir / image_file.name
                    if use_symlinks:
                        # Remove existing symlink if it exists
                        if dest.exists() or dest.is_symlink():
                            dest.unlink()
                        dest.symlink_to(image_file.resolve())
                    else:
                        shutil.copy2(image_file, dest)
    
    print(f"\n✓ Dataset restructured successfully!")
    print(f"  Output: {output_path}")
    print(f"  Structure: fresh/{produce_name}/, rotten/{produce_name}/")
    
    # Print summary
    fresh_count = sum(1 for _ in fresh_dir.rglob("*") if _.is_file())
    rotten_count = sum(1 for _ in rotten_dir.rglob("*") if _.is_file())
    print(f"  Fresh images: {fresh_count}")
    print(f"  Rotten images: {rotten_count}")
    print(f"  Total images: {fresh_count + rotten_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restructure dataset by freshness classification")
    parser.add_argument("--input_dir", type=str, required=True, help="Input dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for restructured dataset")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of using symlinks (uses more disk space)")
    args = parser.parse_args()
    
    use_symlinks = not args.copy
    restructure_dataset(args.input_dir, args.output_dir, use_symlinks=use_symlinks)
