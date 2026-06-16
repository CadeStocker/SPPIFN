# Copyright 2026 Cade Stocker

"""
Batch training script to train multiple models while you're away.

This script reads a configuration file and trains multiple models sequentially.
You can customize the models, hyperparameters, and datasets in the config file.

Usage:
    python scripts/train_batch.py --config scripts/training_config.json
    
    # Or with custom config:
    python scripts/train_batch.py --config my_custom_config.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def load_config(config_path: str):
    """Load training configuration from JSON file"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    return config

def run_training(model_config: dict, data_dir: str, base_args: dict):
    """Run a single training job"""
    model_name = model_config.get("model")
    epochs = model_config.get("epochs", base_args.get("epochs", 20))
    lr = model_config.get("lr", base_args.get("lr", 0.0007))
    batch_size = model_config.get("batch_size", base_args.get("batch_size", 32))
    num_workers = model_config.get("num_workers", base_args.get("num_workers", 5))
    freeze_backbone = model_config.get("freeze_backbone", base_args.get("freeze_backbone", True))
    
    run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    cmd = [
        "python", "runner.py", "train",
        "--data_dir", data_dir,
        "--model", model_name,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--num_workers", str(num_workers),
        "--run_name", run_name,
    ]
    
    if freeze_backbone:
        cmd.append("--freeze_backbone")
    else:
        cmd.append("--no-freeze_backbone")
    
    print(f"\n{'='*80}")
    print(f"Training: {model_name}")
    print(f"{'='*80}")
    print(f"Config:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Num workers: {num_workers}")
    print(f"  Freeze backbone: {freeze_backbone}")
    print(f"  Run name: {run_name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode != 0:
        print(f"⚠️  Training failed for {model_name} with return code {result.returncode}")
        return False
    else:
        print(f"✓ Training completed for {model_name}")
        return True

def main(argv=None):
    parser = argparse.ArgumentParser(description="Batch train multiple models")
    parser.add_argument("--config", type=str, default="scripts/training_config.json", 
                        help="Path to training configuration file")
    args = parser.parse_args(argv)
    
    config = load_config(args.config)
    
    data_dir = config.get("data_dir")
    if not data_dir:
        raise ValueError("data_dir must be specified in config file")
    
    base_args = config.get("base_args", {})
    models = config.get("models", [])
    
    if not models:
        raise ValueError("No models specified in config file")
    
    print(f"\n{'='*80}")
    print(f"BATCH TRAINING START")
    print(f"{'='*80}")
    print(f"Data directory: {data_dir}")
    print(f"Total models to train: {len(models)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    successful = 0
    failed = 0
    
    for i, model_config in enumerate(models, 1):
        print(f"[{i}/{len(models)}] Starting training job...")
        success = run_training(model_config, data_dir, base_args)
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"BATCH TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {successful}/{len(models)}")
    print(f"Failed: {failed}/{len(models)}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
