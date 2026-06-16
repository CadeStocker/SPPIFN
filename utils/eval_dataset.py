# Copyright 2026 Cade Stocker

"""
Evaluate and compare pretrained vs fine-tuned models on the freshness dataset.

This script allows you to:
1. Evaluate a fine-tuned checkpoint on the dataset
2. Evaluate a baseline pretrained model on the dataset
3. Compare accuracy between the two

Usage examples:

# Evaluate just a fine-tuned checkpoint
python utils/eval_dataset.py --data_dir data/processed/kaggle_freshness_structured --checkpoint checkpoints/mobilenet_v3_large_20260522_115901_best/best.pt

# Evaluate fine-tuned vs pretrained baseline
python utils/eval_dataset.py --data_dir data/processed/kaggle_freshness_structured --checkpoint checkpoints/mobilenet_v3_large_20260522_115901_best/best.pt --baseline

# Evaluate just the baseline
python utils/eval_dataset.py --data_dir data/processed/kaggle_freshness_structured --model mobilenet_v3_large --baseline_only
"""

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import pretrained_hub

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")

def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint"""
    checkpoint_path = Path(checkpoint_path)
    
    # Try to find config file
    config_path = checkpoint_path.with_suffix(".json")
    if not config_path.exists():
        config_path = checkpoint_path.parent / "best.json"
    if not config_path.exists():
        raise ValueError(f"Config file not found")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    model_name = config["model_name"]
    num_classes = config["num_classes"]
    class_names = config["class_names"]
    image_size = config.get("image_size", 224)
    
    # Model factory map
    model_map = {
        "mobilenet_v3_large": pretrained_hub.get_mobilenet_v3_large,
        "efficientnet_b0": pretrained_hub.get_efficientnet_b0,
        "efficientnet_b2": pretrained_hub.get_efficientnet_b2,
        "resnet18": pretrained_hub.get_resnet18,
        "resnet34": pretrained_hub.get_resnet34,
        "convnext_tiny": pretrained_hub.get_convnext_tiny,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load model architecture
    model = model_map[model_name](num_classes=num_classes, freeze_backbone=False)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    
    return model, class_names, image_size

def load_pretrained_baseline(model_name: str, num_classes: int, device: torch.device):
    """Load a pretrained model without fine-tuning (baseline)"""
    model_map = {
        "mobilenet_v3_large": pretrained_hub.get_mobilenet_v3_large,
        "efficientnet_b0": pretrained_hub.get_efficientnet_b0,
        "efficientnet_b2": pretrained_hub.get_efficientnet_b2,
        "resnet18": pretrained_hub.get_resnet18,
        "resnet34": pretrained_hub.get_resnet34,
        "convnext_tiny": pretrained_hub.get_convnext_tiny,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load with pretrained weights but don't fine-tune
    model = model_map[model_name](num_classes=num_classes, freeze_backbone=False)
    model = model.to(device)
    model.eval()
    
    return model

def evaluate_model(model, loader, device, model_name: str):
    """Evaluate a model on the dataset"""
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Evaluating {model_name}", unit="batch", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Per-class accuracy
            for label in range(len(loader.dataset.classes)):
                mask = labels == label
                if mask.any():
                    class_correct[label] = class_correct.get(label, 0) + (predicted[mask] == label).sum().item()
                    class_total[label] = class_total.get(label, 0) + mask.sum().item()
    
    accuracy = correct / total if total > 0 else 0
    
    return accuracy, class_correct, class_total

def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate and compare models on freshness dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to freshness dataset")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to fine-tuned checkpoint")
    parser.add_argument("--baseline", action="store_true", help="Also evaluate pretrained baseline for comparison")
    parser.add_argument("--baseline_only", action="store_true", help="Only evaluate baseline (no checkpoint)")
    parser.add_argument("--model", type=str, default="mobilenet_v3_large", help="Model name for baseline")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args(argv)
    
    device = get_device()
    
    # Load dataset
    print(f"\nLoading dataset from: {args.data_dir}")
    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(root=args.data_dir, transform=eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    class_names = dataset.classes
    num_classes = len(class_names)
    
    print(f"Classes: {class_names}")
    print(f"Total images: {len(dataset)}")
    print()
    
    results = {}
    
    # Evaluate fine-tuned checkpoint if provided
    if args.checkpoint and not args.baseline_only:
        print(f"Loading fine-tuned checkpoint: {args.checkpoint}")
        model, _, image_size = load_checkpoint(args.checkpoint, device)
        accuracy, class_correct, class_total = evaluate_model(model, loader, device, "Fine-tuned model")
        
        results["fine_tuned"] = {
            "accuracy": accuracy,
            "class_accuracies": {}
        }
        
        for i, class_name in enumerate(class_names):
            class_acc = class_correct.get(i, 0) / max(class_total.get(i, 1), 1)
            results["fine_tuned"]["class_accuracies"][class_name] = class_acc
            print(f"  {class_name}: {class_acc:.4f} ({class_correct.get(i, 0)}/{class_total.get(i, 0)})")
        
        print(f"\nFine-tuned model accuracy: {accuracy:.4f}\n")
    
    # Evaluate baseline pretrained model
    if args.baseline or args.baseline_only:
        print(f"Loading pretrained baseline: {args.model}")
        model = load_pretrained_baseline(args.model, num_classes, device)
        accuracy, class_correct, class_total = evaluate_model(model, loader, device, "Pretrained baseline")
        
        results["baseline"] = {
            "accuracy": accuracy,
            "class_accuracies": {}
        }
        
        for i, class_name in enumerate(class_names):
            class_acc = class_correct.get(i, 0) / max(class_total.get(i, 1), 1)
            results["baseline"]["class_accuracies"][class_name] = class_acc
            print(f"  {class_name}: {class_acc:.4f} ({class_correct.get(i, 0)}/{class_total.get(i, 0)})")
        
        print(f"\nBaseline accuracy: {accuracy:.4f}\n")
    
    # Comparison
    if "fine_tuned" in results and "baseline" in results:
        improvement = results["fine_tuned"]["accuracy"] - results["baseline"]["accuracy"]
        improvement_pct = (improvement / results["baseline"]["accuracy"]) * 100 if results["baseline"]["accuracy"] > 0 else 0
        print("=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print(f"Baseline accuracy:    {results['baseline']['accuracy']:.4f}")
        print(f"Fine-tuned accuracy:  {results['fine_tuned']['accuracy']:.4f}")
        print(f"Improvement:          {improvement:+.4f} ({improvement_pct:+.1f}%)")
        print("=" * 60)

if __name__ == "__main__":
    main()
