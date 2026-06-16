# Copyright 2026 Cade Stocker

"""
Inference utility for testing freshness classification on individual images.

This script allows you to:
1. Run inference on random images from a directory
2. Run inference on specific images
3. Visualize predictions with confidence scores

Usage examples:

# Test on 5 random images from your data directory
python utils/inference.py --checkpoint checkpoints/mobilenet_v3_large_20260522_115901_best/best.pt --image_dir data/processed/kaggle_freshness_structured --num_random 5

# Test on specific images
python utils/inference.py --checkpoint checkpoints/mobilenet_v3_large_20260522_115901_best/best.pt --image_paths image1.jpg image2.jpg image3.jpg

# Test on all images in a directory
python utils/inference.py --checkpoint checkpoints/mobilenet_v3_large_20260522_115901_best/best.pt --image_dir /path/to/images --all
"""

import argparse
import json
from pathlib import Path
import random
import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Add parent directory to path so we can import models
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import pretrained_hub

def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint"""
    checkpoint_path = Path(checkpoint_path)
    
    # Try to find config file - it could be:
    # 1. Named after checkpoint (old format): mobilenet_v3_large_20260522_115901_best.json
    # 2. Named best.json (new format): best.json
    config_path = checkpoint_path.with_suffix(".json")
    if not config_path.exists():
        config_path = checkpoint_path.parent / "best.json"
    if not config_path.exists():
        raise ValueError(f"Config file not found. Checked: {checkpoint_path.with_suffix('.json')} and {checkpoint_path.parent / 'best.json'}")
    
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

def get_transform(image_size: int):
    """Get preprocessing transform"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

def load_image(image_path: str, transform) -> Tuple[torch.Tensor, Image.Image]:
    """Load and preprocess image"""
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image)
    return tensor, image

def run_inference(model, image_tensor: torch.Tensor, class_names: List[str], device: torch.device):
    """Run inference on a single image"""
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        predicted_class_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class_idx].item()
    
    return predicted_class_idx, confidence, probabilities

def get_random_images(image_dir: str, num_images: int = 5) -> List[Path]:
    """Get random image paths from directory"""
    image_dir = Path(image_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    
    image_paths = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    ]
    
    if len(image_paths) < num_images:
        print(f"Warning: Only {len(image_paths)} images found, requested {num_images}")
        return image_paths
    
    return random.sample(image_paths, num_images)

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run inference on images using a trained freshness classifier")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (best.pt)")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing images")
    parser.add_argument("--image_paths", type=str, nargs="+", default=None, help="Specific image paths to test")
    parser.add_argument("--num_random", type=int, default=5, help="Number of random images to test (if using --image_dir)")
    parser.add_argument("--all", action="store_true", help="Test on all images in directory (if using --image_dir)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args(argv)
    
    random.seed(args.seed)
    
    # Determine which device to use
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load model and config
    print(f"\nLoading model from: {args.checkpoint}")
    model, class_names, image_size = load_checkpoint(args.checkpoint, device)
    transform = get_transform(image_size)
    
    # Determine which images to test
    image_paths = []
    
    if args.image_paths:
        image_paths = [Path(p) for p in args.image_paths]
    elif args.image_dir:
        if args.all:
            image_dir = Path(args.image_dir)
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
            image_paths = sorted([
                p for p in image_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in image_extensions
            ])
        else:
            image_paths = get_random_images(args.image_dir, args.num_random)
    else:
        raise ValueError("Must provide either --image_dir or --image_paths")
    
    print(f"\nClass names: {class_names}")
    print(f"Testing on {len(image_paths)} images\n")
    print("-" * 80)
    
    # Run inference
    correct_predictions = 0
    all_results = []
    
    for image_path in image_paths:
        image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"⚠️  Image not found: {image_path}")
            continue
        
        try:
            # Load and preprocess
            image_tensor, pil_image = load_image(str(image_path), transform)
            
            # Run inference
            pred_idx, confidence, probabilities = run_inference(model, image_tensor, class_names, device)
            
            # Get all class probabilities
            probs_dict = {
                class_names[i]: f"{probabilities[i].item():.4f}"
                for i in range(len(class_names))
            }
            
            # Print results
            print(f"Image: {image_path.name}")
            print(f"  Path: {image_path}")
            print(f"  Prediction: {class_names[pred_idx]} (confidence: {confidence:.4f})")
            print(f"  All probabilities: {probs_dict}")
            
            # Check if prediction matches ground truth (if directory structure indicates it)
            # e.g., fresh/apple/image.jpg -> class is "fresh"
            relative_path = image_path.relative_to(image_path.parents[3]) if len(image_path.parents) >= 3 else None
            if relative_path and len(relative_path.parts) >= 1:
                ground_truth = relative_path.parts[0]
                if ground_truth in class_names:
                    is_correct = class_names[pred_idx] == ground_truth
                    correct_predictions += is_correct
                    status = "✓" if is_correct else "✗"
                    print(f"  Ground truth: {ground_truth} {status}")
            
            print()
            
            all_results.append({
                "image": str(image_path),
                "prediction": class_names[pred_idx],
                "confidence": float(confidence),
                "probabilities": probs_dict,
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}\n")
    
    print("-" * 80)
    if correct_predictions > 0 and len(all_results) > 0:
        accuracy = correct_predictions / len(all_results)
        print(f"\nAccuracy: {accuracy:.2%} ({correct_predictions}/{len(all_results)})")

if __name__ == "__main__":
    main()
