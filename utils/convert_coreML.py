# Copyright 2026 Cade Stocker

"""
Convert PyTorch model checkpoints to CoreML format for deployment on iOS/iPad.

Usage:
    python convert_coreML.py --checkpoint path/to/model.pt --output_dir path/to/output
    python convert_coreML.py --checkpoint_dir checkpoints/ --convert_all
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path so we can import models
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import coremltools as ct
from coremltools.models.neural_network import quantization_utils

from models import pretrained_hub


def _build_conversion_dir(output_dir, timestamp=None):
    """Build a unique output directory for a conversion run."""
    run_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / run_timestamp


def _build_output_filename(checkpoint_path):
    """Build a descriptive CoreML filename from a checkpoint path."""
    checkpoint_path = Path(checkpoint_path)
    parent_name = checkpoint_path.parent.name

    if checkpoint_path.stem == "best" and parent_name:
        return f"{parent_name}.mlmodel"

    if parent_name and parent_name not in {".", ".."}:
        return f"{parent_name}_{checkpoint_path.stem}.mlmodel"

    return checkpoint_path.stem + ".mlmodel"


def _get_device():
    """Get the best available device (GPU if available, otherwise CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _create_traced_model(checkpoint_path, device):
    """
    Load PyTorch model and prepare for conversion.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: torch device to use
        
    Returns:
        model: Loaded and traced PyTorch model
        input_shape: Expected input shape (for tracing)
        config: Model configuration from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config
    model_name = checkpoint.get("model_name")
    num_classes = checkpoint.get("num_classes")
    config = {
        "model_name": model_name,
        "num_classes": num_classes,
        "image_size": checkpoint.get("image_size", 224),
    }
    
    # Load model
    model = pretrained_hub.get_model(model_name, num_classes, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    return model, config


def convert_to_coreml(
    checkpoint_path,
    output_path,
    image_size=224,
    quantize=True,
    mlmodel_input_name="image",
):
    """
    Convert a PyTorch checkpoint to CoreML format.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        output_path: Path where .mlmodel will be saved
        image_size: Input image size (assumes square images)
        quantize: Whether to apply quantization (reduces model size)
        mlmodel_input_name: Name of the input in the CoreML model
        
    Returns:
        output_path: Path to the saved CoreML model
    """
    device = _get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading checkpoint from {checkpoint_path}...")
    model, config = _create_traced_model(checkpoint_path, device)
    
    # Create dummy input for tracing
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    
    # Trace the model
    print("Tracing model...")
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Convert to CoreML
    print("Converting to CoreML...")
    ml_model = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name=mlmodel_input_name,
                shape=(1, 3, image_size, image_size),
                scale=1.0 / 255.0,  # Normalize pixel values from [0, 255] to [0, 1]
                bias=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],  # ImageNet normalization
            )
        ],
        outputs=[ct.TensorType(name="logits")],  # Shape is automatically inferred
        compute_units=ct.ComputeUnit.CPU_AND_NE,  # Use Neural Engine on supported devices
        convert_to="neuralnetwork",  # Use neuralnetwork format for .mlmodel (more compatible)
    )
    
    # Add model metadata
    ml_model.author = "Cade Stocker"
    ml_model.short_description = f"Fruit freshness classifier: {config['model_name']}"
    ml_model.input_description[mlmodel_input_name] = "Input image (RGB, normalized to [0, 1])"
    ml_model.output_description["logits"] = f"Classification logits ({config['num_classes']} classes)"
    
    # Optional: Quantize to reduce model size
    if quantize:
        print("Applying quantization...")
        ml_model = quantization_utils.quantize_weights(ml_model, nbits=8)
    
    # Save model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ml_model.save(str(output_path))
    print(f"✓ CoreML model saved to {output_path}")
    
    # Save conversion metadata
    metadata_path = output_path.with_suffix(".json")
    metadata = {
        "original_checkpoint": str(checkpoint_path),
        "model_name": config["model_name"],
        "num_classes": config["num_classes"],
        "image_size": image_size,
        "quantized": quantize,
        "input_name": mlmodel_input_name,
        "output_name": "logits",
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_path}")
    
    return output_path


def convert_directory(
    checkpoint_dir,
    output_dir,
    image_size=224,
    quantize=True,
    pattern="**/*.pt",
    timestamp=None,
):
    """
    Convert all checkpoints in a directory to CoreML.
    
    Args:
        checkpoint_dir: Directory containing .pt files
        output_dir: Directory where .mlmodel files will be saved
        image_size: Input image size
        quantize: Whether to apply quantization
        pattern: Glob pattern for finding checkpoints
        timestamp: Optional timestamp override for deterministic output paths
    """
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = _build_conversion_dir(output_dir, timestamp=timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints = list(checkpoint_dir.glob(pattern))
    if not checkpoints:
        print(f"No checkpoints found matching pattern '{pattern}' in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoints)} checkpoint(s) to convert")
    
    for i, checkpoint_path in enumerate(checkpoints, 1):
        try:
            print(f"\n[{i}/{len(checkpoints)}] Converting {checkpoint_path.name}...")
            
            # Generate output filename
            output_filename = _build_output_filename(checkpoint_path)
            output_path = output_dir / output_filename
            
            convert_to_coreml(
                checkpoint_path,
                output_path,
                image_size=image_size,
                quantize=quantize,
            )
        except Exception as e:
            print(f"✗ Failed to convert {checkpoint_path.name}: {e}")
            import traceback
            traceback.print_exc()


def main(argv=None):
    """Main entry point for the conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert PyTorch model checkpoints to CoreML format"
    )
    
    # Single checkpoint conversion
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a single .pt checkpoint file to convert",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for the .mlmodel file",
    )
    
    # Batch conversion
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Directory containing checkpoint files to convert",
    )
    parser.add_argument(
        "--convert_all",
        action="store_true",
        help="Convert all .pt files in checkpoint_dir",
    )
    
    # Conversion options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="coreml_models",
        help="Output directory for CoreML models (default: coreml_models)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input image size (default: 224)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=True,
        help="Apply 8-bit quantization to reduce model size (default: True)",
    )
    parser.add_argument(
        "--no_quantize",
        action="store_true",
        help="Disable quantization",
    )
    
    args = parser.parse_args(argv)
    
    # Determine quantization setting
    quantize = not args.no_quantize if args.no_quantize else True
    
    # Single checkpoint conversion
    if args.checkpoint:
        if not args.output:
            checkpoint_path = Path(args.checkpoint)
            output_dir = _build_conversion_dir(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            args.output = output_dir / _build_output_filename(checkpoint_path)
        
        print(f"Converting single checkpoint: {args.checkpoint}")
        convert_to_coreml(
            args.checkpoint,
            args.output,
            image_size=args.image_size,
            quantize=quantize,
        )
    
    # Batch conversion
    elif args.checkpoint_dir and args.convert_all:
        print(f"Converting all checkpoints from: {args.checkpoint_dir}")
        convert_directory(
            args.checkpoint_dir,
            args.output_dir,
            image_size=args.image_size,
            quantize=quantize,
        )
    
    else:
        parser.print_help()
        print("\n" + "=" * 60)
        print("EXAMPLES:")
        print("=" * 60)
        print("\nConvert a single checkpoint:")
        print("  python utils/convert_coreML.py \\")
        print("    --checkpoint checkpoints/efficientnet_b2_20260528_160526/best.pt \\")
        print("    --output coreml_models/20260609_120000/efficientnet_b2_20260528_160526.mlmodel")
        print("\nConvert all checkpoints in a directory:")
        print("  python utils/convert_coreML.py \\")
        print("    --checkpoint_dir checkpoints \\")
        print("    --convert_all \\")
        print("    --output_dir coreml_models")
        print("\nDisable quantization:")
        print("  python utils/convert_coreML.py \\")
        print("    --checkpoint checkpoints/efficientnet_b2_20260528_160526/best.pt \\")
        print("    --output coreml_models/20260609_120000/efficientnet_b2_20260528_160526.mlmodel \\")
        print("    --no_quantize")


if __name__ == "__main__":
    main()