# Copyright 2026 Cade Stocker

"""
Tests for CoreML conversion.
"""

import pytest
import torch
import tempfile
from pathlib import Path

import coremltools as ct

from models import pretrained_hub
from utils.convert_coreML import (
    _build_conversion_dir,
    _build_output_filename,
    _create_traced_model,
    convert_to_coreml,
)


class TestCoreMLConversion:
    """Test CoreML conversion functionality."""

    def test_build_conversion_dir_uses_timestamp(self, temp_dir):
        """Default conversion runs should land in a timestamped directory."""
        output_dir = _build_conversion_dir(temp_dir / "coreml_models", timestamp="20260609_120000")

        assert output_dir == temp_dir / "coreml_models" / "20260609_120000"

    def test_build_output_filename_uses_checkpoint_parent_for_best(self):
        """best.pt exports should keep the checkpoint folder identity to avoid collisions."""
        checkpoint_path = Path("checkpoints/efficientnet_b2_20260528_160526/best.pt")

        assert _build_output_filename(checkpoint_path) == "efficientnet_b2_20260528_160526.mlmodel"

    def test_build_output_filename_keeps_stem_for_named_checkpoint(self):
        """Named checkpoints should remain descriptive inside timestamped output directories."""
        checkpoint_path = Path("checkpoints/experiment42/epoch_12.pt")

        assert _build_output_filename(checkpoint_path) == "experiment42_epoch_12.mlmodel"

    def test_create_traced_model(self, temp_dir):
        """Test that we can load and trace a model from checkpoint."""
        # Create a simple checkpoint
        model = pretrained_hub.get_model("mobilenet_v3_large", num_classes=2, freeze_backbone=False)
        checkpoint = {
            "model_name": "mobilenet_v3_large",
            "num_classes": 2,
            "image_size": 224,
            "model_state": model.state_dict(),
        }
        
        checkpoint_path = temp_dir / "model.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Load using _create_traced_model
        device = torch.device("cpu")
        model, config = _create_traced_model(checkpoint_path, device)
        
        assert config["model_name"] == "mobilenet_v3_large"
        assert config["num_classes"] == 2
        assert config["image_size"] == 224
        assert isinstance(model, torch.nn.Module)

    def test_convert_to_coreml_creates_file(self, temp_dir):
        """Test that conversion creates a valid CoreML file."""
        # Create checkpoint
        model = pretrained_hub.get_model("mobilenet_v3_large", num_classes=2, freeze_backbone=False)
        checkpoint = {
            "model_name": "mobilenet_v3_large",
            "num_classes": 2,
            "image_size": 224,
            "model_state": model.state_dict(),
        }
        
        checkpoint_path = temp_dir / "model.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Convert to CoreML
        output_path = temp_dir / "model.mlmodel"
        result = convert_to_coreml(
            checkpoint_path,
            output_path,
            image_size=224,
            quantize=False,
        )
        
        # Check file exists
        assert output_path.exists(), "CoreML model file was not created"
        assert result == output_path
        
        # Check metadata file
        metadata_path = output_path.with_suffix(".json")
        assert metadata_path.exists(), "Metadata JSON file was not created"

    def test_convert_to_coreml_output_valid(self, temp_dir):
        """Test that converted CoreML model can be loaded."""
        # Create checkpoint
        model = pretrained_hub.get_model("mobilenet_v3_large", num_classes=2, freeze_backbone=False)
        checkpoint = {
            "model_name": "mobilenet_v3_large",
            "num_classes": 2,
            "image_size": 224,
            "model_state": model.state_dict(),
        }
        
        checkpoint_path = temp_dir / "model.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Convert
        output_path = temp_dir / "model.mlmodel"
        convert_to_coreml(
            checkpoint_path,
            output_path,
            image_size=224,
            quantize=False,
        )
        
        # Load the CoreML model
        coreml_model = ct.models.MLModel(str(output_path))
        
        # Check input/output specs
        assert len(coreml_model.input_description) > 0
        assert len(coreml_model.output_description) > 0

    def test_convert_different_architectures(self, temp_dir):
        """Test conversion of different model architectures."""
        models_to_test = [
            ("mobilenet_v3_large", 2),
            ("efficientnet_b0", 2),
            ("resnet18", 2),
        ]
        
        for model_name, num_classes in models_to_test:
            # Create checkpoint
            model = pretrained_hub.get_model(model_name, num_classes, freeze_backbone=False)
            checkpoint = {
                "model_name": model_name,
                "num_classes": num_classes,
                "image_size": 224,
                "model_state": model.state_dict(),
            }
            
            checkpoint_path = temp_dir / f"{model_name}.pt"
            torch.save(checkpoint, checkpoint_path)
            
            # Convert
            output_path = temp_dir / f"{model_name}.mlmodel"
            try:
                convert_to_coreml(
                    checkpoint_path,
                    output_path,
                    image_size=224,
                    quantize=False,
                )
                assert output_path.exists(), f"Failed to convert {model_name}"
            except Exception as e:
                pytest.fail(f"Failed to convert {model_name}: {e}")

    def test_convert_with_quantization(self, temp_dir):
        """Test conversion with quantization enabled."""
        # Create checkpoint
        model = pretrained_hub.get_model("mobilenet_v3_large", num_classes=2, freeze_backbone=False)
        checkpoint = {
            "model_name": "mobilenet_v3_large",
            "num_classes": 2,
            "image_size": 224,
            "model_state": model.state_dict(),
        }
        
        checkpoint_path = temp_dir / "model.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Convert with quantization
        output_path = temp_dir / "model_quantized.mlmodel"
        convert_to_coreml(
            checkpoint_path,
            output_path,
            image_size=224,
            quantize=True,
        )
        
        assert output_path.exists()
        
        # Quantized model should be smaller
        quantized_size = output_path.stat().st_size
        
        # Also create unquantized version for comparison
        output_path_unquantized = temp_dir / "model_unquantized.mlmodel"
        convert_to_coreml(
            checkpoint_path,
            output_path_unquantized,
            image_size=224,
            quantize=False,
        )
        
        unquantized_size = output_path_unquantized.stat().st_size
        
        # Quantized should be smaller (usually 25-30% of original)
        assert quantized_size < unquantized_size, \
            f"Quantized model ({quantized_size}) should be smaller than unquantized ({unquantized_size})"
