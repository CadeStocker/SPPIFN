# Copyright 2026 Cade Stocker

"""
Tests for checkpoint save/load functionality.
"""

import pytest
import torch
import json
from pathlib import Path

from models import pretrained_hub


class TestCheckpointSaveLoad:
    """Test checkpoint save and load operations."""

    def test_checkpoint_save_and_load(self, temp_dir):
        """Test that we can save and load a checkpoint without data loss."""
        # Create and save a simple checkpoint
        model = pretrained_hub.get_model("mobilenet_v3_large", num_classes=2, freeze_backbone=False)
        checkpoint = {
            "model_name": "mobilenet_v3_large",
            "num_classes": 2,
            "image_size": 224,
            "epoch": 5,
            "model_state": model.state_dict(),
        }
        
        checkpoint_path = temp_dir / "test_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Load and verify
        assert checkpoint_path.exists(), "Checkpoint file was not created"
        loaded = torch.load(checkpoint_path, map_location="cpu")
        
        assert loaded["model_name"] == "mobilenet_v3_large"
        assert loaded["num_classes"] == 2
        assert loaded["epoch"] == 5
        assert "model_state" in loaded

    def test_load_checkpoint_into_model(self, temp_dir):
        """Test that we can load checkpoint state into a model."""
        # Create a model and save its state
        model1 = pretrained_hub.get_model("mobilenet_v3_large", num_classes=2, freeze_backbone=False)
        original_state = model1.state_dict()
        
        checkpoint = {
            "model_name": "mobilenet_v3_large",
            "num_classes": 2,
            "model_state": original_state,
        }
        
        checkpoint_path = temp_dir / "model_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Load the checkpoint into a new model
        checkpoint_loaded = torch.load(checkpoint_path, map_location="cpu")
        model2 = pretrained_hub.get_model(
            checkpoint_loaded["model_name"],
            checkpoint_loaded["num_classes"],
            freeze_backbone=False
        )
        model2.load_state_dict(checkpoint_loaded["model_state"])
        
        # Verify they produce the same output
        model1.eval()
        model2.eval()
        dummy_input = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            output1 = model1(dummy_input)
            output2 = model2(dummy_input)
        
        assert torch.allclose(output1, output2, atol=1e-5), \
            "Models with same state should produce identical outputs"

    def test_checkpoint_metadata(self, sample_checkpoint):
        """Test that checkpoint contains required metadata."""
        checkpoint_path, checkpoint = sample_checkpoint
        
        loaded = torch.load(checkpoint_path, map_location="cpu")
        
        required_keys = ["model_name", "num_classes", "image_size"]
        for key in required_keys:
            assert key in loaded, f"Checkpoint missing required key: {key}"

    def test_multiple_models_checkpoints(self, temp_dir):
        """Test saving and loading different model architectures."""
        models_to_test = [
            ("mobilenet_v3_large", 2),
            ("efficientnet_b2", 5),
            ("resnet18", 3),
        ]
        
        for model_name, num_classes in models_to_test:
            model = pretrained_hub.get_model(model_name, num_classes, freeze_backbone=False)
            
            checkpoint = {
                "model_name": model_name,
                "num_classes": num_classes,
                "model_state": model.state_dict(),
            }
            
            checkpoint_path = temp_dir / f"{model_name}_checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)
            
            # Load back
            loaded = torch.load(checkpoint_path, map_location="cpu")
            assert loaded["model_name"] == model_name
            assert loaded["num_classes"] == num_classes
            
            # Can instantiate model with these configs
            reloaded_model = pretrained_hub.get_model(
                loaded["model_name"],
                loaded["num_classes"],
                freeze_backbone=False
            )
            assert reloaded_model is not None
