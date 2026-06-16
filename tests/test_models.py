# Copyright 2026 Cade Stocker

"""
Tests for model factory functions (pretrained_hub).
"""

import pytest
import torch

from models import pretrained_hub


class TestModelFactory:
    """Test the get_model factory function."""

    @pytest.mark.parametrize("model_name,num_classes", [
        ("mobilenet_v3_large", 2),
        ("efficientnet_b0", 5),
        ("efficientnet_b2", 2),
        ("resnet18", 3),
        ("resnet34", 2),
        ("convnext_tiny", 2),
    ])
    def test_get_model_creates_valid_model(self, model_name, num_classes):
        """Test that get_model creates a valid model with correct output size."""
        model = pretrained_hub.get_model(model_name, num_classes, freeze_backbone=True)
        
        # Model should be a nn.Module
        assert isinstance(model, torch.nn.Module)
        
        # Model should be evaluable
        model.eval()
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        # Output shape should match num_classes
        assert output.shape == (1, num_classes), \
            f"Expected output shape (1, {num_classes}), got {output.shape}"

    def test_get_model_freeze_backbone(self):
        """Test that freeze_backbone parameter works."""
        model_frozen = pretrained_hub.get_model("mobilenet_v3_large", 2, freeze_backbone=True)
        model_unfrozen = pretrained_hub.get_model("mobilenet_v3_large", 2, freeze_backbone=False)
        
        # Get parameter counts
        frozen_trainable = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
        unfrozen_trainable = sum(p.numel() for p in model_unfrozen.parameters() if p.requires_grad)
        
        # Frozen model should have fewer trainable parameters
        assert frozen_trainable < unfrozen_trainable, \
            "Frozen model should have fewer trainable parameters than unfrozen"

    def test_get_model_invalid_name(self):
        """Test that invalid model names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            pretrained_hub.get_model("invalid_model_name", 2)

    def test_individual_get_functions(self):
        """Test that individual get_* functions work."""
        models_to_test = [
            (pretrained_hub.get_mobilenet_v3_large, "mobilenet"),
            (pretrained_hub.get_efficientnet_b0, "efficientnet_b0"),
            (pretrained_hub.get_efficientnet_b2, "efficientnet_b2"),
            (pretrained_hub.get_resnet18, "resnet18"),
            (pretrained_hub.get_resnet34, "resnet34"),
            (pretrained_hub.get_convnext_tiny, "convnext"),
        ]
        
        for get_fn, model_name in models_to_test:
            model = get_fn(num_classes=2, freeze_backbone=True)
            assert isinstance(model, torch.nn.Module), \
                f"{model_name} function did not return a nn.Module"
            
            # Quick forward pass test
            model.eval()
            with torch.no_grad():
                output = model(torch.randn(1, 3, 224, 224))
            assert output.shape == (1, 2)
