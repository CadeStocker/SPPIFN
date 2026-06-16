# Copyright 2026 Cade Stocker

"""
Pytest fixtures and configuration for SPPIFN tests.
"""

import pytest
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_checkpoint(temp_dir):
    """Create a minimal sample checkpoint for testing."""
    checkpoint = {
        "model_name": "mobilenet_v3_large",
        "num_classes": 2,
        "image_size": 224,
        "model_state": {
            "layer.weight": torch.randn(10, 5),
            "layer.bias": torch.randn(10),
        },
        "optimizer_state": None,
        "epoch": 1,
    }
    checkpoint_path = temp_dir / "sample_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path, checkpoint


@pytest.fixture
def sample_image_folder(temp_dir):
    """Create a minimal ImageFolder dataset with 2 classes and 4 images."""
    # Create directory structure
    fresh_dir = temp_dir / "fresh"
    rotten_dir = temp_dir / "rotten"
    fresh_dir.mkdir()
    rotten_dir.mkdir()
    
    # Create dummy images
    for i in range(2):
        # Fresh images
        img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
        img.save(fresh_dir / f"image_{i}.jpg")
        
        # Rotten images
        img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
        img.save(rotten_dir / f"image_{i}.jpg")
    
    return temp_dir


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
