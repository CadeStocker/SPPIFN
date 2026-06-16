# Copyright 2026 Cade Stocker

"""
Tests for data loading and augmentation pipeline.
"""

import pytest
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

import sys
from pathlib import Path

# Add parent directory to path to import runner
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataPipeline:
    """Test data loading and augmentation."""

    def test_imagefolder_loading(self, sample_image_folder):
        """Test that ImageFolder correctly loads sample data."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        dataset = datasets.ImageFolder(root=str(sample_image_folder), transform=transform)
        
        # Should have 2 classes
        assert len(dataset.classes) == 2
        assert "fresh" in dataset.classes
        assert "rotten" in dataset.classes
        
        # Should have 4 images (2 per class)
        assert len(dataset) == 4
        
        # Images should have correct shape and type
        image, label = dataset[0]
        assert image.shape == (3, 224, 224)
        assert isinstance(label, int)
        assert label in [0, 1]

    def test_dataloader_batching(self, sample_image_folder):
        """Test that DataLoader batches correctly."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        dataset = datasets.ImageFolder(root=str(sample_image_folder), transform=transform)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Should have 2 batches (4 images / batch_size 2)
        batches = list(loader)
        assert len(batches) == 2
        
        # Each batch should have correct shape
        images, labels = batches[0]
        assert images.shape == (2, 3, 224, 224)
        assert labels.shape == (2,)

    def test_train_val_split(self, sample_image_folder):
        """Test that train/val split works correctly."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        dataset = datasets.ImageFolder(root=str(sample_image_folder), transform=transform)
        total_size = len(dataset)
        
        # Split 75/25 (more reasonable for small dataset)
        val_size = max(1, int(total_size * 0.25))  # At least 1 sample
        train_size = total_size - val_size
        
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=generator
        )
        
        # Sizes should sum to total
        assert len(train_dataset) + len(val_dataset) == total_size
        assert len(train_dataset) > 0, "Train set should not be empty"
        assert len(val_dataset) > 0, "Val set should not be empty"
        
        # Datasets should not overlap
        train_indices = set(train_dataset.indices)
        val_indices = set(val_dataset.indices)
        assert len(train_indices & val_indices) == 0, "Train/val split should not overlap"

    def test_augmentation_transform(self, sample_image_folder):
        """Test that augmentation transforms apply without errors."""
        # Advanced augmentation like in runner.py
        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        dataset = datasets.ImageFolder(root=str(sample_image_folder), transform=augment_transform)
        
        # Load a few samples - should not raise
        for i in range(min(3, len(dataset))):
            image, label = dataset[i]
            assert image.shape == (3, 224, 224)
            # Verify normalization roughly works (mean should be near 0, std near 1)
            assert image.mean().abs() < 2.0
            assert image.std() > 0.1

    def test_deterministic_split_with_seed(self, sample_image_folder):
        """Test that seeding produces deterministic splits."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        dataset = datasets.ImageFolder(root=str(sample_image_folder), transform=transform)
        
        # Create splits with same seed
        split1_indices = set()
        split2_indices = set()
        
        for seed in [42, 42]:
            val_size = int(len(dataset) * 0.2)
            train_size = len(dataset) - val_size
            
            generator = torch.Generator().manual_seed(seed)
            train_ds, val_ds = random_split(
                dataset,
                [train_size, val_size],
                generator=generator
            )
            
            if seed == 42 and not split1_indices:
                split1_indices = set(train_ds.indices)
            else:
                split2_indices = set(train_ds.indices)
        
        # Same seed should produce same split
        assert split1_indices == split2_indices, "Same seed should produce deterministic splits"
