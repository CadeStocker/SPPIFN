# Test Suite for SPPIFN

This directory contains pytest tests for the SPPIFN (Smart Produce Postharvest Integrity & Freshness Network) project.

## Running Tests

Install pytest first:
```bash
pip install pytest
```

Run all tests:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/test_models.py
```

Run with verbose output:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pip install pytest-cov
pytest tests/ --cov=. --cov-report=html
```

## Test Structure

### `conftest.py`
Shared pytest fixtures used across all tests:
- `temp_dir` - Temporary directory for test artifacts
- `sample_checkpoint` - Pre-made checkpoint file for testing
- `sample_image_folder` - Minimal ImageFolder dataset
- `device` - Appropriate device (CPU/GPU/MPS)
- `set_seed` - Automatic seed setting for reproducibility

### `test_models.py`
Tests for model factory functions (`pretrained_hub.py`):
- ✓ Model creation and forward passes
- ✓ Freeze/unfreeze backbone
- ✓ All supported architectures
- ✓ Invalid model name handling

### `test_checkpoint.py`
Tests for checkpoint save/load:
- ✓ Save and load checkpoints
- ✓ Load checkpoint state into model
- ✓ Checkpoint metadata validation
- ✓ Multiple architecture checkpoint handling

### `test_coreml_conversion.py`
Tests for CoreML model conversion:
- ✓ Model loading from checkpoint
- ✓ CoreML file creation
- ✓ Model validity
- ✓ Different architectures
- ✓ Quantization reduces model size

### `test_data_pipeline.py`
Tests for data loading and augmentation:
- ✓ ImageFolder loading
- ✓ DataLoader batching
- ✓ Train/val splits
- ✓ Augmentation transforms
- ✓ Deterministic splits with seeding

## Test Coverage

The test suite covers critical paths:
- **Model I/O**: Loading checkpoints, creating models
- **Data Pipeline**: Loading, splitting, augmenting images
- **Conversion**: Converting PyTorch → CoreML
- **Robustness**: Error handling, invalid inputs

## Quick Statistics

- **Total Tests**: ~30
- **Estimated Runtime**: ~2-5 minutes (CoreML conversion is slower)
- **Dependencies**: torch, torchvision, coremltools, pytest

## Notes

- Tests use CPU by default for speed, but will use GPU/MPS if available
- CoreML conversion tests may take 30-60 seconds due to model conversion overhead
- All tests seed random generators for deterministic results
- Sample data is generated on-the-fly (not included in repo)
