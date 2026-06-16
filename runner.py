# Copyright 2026 Cade Stocker

"""
This file will be responsible for running training and eval of models.

command for training:
python runner.py train --data_dir /path/to/data --model mobilenet_v3_large --epochs 5 --batch_size 32 --lr 1e-3 --image_size 224 --num_workers 4 --val_split 0.2 --seed 42 --freeze_backbone True

command for eval:
python runner.py eval --data_dir /path/to/data --checkpoint /path/to/checkpoint.pt --batch_size 32 --image_size 224 --num_workers 4
"""

import argparse
import csv
import json
from pathlib import Path
import random
import time
import resource
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import v2
from tqdm import tqdm
from models import pretrained_hub
import numpy as np


def _parse_bool(value):

    """
    Parse boolean CLI values while still allowing flag-only usage.
    """

    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _add_bool_argument(parser, name, default, help_text):

    """
    Add a boolean option that supports both '--flag False' and '--no-flag'.
    """

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        f"--{name}",
        nargs="?",
        const=True,
        default=default,
        type=_parse_bool,
        help=help_text,
    )
    group.add_argument(
        f"--no-{name}",
        dest=name,
        action="store_false",
        help=f"Disable {help_text.lower()}",
    )

def _set_seed(seed):

    """
    Set random seeds for reproducibility. This is important for debugging and comparing results across runs.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _get_device():

    """
    get the best available device (GPU if available, otherwise CPU). This allows us to leverage hardware acceleration when possible.
    """

    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")

def _build_dataloaders(data_dir, image_size, batch_size, num_workers, val_split, seed, pin_memory, use_advanced_aug=True):

    """
    Builds PyTorch DataLoaders for training and validation datasets.
    Advanced augmentations include Mixup, CutMix, and RandAugment for better regularization.
    """

    if use_advanced_aug:
        # Advanced augmentation strategy for production models
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
        eval_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
    else:
        # Original simple augmentation
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
        ])
        eval_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    num_classes = len(full_dataset.classes)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # override transforms for validation subset
    val_dataset.dataset = datasets.ImageFolder(root=data_dir, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, num_classes



def _run_epoch(model, loader, criterion, optimizer, device, train_mode, desc, mixup=None, cutmix=None, scheduler=None):

    """
    runs one epoch of training.
    If train_mode is True, will perform backprop and update model weights.
    Otherwise, just runs inference to compute loss and accuracy.
    Supports Mixup and CutMix augmentations during training.
    """

    if train_mode:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=desc, unit="batch", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        # Apply Mixup or CutMix augmentation
        use_mixup_loss = False
        if train_mode and (mixup is not None or cutmix is not None):
            if cutmix is not None and np.random.rand() < 0.5:
                images, labels_a, labels_b, lam = cutmix(images, labels)
                use_mixup_loss = True
            elif mixup is not None:
                images, labels_a, labels_b, lam = mixup(images, labels)
                use_mixup_loss = True

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            outputs = model(images)
            
            if use_mixup_loss:
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)

            if train_mode:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy

def _get_resource_snapshot():

    """
    gets a snapshot of current resource usage (CPU time and memory). This can be used to log performance metrics during training and evaluation.
    """

    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is KB on macOS
    return {
        "cpu_user_s": usage.ru_utime,
        "cpu_system_s": usage.ru_stime,
        "max_rss_kb": usage.ru_maxrss,
    }

class Mixup:
    """Mixup augmentation - combines two random images with interpolated labels."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch, target):
        images, targets = batch, target
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)
        y_a, y_b = targets, targets[index]
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        
        mixed_images = lam * images + (1 - lam) * images[index, :]
        return mixed_images, y_a, y_b, lam

class CutMix:
    """CutMix augmentation - randomly cuts and pastes patches from another image."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch, target):
        images, targets = batch, target
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)
        y_a, y_b = targets, targets[index]
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        
        # Get random box
        W = images.size(2)
        H = images.size(3)
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        
        cx = np.random.randint(0, W)
        cy = np.random.randint(0, H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        # adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        return images, y_a, y_b, lam

def _get_optimizer(model, optimizer_type, lr, backbone_lr=None, momentum=0.9, weight_decay=1e-4):
    """
    Create optimizer with optional differential learning rates for backbone vs head.
    """
    if backbone_lr is None:
        # All params use same learning rate
        param_groups = [{'params': model.parameters(), 'lr': lr}]
    else:
        # Differential learning rates: backbone gets lower LR, head gets higher LR
        head_params = []
        backbone_params = []
        
        # Identify head parameters (typically the last layer)
        if hasattr(model, 'fc'):  # ResNet
            head_params = list(model.fc.parameters())
            backbone_params = [p for name, p in model.named_parameters() if 'fc' not in name]
        elif hasattr(model, 'classifier'):  # MobileNet, EfficientNet
            if isinstance(model.classifier, nn.Sequential):
                head_params = list(model.classifier[-1].parameters())
            else:
                head_params = list(model.classifier.parameters())
            backbone_params = [p for name, p in model.named_parameters() if 'classifier' not in name]
        
        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': lr}
        ]
    
    if optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(param_groups, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    elif optimizer_type.lower() == 'adam':
        return torch.optim.Adam(param_groups, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

class MetricsLogger:

    """
    Class for logging training and evaluation metrics to both CSV and JSONL files.
    This allows for easy analysis and visualization of results across runs.

    you can use this class like:

    logger = MetricsLogger(log_dir="logs", run_name="experiment1")
    logger.log({"epoch": 1, "train_loss": 0.5, "train_acc": 0.8, "val_loss": 0.4, "val_acc": 0.85})
    logger.close()
    """

    def __init__(self, log_dir, run_name):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.csv_path = self.log_dir / f"{run_name}_metrics.csv"
        self.jsonl_path = self.log_dir / f"{run_name}_metrics.jsonl"
        self._csv_file = None
        self._csv_writer = None

    def _ensure_csv(self, fieldnames):

        """
        Function to lazily initialize the CSV file and writer when the first record is logged.
        This allows us to determine the fieldnames dynamically based on the first record.
        """

        if self._csv_writer is not None:
            return
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        self._csv_writer.writeheader()

    def log(self, record):
        fieldnames = list(record.keys())
        self._ensure_csv(fieldnames)
        self._csv_writer.writerow(record)
        self._csv_file.flush()

        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def close(self):
        if self._csv_file is not None:
            self._csv_file.close()

def train(argv=None):

    """
    handles the training loop, including argument parsing, data loading, model setup, and logging.
    It trains the model for a specified number of epochs and saves the best checkpoint based on validation accuracy.
    """

    parser = argparse.ArgumentParser(description="Train a model on ImageFolder data")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to ImageFolder dataset")
    parser.add_argument("--model", type=str, default="mobilenet_v3_large", help="Model name")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Head learning rate (or global LR if freeze_backbone=True)")
    parser.add_argument("--backbone_lr", type=float, default=None, help="Backbone learning rate (enables differential LR)")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for metrics logs")
    parser.add_argument("--run_name", type=str, default="train", help="Run name prefix for logs")
    _add_bool_argument(parser, "freeze_backbone", True, "Freeze backbone parameters")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer type")
    _add_bool_argument(parser, "use_lr_scheduler", True, "Use learning rate scheduler")
    _add_bool_argument(parser, "use_advanced_aug", True, "Use advanced augmentations")
    _add_bool_argument(parser, "use_mixup", True, "Use Mixup augmentation")
    _add_bool_argument(parser, "use_cutmix", True, "Use CutMix augmentation")
    args = parser.parse_args(argv)
    
    _set_seed(args.seed)
    
    device = _get_device()
    pin_memory = device.type == "cuda"
    train_loader, val_loader, num_classes = _build_dataloaders(
        args.data_dir,
        args.image_size,
        args.batch_size,
        args.num_workers,
        args.val_split,
        args.seed,
        pin_memory,
        use_advanced_aug=args.use_advanced_aug,
    )
    
    model = pretrained_hub.get_model(args.model, num_classes, args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Added label smoothing for regularization
    
    # Setup optimizer with optional differential learning rates
    optimizer = _get_optimizer(
        model, 
        args.optimizer, 
        lr=args.lr, 
        backbone_lr=args.backbone_lr if not args.freeze_backbone else None,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # Setup learning rate scheduler
    scheduler = None
    if args.use_lr_scheduler:
        total_steps = len(train_loader) * args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    # Setup augmentations
    mixup = Mixup(alpha=1.0) if args.use_mixup else None
    cutmix = CutMix(alpha=1.0) if args.use_cutmix else None
    
    base_checkpoint_dir = Path(args.checkpoint_dir)
    base_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create a subdirectory for this specific model/run
    model_run_name = f"{args.model}_{run_timestamp}"
    checkpoint_dir = base_checkpoint_dir / model_run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    best_checkpoint_path = None
    
    logger = MetricsLogger(args.log_dir, args.run_name)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        usage_start = _get_resource_snapshot()
        train_loss, train_acc = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            True,
            desc=f"Epoch {epoch} train",
            mixup=mixup,
            cutmix=cutmix,
            scheduler=scheduler,
        )
        val_loss, val_acc = _run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            False,
            desc=f"Epoch {epoch} val",
        )
        epoch_time_s = time.perf_counter() - epoch_start
        usage_end = _get_resource_snapshot()

        samples = len(train_loader.dataset) + len(val_loader.dataset)
        images_per_s = samples / max(epoch_time_s, 1e-9)
    
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        logger.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_s": epoch_time_s,
            "images_per_s": images_per_s,
            "cpu_user_s_start": usage_start["cpu_user_s"],
            "cpu_system_s_start": usage_start["cpu_system_s"],
            "cpu_user_s_end": usage_end["cpu_user_s"],
            "cpu_system_s_end": usage_end["cpu_system_s"],
            "max_rss_kb": usage_end["max_rss_kb"],
            "device": str(device),
            "model": args.model,
            "batch_size": args.batch_size,
        })
    
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = checkpoint_dir / "best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_classes": num_classes,
                    "model_name": args.model,
                },
                checkpoint_path,
            )
            best_checkpoint_path = checkpoint_path

            config_path = checkpoint_path.with_suffix(".json")
            config = {
                "model_name": args.model,
                "num_classes": num_classes,
                "class_names": train_loader.dataset.dataset.classes,
                "data_dir": args.data_dir,
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "val_split": args.val_split,
                "seed": args.seed,
                "freeze_backbone": args.freeze_backbone,
            }
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

    logger.close()

    if best_checkpoint_path is not None:
        print(f"Best checkpoint saved to: {best_checkpoint_path}")

def eval(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a model on ImageFolder data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to ImageFolder dataset")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--image_size", type=int, default=None, help="Input image size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for metrics logs")
    parser.add_argument("--run_name", type=str, default="eval", help="Run name prefix for logs")
    args = parser.parse_args(argv)

    checkpoint_path = Path(args.checkpoint)
    config_path = checkpoint_path.with_suffix(".json")
    config = None
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)

    if args.data_dir is None:
        if config is None or "data_dir" not in config:
            raise ValueError("data_dir not provided and no config file found next to checkpoint")
        args.data_dir = config["data_dir"]

    if args.image_size is None:
        args.image_size = config.get("image_size", 224) if config else 224

    if args.batch_size is None:
        args.batch_size = config.get("batch_size", 32) if config else 32
    
    device = _get_device()
    
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
        pin_memory=device.type == "cuda",
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = pretrained_hub.get_model(checkpoint["model_name"], checkpoint["num_classes"], freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    logger = MetricsLogger(args.log_dir, args.run_name)
    eval_start = time.perf_counter()
    usage_start = _get_resource_snapshot()
    eval_loss, eval_acc = _run_epoch(
        model,
        loader,
        criterion,
        None,
        device,
        False,
        desc="Eval",
    )
    eval_time_s = time.perf_counter() - eval_start
    usage_end = _get_resource_snapshot()

    print(f"Eval loss {eval_loss:.4f} acc {eval_acc:.4f}")
    logger.log({
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "eval_time_s": eval_time_s,
        "images_per_s": len(loader.dataset) / max(eval_time_s, 1e-9),
        "cpu_user_s_start": usage_start["cpu_user_s"],
        "cpu_system_s_start": usage_start["cpu_system_s"],
        "cpu_user_s_end": usage_end["cpu_user_s"],
        "cpu_system_s_end": usage_end["cpu_system_s"],
        "max_rss_kb": usage_end["max_rss_kb"],
        "device": str(device),
        "batch_size": args.batch_size,
    })
    logger.close()

if __name__ == "__main__":
    # parse args to determine whether to train or eval, and which model to use, etc.
    parser = argparse.ArgumentParser(description="SPPIFN runner")
    parser.add_argument("command", choices=["train", "eval"], help="Command to run")
    args, remaining = parser.parse_known_args()
    
    if args.command == "train":
        train(remaining)
    else:
        eval(remaining)