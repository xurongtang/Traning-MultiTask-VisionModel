"""
Multi-Task Training Script
============================
Main training script for joint Instance Segmentation + Keypoint Detection.

Usage:
    # Train with default config (CPU debug mode)
    python train.py

    # Train with custom settings
    python train.py --epochs 24 --batch_size 4 --device cuda

    # Resume training from checkpoint
    python train.py --resume ./output/checkpoint_epoch_10.pth

Training Pipeline:
    1. Load COCO2017 dataset with instance segmentation + keypoint annotations
    2. Build multi-task model (ResNet-50-FPN + Mask R-CNN + Keypoint Head)
    3. Train with multi-task weighted loss
    4. Log all metrics to TensorBoard
    5. Save checkpoints periodically

Loss Components:
    - RPN Loss (objectness + regression)
    - Box Loss (classification + regression)
    - Mask Loss (instance segmentation)
    - Keypoint Loss (keypoint heatmaps)
"""

import os
import sys
import argparse
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from model.multitask_model import build_multitask_model
from datasets.coco_dataset import build_dataloaders
from loss.multitask_loss import build_multitask_loss, DynamicLossWeightScheduler
from logger.training_logger import build_logger
from utils.visualization import build_visualizer


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg):
    """
    Build optimizer with different learning rates for different parameter groups.

    - Backbone: lower learning rate (pretrained)
    - Heads: higher learning rate (trained from scratch)
    """
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Lower LR for backbone parameters
        if "backbone" in name:
            lr = cfg.learning_rate * 0.1
        else:
            lr = cfg.learning_rate

        params.append({"params": [param], "lr": lr, "initial_lr": lr})

    optimizer = optim.SGD(
        params,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    return optimizer


def build_scheduler(optimizer, cfg):
    """Build learning rate scheduler."""
    if cfg.lr_scheduler == "step":
        scheduler = StepLR(
            optimizer,
            step_size=cfg.lr_step_size,
            gamma=cfg.lr_gamma,
        )
    elif cfg.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.num_epochs,
        )
    else:
        scheduler = StepLR(
            optimizer,
            step_size=cfg.lr_step_size,
            gamma=cfg.lr_gamma,
        )
    return scheduler


def save_checkpoint(model, optimizer, scheduler, epoch, loss_dict, cfg, filepath):
    """Save training checkpoint (handles DataParallel wrapper)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Unwrap DataParallel to save the raw model state_dict
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss_dict": loss_dict,
        "config": cfg,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    print(f"Checkpoint loaded: {filepath} (epoch {checkpoint['epoch']})")
    return checkpoint["epoch"], checkpoint.get("loss_dict", {})


def train_one_epoch(model, optimizer, dataloader, loss_fn, device, epoch, logger, cfg):
    """
    Train for one epoch.

    Args:
        model: Multi-task model
        optimizer: Optimizer
        dataloader: Training dataloader
        loss_fn: Multi-task loss computer
        device: Device (cpu/cuda)
        epoch: Current epoch number
        logger: Training logger
        cfg: Configuration

    Returns:
        avg_loss: Average loss for this epoch
        avg_loss_dict: Average of individual losses
    """
    model.train()
    total_loss_accum = 0.0
    loss_accum = {}
    num_batches = len(dataloader)

    for batch_idx, (images, targets) in enumerate(dataloader):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in t.items()} for t in targets]

        # Forward pass - model returns loss dict in training mode
        optimizer.zero_grad()

        # Get raw losses from model
        # print(images[0].shape)
        loss_dict = model(images, targets)

        # Apply multi-task loss weights
        total_loss = loss_fn.compute_total_loss(loss_dict)

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Accumulate losses
        total_loss_val = total_loss.item()
        total_loss_accum += total_loss_val

        weighted_losses = loss_fn.get_weighted_losses(loss_dict)
        for loss_name, loss_val in weighted_losses.items():
            if loss_name not in loss_accum:
                loss_accum[loss_name] = 0.0
            loss_accum[loss_name] += loss_val

        # Logging
        global_step = epoch * num_batches + batch_idx
        logger.log_losses(weighted_losses, global_step, phase="train")

        # Aggregate task-level losses
        rpn_loss = sum(v for k, v in weighted_losses.items() if "rpn" in k)
        box_loss = sum(v for k, v in weighted_losses.items() if "classifier" in k or "box_reg" in k)
        mask_loss = sum(v for k, v in weighted_losses.items() if "mask" in k)
        kp_loss = sum(v for k, v in weighted_losses.items() if "keypoint" in k)
        logger.log_task_losses(rpn_loss, box_loss, mask_loss, kp_loss, global_step)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.log_lr(current_lr, global_step)

        if (batch_idx + 1) % cfg.log_interval == 0 or batch_idx == 0:
            logger.log_batch(batch_idx, num_batches, weighted_losses, current_lr)

    # Compute averages
    avg_loss = total_loss_accum / max(num_batches, 1)
    avg_loss_dict = {k: v / max(num_batches, 1) for k, v in loss_accum.items()}

    return avg_loss, avg_loss_dict


@torch.no_grad()
def validate(model, dataloader, loss_fn, device, epoch, logger):
    """
    Validate the model.

    Args:
        model: Multi-task model
        dataloader: Validation dataloader
        loss_fn: Multi-task loss computer
        device: Device
        epoch: Current epoch
        logger: Training logger

    Returns:
        avg_val_loss: Average validation loss
    """
    model.train()  # Need train mode to compute losses
    total_loss_accum = 0.0
    loss_accum = {}
    num_batches = len(dataloader)

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        total_loss = loss_fn.compute_total_loss(loss_dict)

        total_loss_accum += total_loss.item()

        weighted_losses = loss_fn.get_weighted_losses(loss_dict)
        for loss_name, loss_val in weighted_losses.items():
            if loss_name not in loss_accum:
                loss_accum[loss_name] = 0.0
            loss_accum[loss_name] += loss_val

        global_step = epoch * num_batches + batch_idx
        logger.log_losses(weighted_losses, global_step, phase="val")

    avg_val_loss = total_loss_accum / max(num_batches, 1)
    avg_loss_dict = {k: v / max(num_batches, 1) for k, v in loss_accum.items()}

    return avg_val_loss, avg_loss_dict


def main():
    parser = argparse.ArgumentParser(description="Multi-Task Training (Instance Seg + Keypoint)")
    parser.add_argument("--data_root", type=str, default=None, help="COCO dataset root")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--log_interval", type=int, default=None, help="Log every N batches")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    # Build config with overrides
    cfg = Config()
    if args.data_root:
        cfg.data_root = args.data_root
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.epochs:
        cfg.num_epochs = args.epochs
    if args.lr:
        cfg.learning_rate = args.lr
    if args.device:
        cfg.device = args.device
    if args.num_workers:
        cfg.num_workers = args.num_workers
    if args.log_interval:
        cfg.log_interval = args.log_interval
    if args.seed:
        cfg.seed = args.seed
    if args.resume:
        cfg.resume = args.resume

    # Set seed
    set_seed(cfg.seed)

    # ── Parse device (support multi-GPU: "cuda:0,1") ───────
    use_multi_gpu = False
    device_ids = None

    if cfg.device.startswith("cuda") and "," in cfg.device:
        # Multi-GPU: "cuda:0,1" → device_ids=[0, 1], primary cuda:0
        gpu_part = cfg.device.split(":")[1] if ":" in cfg.device else cfg.device.replace("cuda", "")
        device_ids = [int(x.strip()) for x in gpu_part.split(",")]
        device = torch.device(f"cuda:{device_ids[0]}")
        use_multi_gpu = True
        print(f"Using multi-GPU: devices {device_ids}, primary {device}")
        print(f"  Available GPUs: {torch.cuda.device_count()}")
        for did in device_ids:
            name = torch.cuda.get_device_name(did)
            mem = torch.cuda.get_device_properties(did).total_memory / (1024 ** 3)
            print(f"  GPU {did}: {name} ({mem:.1f} GB)")
    else:
        device = torch.device(cfg.device)
        print(f"Using device: {device}")

    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Build Components ────────────────────────────────────
    print("\n[1/5] Building multi-task model...")
    model = build_multitask_model(cfg)
    model.to(device)

    # Wrap with DataParallel for multi-GPU
    if use_multi_gpu:
        model = nn.DataParallel(model, device_ids=device_ids)
        print(f"  Model wrapped with DataParallel on {len(device_ids)} GPUs")

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    print("\n[2/5] Building dataset...")
    train_loader, val_loader = build_dataloaders(cfg, mode="both")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Train batches: {len(train_loader)}")

    print("\n[3/5] Building optimizer and scheduler...")
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    print("\n[4/5] Building loss function...")
    loss_fn = build_multitask_loss(cfg)

    print("\n[5/5] Building logger and visualizer...")
    logger = build_logger(cfg)
    visualizer = build_visualizer(cfg)

    # ── Resume if needed ────────────────────────────────────
    # For DataParallel models, checkpoint saves model.module.state_dict()
    start_epoch = 0
    if cfg.resume and os.path.exists(cfg.resume):
        raw_model = model.module if use_multi_gpu else model
        start_epoch, _ = load_checkpoint(cfg.resume, raw_model, optimizer, scheduler)

    # ── Training Loop ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Starting Multi-Task Training")
    print(f"  Tasks: Instance Segmentation + Keypoint Detection")
    print(f"  Epochs: {cfg.num_epochs}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Loss weights: RPN={cfg.loss_weight_rpn}, Box={cfg.loss_weight_box}, "
          f"Mask={cfg.loss_weight_mask}, Keypoint={cfg.loss_weight_keypoint}")
    print(f"{'='*60}")

    best_loss = float("inf")

    for epoch in range(start_epoch, cfg.num_epochs):
        logger.start_epoch(epoch)

        # Train
        avg_train_loss, train_loss_dict = train_one_epoch(
            model, optimizer, train_loader, loss_fn, device, epoch, logger, cfg
        )

        # Validate
        print(f"\n  Validating...")
        avg_val_loss, val_loss_dict = validate(
            model, val_loader, loss_fn, device, epoch, logger
        )

        # Update learning rate
        scheduler.step()

        # Log epoch summary
        train_loss_dict["val_loss"] = avg_val_loss
        logger.end_epoch(epoch, avg_train_loss, train_loss_dict)

        # Save checkpoint
        if (epoch + 1) % cfg.save_interval == 0:
            ckpt_path = os.path.join(cfg.output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss_dict, cfg, ckpt_path)

        # Save best model
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_path = os.path.join(cfg.output_dir, "best_model.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss_dict, cfg, best_path)
            print(f"  New best model! Loss: {best_loss:.4f}")

        # ── Periodic Visualization ──────────────────────────
        if (epoch + 1) % cfg.vis_interval == 0:
            print(f"\n  Generating visualizations for epoch {epoch + 1}...")
            vis_paths = visualizer.visualize_epoch(
                model=model,
                val_dataset=val_loader.dataset,
                epoch=epoch + 1,
                device=device,
            )
            if vis_paths:
                for task, path in vis_paths.items():
                    logger.log_text(f"visualization/epoch_{epoch+1}/{task}", path, epoch + 1)

    # ── Training Complete ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Checkpoints saved to: {cfg.output_dir}")
    print(f"  Logs saved to: {logger.log_dir}")
    print(f"{'='*60}")

    # Save final model
    final_path = os.path.join(cfg.output_dir, "final_model.pth")
    save_checkpoint(model, optimizer, scheduler, cfg.num_epochs - 1, {}, cfg, final_path)

    logger.close()


if __name__ == "__main__":
    main()