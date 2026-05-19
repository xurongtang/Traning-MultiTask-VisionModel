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
import torch.distributed as dist
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP

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


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def unwrap_model(model):
    if isinstance(model, (nn.DataParallel, DDP)):
        return model.module
    return model


def print_main(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def init_distributed_mode(cfg):
    """
    Initialize torch.distributed when launched with torchrun.

    Multi-GPU training is supported through:
        torchrun --nproc_per_node=<N> train.py --device cuda
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if cfg.device.startswith("cuda") and "," in cfg.device and not distributed:
        raise ValueError(
            "Multi-GPU training now uses DDP. Launch with "
            "`torchrun --nproc_per_node=<num_gpus> train.py --device cuda`."
        )

    if not distributed:
        device = torch.device(cfg.device)
        print(f"Using device: {device}")
        return device, False

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed multi-GPU training requires CUDA.")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{local_rank}")
    print(
        f"[rank {get_rank()}] Distributed initialized "
        f"(world_size={get_world_size()}, local_rank={local_rank}, device={device})"
    )
    return device, True


def cleanup_distributed():
    if is_dist_initialized():
        dist.destroy_process_group()


def reduce_mean(value: float, device: torch.device, count: int = 1) -> float:
    if not is_dist_initialized():
        return value / max(count, 1)

    tensor = torch.tensor([value, count], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor[0] / tensor[1].clamp_min(1.0)).item()


def reduce_loss_dict(loss_accum, device, count):
    if count == 0:
        return {}

    if not is_dist_initialized():
        return {k: v / count for k, v in loss_accum.items()}

    keys = sorted(loss_accum.keys())
    values = torch.tensor([loss_accum[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    count_tensor = torch.tensor([count], dtype=torch.float64, device=device)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    denom = count_tensor.item() if count_tensor.item() > 0 else 1.0
    return {k: (values[idx] / denom).item() for idx, k in enumerate(keys)}


def save_checkpoint(model, optimizer, scheduler, epoch, loss_dict, cfg, filepath):
    """Save training checkpoint (handles wrapped models such as DDP)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    raw_model = unwrap_model(model)
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
        if logger is not None:
            logger.log_losses(weighted_losses, global_step, phase="train")

        # Aggregate task-level losses
        rpn_loss = sum(
            v for k, v in weighted_losses.items()
            if k in {"loss_objectness", "loss_rpn_box_reg"}
        )
        box_loss = sum(v for k, v in weighted_losses.items() if k in {"loss_classifier", "loss_box_reg"})
        mask_loss = sum(v for k, v in weighted_losses.items() if "mask" in k)
        kp_loss = sum(v for k, v in weighted_losses.items() if "keypoint" in k)
        if logger is not None:
            logger.log_task_losses(rpn_loss, box_loss, mask_loss, kp_loss, global_step)

        current_lr = optimizer.param_groups[0]["lr"]
        if logger is not None:
            logger.log_lr(current_lr, global_step)

        if logger is not None and ((batch_idx + 1) % cfg.log_interval == 0 or batch_idx == 0):
            logger.log_batch(batch_idx, num_batches, weighted_losses, current_lr)

    # Compute averages
    avg_loss = reduce_mean(total_loss_accum, device, num_batches)
    avg_loss_dict = reduce_loss_dict(loss_accum, device, num_batches)

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
        if logger is not None:
            logger.log_losses(weighted_losses, global_step, phase="val")

    avg_val_loss = reduce_mean(total_loss_accum, device, num_batches)
    avg_loss_dict = reduce_loss_dict(loss_accum, device, num_batches)

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

    device = None
    logger = None
    distributed = False

    try:
        device, distributed = init_distributed_mode(cfg)

        # Make stochastic ops differ across ranks while remaining reproducible.
        set_seed(cfg.seed + get_rank())

        if is_main_process():
            os.makedirs(cfg.output_dir, exist_ok=True)

        print_main("\n[1/5] Building multi-task model...")
        model = build_multitask_model(cfg)
        model.to(device)
        if distributed:
            model = DDP(
                model,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=True,
            )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print_main(f"  Total params: {total_params:,}")
        print_main(f"  Trainable params: {trainable_params:,}")

        print_main("\n[2/5] Building dataset...")
        train_loader, val_loader = build_dataloaders(cfg, mode="both", distributed=distributed)
        print_main(f"  Train samples: {len(train_loader.dataset)}")
        print_main(f"  Val samples: {len(val_loader.dataset)}")
        print_main(f"  Train batches/rank: {len(train_loader)}")

        print_main("\n[3/5] Building optimizer and scheduler...")
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)

        print_main("\n[4/5] Building loss function...")
        loss_fn = build_multitask_loss(cfg)

        print_main("\n[5/5] Building logger and visualizer...")
        logger = build_logger(cfg, enabled=is_main_process())
        visualizer = build_visualizer(cfg) if is_main_process() else None

        start_epoch = 0
        if cfg.resume and os.path.exists(cfg.resume):
            raw_model = unwrap_model(model)
            start_epoch, _ = load_checkpoint(cfg.resume, raw_model, optimizer, scheduler)

        print_main(f"\n{'='*60}")
        print_main("Starting Multi-Task Training")
        print_main("  Tasks: Instance Segmentation + Keypoint Detection")
        print_main(f"  Epochs: {cfg.num_epochs}")
        print_main(f"  Batch size per rank: {cfg.batch_size}")
        print_main(f"  World size: {get_world_size()}")
        print_main(f"  Learning rate: {cfg.learning_rate}")
        print_main(
            f"  Loss weights: RPN={cfg.loss_weight_rpn}, Box={cfg.loss_weight_box}, "
            f"Mask={cfg.loss_weight_mask}, Keypoint={cfg.loss_weight_keypoint}"
        )
        print_main(f"{'='*60}")

        best_loss = float("inf")

        for epoch in range(start_epoch, cfg.num_epochs):
            if distributed and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            if distributed and hasattr(val_loader.sampler, "set_epoch"):
                val_loader.sampler.set_epoch(epoch)

            if logger is not None:
                logger.start_epoch(epoch)

            avg_train_loss, train_loss_dict = train_one_epoch(
                model, optimizer, train_loader, loss_fn, device, epoch, logger, cfg
            )

            print_main("\n  Validating...")
            avg_val_loss, val_loss_dict = validate(
                model, val_loader, loss_fn, device, epoch, logger
            )

            scheduler.step()

            if logger is not None:
                epoch_loss_dict = dict(train_loss_dict)
                epoch_loss_dict["val_loss"] = avg_val_loss
                for name, value in val_loss_dict.items():
                    epoch_loss_dict[f"val_{name}"] = value
                logger.end_epoch(epoch, avg_train_loss, epoch_loss_dict)

            if is_main_process() and (epoch + 1) % cfg.save_interval == 0:
                ckpt_path = os.path.join(cfg.output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                save_checkpoint(model, optimizer, scheduler, epoch, train_loss_dict, cfg, ckpt_path)

            if is_main_process() and avg_train_loss < best_loss:
                best_loss = avg_train_loss
                best_path = os.path.join(cfg.output_dir, "best_model.pth")
                save_checkpoint(model, optimizer, scheduler, epoch, train_loss_dict, cfg, best_path)
                print_main(f"  New best model! Loss: {best_loss:.4f}")

            if is_main_process() and visualizer is not None and (epoch + 1) % cfg.vis_interval == 0:
                print_main(f"\n  Generating visualizations for epoch {epoch + 1}...")
                vis_paths = visualizer.visualize_epoch(
                    model=unwrap_model(model),
                    val_dataset=val_loader.dataset,
                    epoch=epoch + 1,
                    device=device,
                )
                if vis_paths and logger is not None:
                    for task, path in vis_paths.items():
                        logger.log_text(f"visualization/epoch_{epoch+1}/{task}", path, epoch + 1)

            if distributed:
                dist.barrier()

        print_main(f"\n{'='*60}")
        print_main("Training Complete!")
        print_main(f"  Best loss: {best_loss:.4f}")
        print_main(f"  Checkpoints saved to: {cfg.output_dir}")
        if logger is not None:
            print_main(f"  Logs saved to: {logger.log_dir}")
        print_main(f"{'='*60}")

        if is_main_process():
            final_path = os.path.join(cfg.output_dir, "final_model.pth")
            save_checkpoint(model, optimizer, scheduler, cfg.num_epochs - 1, {}, cfg, final_path)
    finally:
        if logger is not None:
            logger.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
