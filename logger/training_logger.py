"""
Training Logger
================
TensorBoard-based logger for multi-task training monitoring.

Logs:
    - Total loss
    - Individual task losses (RPN, Box, Mask, Keypoint)
    - Learning rate
    - Epoch statistics

Usage:
    logger = TrainingLogger(log_dir="./runs/experiment_1")
    logger.log_losses(loss_dict, step=global_step)
    logger.log_lr(learning_rate, step=global_step)
    logger.close()
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


class TrainingLogger:
    """
    Multi-task training logger with TensorBoard and console output.

    Args:
        log_dir: Directory for TensorBoard logs
        experiment_name: Name of the experiment
        use_tensorboard: Whether to use TensorBoard (default: True)
    """

    def __init__(
        self,
        log_dir: str = "./runs",
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = True,
    ):
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # TensorBoard writer
        self.writer = None
        if use_tensorboard and HAS_TENSORBOARD:
            self.writer = SummaryWriter(self.log_dir)
            print(f"TensorBoard logging to: {self.log_dir}")
        else:
            print("TensorBoard logging disabled.")

        # Training statistics
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "task_losses": [],
            "learning_rates": [],
            "epoch_times": [],
        }

        self.current_epoch = 0
        self.epoch_start_time = None
        self.epoch_losses = []

    def log_losses(self, loss_dict: Dict[str, float], step: int, phase: str = "train"):
        """
        Log individual and total losses.

        Args:
            loss_dict: Dictionary of loss_name → loss_value
            step: Global step number
            phase: "train" or "val"
        """
        if self.writer is not None:
            for loss_name, loss_value in loss_dict.items():
                self.writer.add_scalar(f"{phase}/{loss_name}", loss_value, step)

            # Log total loss
            total = sum(loss_dict.values())
            self.writer.add_scalar(f"{phase}/total_loss", total, step)

    def log_task_losses(
        self,
        rpn_loss: float,
        box_loss: float,
        mask_loss: float,
        keypoint_loss: float,
        step: int,
        phase: str = "train",
    ):
        """
        Log task-specific losses grouped by task.

        Args:
            rpn_loss: Combined RPN loss
            box_loss: Combined Box head loss
            mask_loss: Instance segmentation loss
            keypoint_loss: Keypoint detection loss
            step: Global step
            phase: "train" or "val"
        """
        if self.writer is not None:
            self.writer.add_scalar(f"{phase}_tasks/rpn", rpn_loss, step)
            self.writer.add_scalar(f"{phase}_tasks/box", box_loss, step)
            self.writer.add_scalar(f"{phase}_tasks/mask", mask_loss, step)
            self.writer.add_scalar(f"{phase}_tasks/keypoint", keypoint_loss, step)

            # Log task balance (ratio between tasks)
            if mask_loss > 0 and keypoint_loss > 0:
                ratio = mask_loss / (keypoint_loss + 1e-8)
                self.writer.add_scalar(f"{phase}_tasks/mask_keypoint_ratio", ratio, step)

    def log_lr(self, lr: float, step: int):
        """Log learning rate."""
        if self.writer is not None:
            self.writer.add_scalar("learning_rate", lr, step)

    def log_text(self, tag: str, text: str, step: int):
        """Log text to TensorBoard."""
        if self.writer is not None:
            self.writer.add_text(tag, text, step)

    def log_images_with_predictions(
        self, images, targets, predictions, step: int, max_images: int = 4
    ):
        """
        Log sample images with predictions (for visual monitoring).

        Args:
            images: List of image tensors
            targets: List of target dicts
            predictions: List of prediction dicts
            step: Global step
            max_images: Maximum number of images to log
        """
        if self.writer is None:
            return

        import torchvision
        import torch

        for i in range(min(max_images, len(images))):
            img = images[i]
            # Add batch dimension for grid
            if img.dim() == 3:
                self.writer.add_image(f"predictions/{i}", img, step)

    def start_epoch(self, epoch: int):
        """Mark the start of an epoch."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self.epoch_losses = []
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}")
        print(f"{'='*60}")

    def end_epoch(self, epoch: int, avg_loss: float, loss_dict: Optional[Dict] = None):
        """
        Mark the end of an epoch and log summary.

        Args:
            epoch: Epoch number
            avg_loss: Average loss for this epoch
            loss_dict: Optional detailed loss breakdown
        """
        elapsed = time.time() - self.epoch_start_time if self.epoch_start_time else 0

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Time: {elapsed:.1f}s")
        if loss_dict:
            for name, val in loss_dict.items():
                print(f"  {name}: {val:.4f}")

        # Save to history
        self.history["train_loss"].append(avg_loss)
        if loss_dict:
            self.history["task_losses"].append(loss_dict)
        self.history["epoch_times"].append(elapsed)

        # Log epoch summary to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar("epoch/avg_loss", avg_loss, epoch)
            self.writer.add_scalar("epoch/time", elapsed, epoch)
            if loss_dict:
                for name, val in loss_dict.items():
                    self.writer.add_scalar(f"epoch/{name}", val, epoch)

    def log_batch(
        self,
        batch_idx: int,
        total_batches: int,
        loss_dict: Dict[str, float],
        lr: float,
    ):
        """
        Log batch-level training information.

        Args:
            batch_idx: Current batch index
            total_batches: Total number of batches
            loss_dict: Dictionary of losses
            lr: Current learning rate
        """
        total_loss = sum(loss_dict.values())

        # Accumulate for epoch average
        self.epoch_losses.append(total_loss)

        # Console output
        loss_str = "  ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
        print(
            f"  [{batch_idx + 1}/{total_batches}] "
            f"Loss: {total_loss:.4f}  |  {loss_str}  |  LR: {lr:.6f}"
        )

    def save_history(self):
        """Save training history to JSON file."""
        history_path = os.path.join(self.log_dir, "training_history.json")
        # Convert all values to Python types
        history = {}
        for key, values in self.history.items():
            history[key] = [
                {k: float(v) for k, v in item.items()} if isinstance(item, dict) else float(item)
                for item in values
            ]
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to {history_path}")

    def close(self):
        """Close the logger and TensorBoard writer."""
        self.save_history()
        if self.writer is not None:
            self.writer.close()
            print("TensorBoard writer closed.")


def build_logger(cfg) -> TrainingLogger:
    """Build training logger from configuration."""
    return TrainingLogger(
        log_dir=cfg.log_dir,
        use_tensorboard=True,
    )


if __name__ == "__main__":
    # Quick test
    logger = TrainingLogger(log_dir="./runs/test")
    print(f"Logger created at: {logger.log_dir}")

    # Simulate some training
    for epoch in range(3):
        logger.start_epoch(epoch)
        for batch in range(5):
            fake_losses = {
                "rpn_cls": 0.5 - epoch * 0.1,
                "rpn_reg": 0.3 - epoch * 0.05,
                "box_cls": 0.4 - epoch * 0.08,
                "mask": 0.6 - epoch * 0.12,
                "keypoint": 0.8 - epoch * 0.15,
            }
            step = epoch * 5 + batch
            logger.log_losses(fake_losses, step)
            logger.log_batch(batch, 5, fake_losses, lr=0.005)

        avg_loss = sum(logger.epoch_losses) / len(logger.epoch_losses)
        logger.end_epoch(epoch, avg_loss)

    logger.close()