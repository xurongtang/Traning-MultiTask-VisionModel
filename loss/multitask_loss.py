"""
Multi-Task Loss
================
Loss computation and weighting for multi-task learning.

The total training loss combines four components:
    1. RPN Loss (shared):
       - L_rpn_cls: Object/non-object classification (Binary Cross Entropy)
       - L_rpn_reg: Bounding box regression (Smooth L1)

    2. Box Head Loss (shared):
       - L_box_cls: Multi-class classification (Cross Entropy)
       - L_box_reg: Bounding box regression (Smooth L1)

    3. Instance Segmentation Loss (Task 1):
       - L_mask: Per-pixel binary cross-entropy for mask prediction

    4. Keypoint Detection Loss (Task 2):
       - L_keypoint: Keypoint heatmap loss (Cross Entropy over heatmaps)

Total Loss:
    L_total = w_rpn * (L_rpn_cls + L_rpn_reg)
            + w_box * (L_box_cls + L_box_reg)
            + w_mask * L_mask
            + w_kp * L_keypoint

Note:
    The Mask R-CNN model computes all losses internally during training.
    This module provides loss weighting and monitoring utilities.
"""

import torch
import torch.nn as nn
from typing import Dict


class MultiTaskLoss:
    """
    Multi-task loss weighting and monitoring.

    Applies configurable weights to each task's losses and provides
    detailed loss monitoring for training analysis.

    Args:
        loss_weight_rpn: Weight for RPN losses
        loss_weight_box: Weight for Box head losses
        loss_weight_mask: Weight for Instance Segmentation losses
        loss_weight_keypoint: Weight for Keypoint Detection losses
    """

    def __init__(
        self,
        loss_weight_rpn: float = 1.0,
        loss_weight_box: float = 1.0,
        loss_weight_mask: float = 1.0,
        loss_weight_keypoint: float = 1.0,
    ):
        self.weights = {
            "loss_rpn_class": loss_weight_rpn,
            "loss_rpn_bbox_reg": loss_weight_rpn,
            "loss_classifier": loss_weight_box,
            "loss_box_reg": loss_weight_box,
            "loss_mask": loss_weight_mask,
            "loss_keypoint": loss_weight_keypoint,
        }

    def compute_total_loss(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted total loss from individual task losses.

        Args:
            loss_dict: Dictionary of loss_name → loss_value from model

        Returns:
            total_loss: Weighted sum of all losses
        """
        total_loss = torch.tensor(0.0, device=self._get_device(loss_dict))

        for loss_name, loss_value in loss_dict.items():
            weight = self._get_weight(loss_name)
            total_loss = total_loss + loss_value * weight

        return total_loss

    def get_weighted_losses(self, loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Get individual weighted losses as floats for logging.

        Args:
            loss_dict: Dictionary of loss_name → loss_value

        Returns:
            Dictionary of loss_name → weighted_value (float)
        """
        result = {}
        for loss_name, loss_value in loss_dict.items():
            weight = self._get_weight(loss_name)
            result[loss_name] = loss_value.item() * weight
        return result

    def _get_weight(self, loss_name: str) -> float:
        """Get loss weight for a given loss name."""
        for key, weight in self.weights.items():
            if key in loss_name:
                return weight
        return 1.0  # Default weight

    def _get_device(self, loss_dict: Dict[str, torch.Tensor]) -> torch.device:
        """Get device from loss tensors."""
        for v in loss_dict.values():
            return v.device
        return torch.device("cpu")


class DynamicLossWeightScheduler:
    """
    Dynamic loss weight scheduler for multi-task learning.

    Supports several strategies for adjusting task loss weights during training:
    - "constant": Fixed weights throughout training
    - "linear_warmup": Gradually increase task weights
    - "alternating": Focus on different tasks in different epochs

    Args:
        weights: Initial weights dict
        strategy: Scheduling strategy
        total_epochs: Total training epochs
    """

    def __init__(
        self,
        weights: Dict[str, float],
        strategy: str = "constant",
        total_epochs: int = 12,
    ):
        self.initial_weights = weights.copy()
        self.current_weights = weights.copy()
        self.strategy = strategy
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def step(self, epoch: int):
        """Update weights for the given epoch."""
        self.current_epoch = epoch

        if self.strategy == "constant":
            pass  # Keep initial weights
        elif self.strategy == "linear_warmup":
            self._linear_warmup(epoch)
        elif self.strategy == "alternating":
            self._alternating(epoch)

    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return self.current_weights.copy()

    def _linear_warmup(self, epoch: int):
        """Linearly increase weights from 0 to initial value over warmup period."""
        warmup_epochs = max(1, self.total_epochs // 10)
        if epoch < warmup_epochs:
            factor = (epoch + 1) / warmup_epochs
            for key in self.current_weights:
                self.current_weights[key] = self.initial_weights[key] * factor

    def _alternating(self, epoch: int):
        """Alternate between focusing on mask and keypoint tasks."""
        cycle = epoch % 4
        if cycle < 2:
            # Focus on instance segmentation
            self.current_weights["loss_mask"] = self.initial_weights["loss_mask"] * 2.0
            self.current_weights["loss_keypoint"] = self.initial_weights["loss_keypoint"] * 0.5
        else:
            # Focus on keypoints
            self.current_weights["loss_mask"] = self.initial_weights["loss_mask"] * 0.5
            self.current_weights["loss_keypoint"] = self.initial_weights["loss_keypoint"] * 2.0


def build_multitask_loss(cfg) -> MultiTaskLoss:
    """Build multi-task loss from configuration."""
    return MultiTaskLoss(
        loss_weight_rpn=cfg.loss_weight_rpn,
        loss_weight_box=cfg.loss_weight_box,
        loss_weight_mask=cfg.loss_weight_mask,
        loss_weight_keypoint=cfg.loss_weight_keypoint,
    )


if __name__ == "__main__":
    # Test multi-task loss
    loss_fn = MultiTaskLoss(
        loss_weight_rpn=1.0,
        loss_weight_box=1.0,
        loss_weight_mask=1.0,
        loss_weight_keypoint=1.0,
    )

    # Simulate loss dict from model
    fake_losses = {
        "loss_rpn_class": torch.tensor(0.5),
        "loss_rpn_bbox_reg": torch.tensor(0.3),
        "loss_classifier": torch.tensor(0.4),
        "loss_box_reg": torch.tensor(0.2),
        "loss_mask": torch.tensor(0.6),
        "loss_keypoint": torch.tensor(0.8),
    }

    total = loss_fn.compute_total_loss(fake_losses)
    weighted = loss_fn.get_weighted_losses(fake_losses)

    print("Multi-Task Loss Test:")
    print(f"  Total loss: {total.item():.4f}")
    print(f"  Individual weighted losses:")
    for name, val in weighted.items():
        print(f"    {name}: {val:.4f}")

    # Test dynamic scheduler
    print("\nDynamic Loss Weight Scheduler Test:")
    scheduler = DynamicLossWeightScheduler(
        weights={"loss_mask": 1.0, "loss_keypoint": 1.0},
        strategy="alternating",
        total_epochs=12,
    )
    for epoch in range(8):
        scheduler.step(epoch)
        print(f"  Epoch {epoch}: {scheduler.get_weights()}")