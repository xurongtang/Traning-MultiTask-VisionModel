"""
Multi-Task Model: Instance Segmentation + Keypoint Detection
=============================================================
Complete multi-task model combining:
    - Shared Backbone (ResNet-50 + FPN)
    - Region Proposal Network (RPN)
    - Box Prediction Head (shared)
    - Instance Segmentation Head (Task 1)
    - Keypoint Detection Head (Task 2)

Architecture:
    ┌─────────────────────────────────────┐
    │            Input Image               │
    └─────────────┬───────────────────────┘
                  │
    ┌─────────────▼───────────────────────┐
    │     Shared Backbone (ResNet+FPN)     │
    │    P2, P3, P4, P5, P6 features      │
    └─────────────┬───────────────────────┘
                  │
    ┌─────────────▼───────────────────────┐
    │          RPN (Region Proposals)      │
    └──────┬──────────┬──────────────────┘
           │          │
    ┌──────▼──┐  ┌────▼──────┐
    │ Box Head│  │  Box Head  │
    │ (shared)│  │  (shared)  │
    └──────┬──┘  └────┬──────┘
           │          │
    ┌──────▼──┐  ┌────▼──────────┐
    │  Mask   │  │  Keypoint     │
    │  Head   │  │  Head         │
    │(Task 1) │  │  (Task 2)     │
    └─────────┘  └───────────────┘

This model extends torchvision's MaskRCNN framework to support
joint instance segmentation and keypoint detection with configurable
loss weights for multi-task learning.
"""

import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


class MultiTaskModel(nn.Module):
    """
    Multi-task model for joint Instance Segmentation and Keypoint Detection.

    Wraps torchvision's MaskRCNN which natively supports:
        - Region Proposal Network (RPN)
        - Bounding Box detection
        - Instance Segmentation masks
        - Keypoint detection (17 COCO keypoints for person)

    The model is trained with a combined multi-task loss:
        L_total = w_rpn * L_rpn + w_box * L_box + w_mask * L_mask + w_kp * L_keypoint

    Args:
        cfg: Configuration object with model and training parameters
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # Build backbone (ResNet-50 + FPN)
        backbone = resnet_fpn_backbone(
            backbone_name=cfg.backbone_name,
            weights="DEFAULT" if cfg.pretrained_backbone else None,
            trainable_layers=cfg.trainable_layers,
            extra_blocks=LastLevelMaxPool(),
        )

        # Anchor generator for RPN
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )

        # ROI align for box head
        from torchvision.ops import MultiScaleRoIAlign
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2,
        )

        # ROI align for mask head
        mask_roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=14,
            sampling_ratio=2,
        )

        # ROI align for keypoint head
        keypoint_roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=14,
            sampling_ratio=2,
        )

        # Build Mask R-CNN with keypoint support
        # MaskRCNN internally creates mask and keypoint predictors
        self.model = MaskRCNN(
            backbone=backbone,
            num_classes=cfg.num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            mask_roi_pool=mask_roi_pooler,
            keypoint_roi_pool=keypoint_roi_pooler,
            num_keypoints=cfg.num_keypoints,
            min_size=cfg.min_size,
            max_size=cfg.max_size,
        )

        # Loss weights for multi-task learning
        self.loss_weights = {
            "rpn": cfg.loss_weight_rpn,
            "box": cfg.loss_weight_box,
            "mask": cfg.loss_weight_mask,
            "keypoint": cfg.loss_weight_keypoint,
        }

    def forward(self, images, targets=None):
        """
        Forward pass.

        In training mode: computes losses for all tasks.
        In eval mode: returns predictions.

        Args:
            images: List of image tensors (C, H, W)
            targets: List of target dicts with keys:
                - boxes: (N, 4) bounding boxes
                - labels: (N,) class labels
                - masks: (N, H, W) instance masks
                - keypoints: (N, K, 3) keypoints (x, y, visibility)

        Returns:
            Training: Dict of losses
            Eval: List of prediction dicts
        """
        # print("in Model---: ", images[0].shape)
        return self.model(images, targets)

    def compute_weighted_loss(self, images, targets):
        """
        Compute multi-task weighted loss.

        Args:
            images: List of image tensors
            targets: List of target dicts

        Returns:
            total_loss: Weighted sum of all task losses
            loss_dict: Dictionary of individual losses
        """
        loss_dict = self.model(images, targets)

        # Apply task-specific loss weights
        weighted_loss_dict = {}
        total_loss = torch.tensor(0.0, device=loss_dict.get(
            "loss_classifier", loss_dict.get("loss_rpn_class")
        ).device)

        for loss_name, loss_value in loss_dict.items():
            # Determine which task this loss belongs to
            weight = 1.0
            if "rpn" in loss_name:
                weight = self.loss_weights["rpn"]
            elif "classifier" in loss_name or "box_reg" in loss_name:
                weight = self.loss_weights["box"]
            elif "mask" in loss_name:
                weight = self.loss_weights["mask"]
            elif "keypoint" in loss_name:
                weight = self.loss_weights["keypoint"]

            weighted_loss = loss_value * weight
            weighted_loss_dict[loss_name] = loss_value  # Log unweighted value
            total_loss += weighted_loss

        return total_loss, weighted_loss_dict


class MaskRCNNPredictor(nn.Module):
    """Mask prediction head for instance segmentation."""

    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__()
        self.conv5_mask = nn.ConvTranspose2d(
            in_channels, dim_reduced, kernel_size=2, stride=2
        )
        self.mask_fcn_logits = nn.Conv2d(
            dim_reduced, num_classes, kernel_size=1
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.conv5_mask(x)
        x = torch.relu(x)
        return self.mask_fcn_logits(x)


class KeypointRCNNPredictor(nn.Module):
    """Keypoint prediction head for pose estimation."""

    def __init__(self, in_channels, num_keypoints):
        super().__init__()
        self.kps_score_lowres = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=4, stride=2
        )
        self.kps_score = nn.Conv2d(
            in_channels, num_keypoints, kernel_size=1
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.kps_score_lowres(x)
        x = torch.relu(x)
        return self.kps_score(x)


def build_multitask_model(cfg) -> MultiTaskModel:
    """Build the multi-task model from configuration."""
    model = MultiTaskModel(cfg)
    return model


if __name__ == "__main__":
    from config import Config
    cfg = Config()
    model = build_multitask_model(cfg)
    print(f"Multi-Task Model built successfully")
    print(f"  Backbone: {cfg.backbone_name}")
    print(f"  Num classes: {cfg.num_classes}")
    print(f"  Num keypoints: {cfg.num_keypoints}")
    print(f"  Loss weights: RPN={cfg.loss_weight_rpn}, Box={cfg.loss_weight_box}, "
          f"Mask={cfg.loss_weight_mask}, KP={cfg.loss_weight_keypoint}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")