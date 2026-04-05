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

        # Build Mask R-CNN
        # NOTE: MaskRCNN.__init__ in torchvision 0.23.0 does NOT pass
        # keypoint_* kwargs through to RoIHeads (they are silently ignored).
        # We must manually assign keypoint components after construction.
        self.model = MaskRCNN(
            backbone=backbone,
            num_classes=cfg.num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            mask_roi_pool=mask_roi_pooler,
            min_size=cfg.min_size,
            max_size=cfg.max_size,
        )

        # ── Manually set up Keypoint Head ─────────────────────
        # (MaskRCNN ignores keypoint_* kwargs, so we assign them directly)
        # Use torchvision's standard KeypointRCNN components for compatibility
        from torchvision.models.detection.keypoint_rcnn import (
            KeypointRCNNHeads as TorchKeypointRCNNHeads,
            KeypointRCNNPredictor as TorchKeypointRCNNPredictor,
        )

        out_channels = backbone.out_channels  # 256 for ResNet-FPN
        keypoint_layers = (512,) * 8  # Standard: 8 conv layers of 512 channels
        keypoint_head = TorchKeypointRCNNHeads(out_channels, keypoint_layers)
        keypoint_predictor = TorchKeypointRCNNPredictor(
            in_channels=512,  # Must match keypoint_layers[-1]
            num_keypoints=cfg.num_keypoints,
        )

        self.model.roi_heads.keypoint_roi_pool = keypoint_roi_pooler
        self.model.roi_heads.keypoint_head = keypoint_head
        self.model.roi_heads.keypoint_predictor = keypoint_predictor

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
    import os
    import argparse

    # ── Parse args FIRST to decide whether to download pretrained backbone ──
    parser = argparse.ArgumentParser(description="Multi-task model test script")
    parser.add_argument(
        "--checkpoint", "-c", type=str, default=None,
        help="Path to .pth checkpoint file. If not provided, uses pretrained backbone weights only."
    )
    parser.add_argument(
        "--device", "-d", type=str, default=None,
        help="Device to run inference on (e.g. 'cpu', 'cuda:0'). Auto-detected if not set."
    )
    args = parser.parse_args()

    # Auto-detect device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Build model ──────────────────────────────────────────
    from config import Config
    cfg = Config()

    # If checkpoint is provided, skip downloading pretrained backbone
    # (checkpoint contains ALL weights including backbone)
    if args.checkpoint:
        print(f"Checkpoint provided: {args.checkpoint}")
        print("Skipping pretrained backbone download, will load from checkpoint.")
        cfg.pretrained_backbone = False

    model = build_multitask_model(cfg)

    # ── Load checkpoint weights ──────────────────────────────
    if args.checkpoint:
        ckpt_path = args.checkpoint
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            epoch = checkpoint.get("epoch", "?")
            print(f"  Checkpoint epoch: {epoch}")
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Load weights (strict=False to allow partial matches)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        print("  Checkpoint loaded successfully!")
    else:
        print("No checkpoint provided (--checkpoint), using pretrained backbone weights only.")

    model.to(device)

    # ── Debug: check keypoint head status ─────────────────────
    print(f"\n=== Model Keypoint Head Debug ===")
    print(f"  roi_heads.keypoint_predictor is None: {model.model.roi_heads.keypoint_predictor is None}")
    print(f"  roi_heads.keypoint_roi_pool is None: {model.model.roi_heads.keypoint_roi_pool is None}")
    if model.model.roi_heads.keypoint_predictor is not None:
        for name, param in model.model.roi_heads.keypoint_predictor.named_parameters():
            print(f"  keypoint_predictor.{name}: shape={param.shape}, mean={param.data.mean().item():.6f}")
    print()

    import cv2
    import numpy as np
    import torchvision.transforms.functional as F

    # 1. Load image (resolve path relative to project root)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_img_path = os.path.join(project_root, "test.jpg")
    test_input = cv2.imread(test_img_path)
    if test_input is None:
        raise FileNotFoundError(f"Cannot load {test_img_path}")

    # 2. BGR -> RGB, then convert to float tensor [C, H, W]
    test_input_rgb = cv2.cvtColor(test_input, cv2.COLOR_BGR2RGB)
    test_input_tensor = F.to_tensor(test_input_rgb).to(device)  # [C, H, W], float32 [0, 1]

    # 3. Set model to eval mode and run inference
    #    MaskRCNN expects a list of image tensors
    model.eval()
    with torch.no_grad():
        res = model([test_input_tensor])

    # 4. Print detection results
    pred = res[0]
    scores = pred["scores"]
    print(f"\n=== Detection Results ===")
    print(f"Prediction keys: {list(pred.keys())}")
    print(f"Detected {len(scores)} instances")
    print(f"Has keypoints: {'keypoints' in pred}")

    if len(scores) > 0:
        # Print top-5 detections regardless of threshold
        topk = min(5, len(scores))
        top_scores, top_indices = scores.topk(topk)
        print(f"\nTop-{topk} detections:")
        for rank, (score, i) in enumerate(zip(top_scores, top_indices)):
            label = pred["labels"][i].item()
            box = pred["boxes"][i].tolist()
            print(f"  [{rank}] label={label}, score={score:.3f}, box={[round(v, 1) for v in box]}")

            # Print keypoints if available
            if "keypoints" in pred:
                kp = pred["keypoints"][i]  # (K, 3): x, y, visibility
                visible = kp[:, 2] > 0.1
                num_visible = visible.sum().item()
                print(f"       keypoints: {num_visible}/{kp.shape[0]} visible")
    else:
        print("No detections found! The model may need training or a valid checkpoint.")

    # 5. Visualize and save results
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # COCO person skeleton
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]

    INSTANCE_COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    ]

    score_threshold = 0.5
    keep = scores > score_threshold
    keep_indices = keep.nonzero(as_tuple=True)[0]

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_output")
    os.makedirs(output_dir, exist_ok=True)

    # --- Draw Instance Segmentation ---
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(test_input_rgb)
    ax.set_title("Instance Segmentation + Keypoints")
    ax.axis("off")

    for i in keep_indices:
        color = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
        color_norm = tuple(c / 255.0 for c in color)

        # Draw bounding box
        x1, y1, x2, y2 = pred["boxes"][i].tolist()
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color_norm, facecolor="none",
        )
        ax.add_patch(rect)
        label_id = pred["labels"][i].item()
        score_val = pred["scores"][i].item()
        ax.text(
            x1, y1 - 5, f"cls={label_id} score={score_val:.2f}",
            fontsize=8, color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=(*color_norm, 0.7)),
        )

        # Draw mask overlay
        if "masks" in pred:
            mask = pred["masks"][i, 0].cpu().numpy()
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask > 0.5, 0] = color[0] / 255.0
            colored_mask[mask > 0.5, 1] = color[1] / 255.0
            colored_mask[mask > 0.5, 2] = color[2] / 255.0
            colored_mask[mask > 0.5, 3] = 0.4
            ax.imshow(colored_mask)

        # Draw keypoints + skeleton
        if "keypoints" in pred:
            kp = pred["keypoints"][i].cpu()  # (K, 3)
            kp_coords = kp[:, :2].numpy()
            kp_vis = kp[:, 2].numpy()

            # Skeleton lines
            for start, end in SKELETON:
                if start < len(kp_vis) and end < len(kp_vis):
                    if kp_vis[start] > 0.1 and kp_vis[end] > 0.1:
                        ax.plot(
                            [kp_coords[start, 0], kp_coords[end, 0]],
                            [kp_coords[start, 1], kp_coords[end, 1]],
                            color=color_norm, linewidth=2,
                        )

            # Keypoint dots
            for j in range(len(kp_coords)):
                if kp_vis[j] > 0.1:
                    ax.plot(
                        kp_coords[j, 0], kp_coords[j, 1], "o",
                        markersize=4, markerfacecolor="red",
                        markeredgecolor="white", markeredgewidth=0.5,
                    )

    fig.tight_layout()
    save_path = os.path.join(output_dir, "test_result.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nVisualization saved to: {save_path}")
