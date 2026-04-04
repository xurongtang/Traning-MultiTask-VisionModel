"""
Instance Segmentation Head
============================
Mask prediction head for instance segmentation.

Architecture:
    RoI Feature (14×14) → Conv layers → Deconv → 28×28 Mask per class

This head predicts a binary segmentation mask for each detected instance,
providing pixel-level object boundaries.

The mask head is applied after the box head has predicted bounding boxes,
and generates a mask for each RoI.

Key Components:
    - mask_roi_pool: RoIAlign to extract fixed-size features from FPN
    - mask_layers: Series of conv layers for mask feature extraction
    - mask_fcn_logits: Final prediction layer (num_classes masks)
"""

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads
from torchvision.ops import MultiScaleRoIAlign


class InstanceSegHead(nn.Module):
    """
    Instance Segmentation head that predicts binary masks for each RoI.

    Uses RoIAlign to extract features from FPN, then processes them
    through conv + deconv layers to produce per-class segmentation masks.

    Args:
        in_channels: Number of input channels from FPN (default: 256)
        roi_output_size: RoIAlign output size (default: 14)
        num_classes: Number of classes including background
        hidden_layers: Number of hidden conv layers (default: 4)
        hidden_channels: Number of channels in hidden layers (default: 256)
        roi_spatial_scale: RoIAlign spatial scale (handled automatically by MultiScaleRoIAlign)
        featmap_names: Feature map names to use for RoI pooling
    """

    def __init__(
        self,
        in_channels: int = 256,
        roi_output_size: int = 14,
        num_classes: int = 91,
        hidden_layers: int = 4,
        hidden_channels: int = 256,
        featmap_names: tuple = ("0", "1", "2", "3"),
        image_size: int = 800,
    ):
        super().__init__()

        self.num_classes = num_classes

        # RoI Align - pools features from multiple FPN levels
        self.mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=list(featmap_names),
            output_size=roi_output_size,
            sampling_ratio=2,
        )

        # Mask feature extraction head (Conv layers)
        self.mask_heads = MaskRCNNHeads(
            in_channels=in_channels,
            layers=[hidden_channels] * hidden_layers,
            dilation=1,
            num_fcs=0,
        )

        # Upsample: 14×14 → 28×28 using transposed convolution
        self.mask_upsample = nn.ConvTranspose2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=2,
            stride=2,
        )

        # Final prediction: one 28×28 mask per class
        self.mask_fcn_logits = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
        )

        self._image_size = image_size

    def forward(
        self,
        features: dict,
        boxes: list,
        image_shapes: list,
    ) -> Tensor:
        """
        Predict instance segmentation masks for each RoI.

        Args:
            features: FPN feature maps dict from backbone
            boxes: List of box tensors, one per image, shape (N_i, 4)
            image_shapes: List of original image shapes (H, W)

        Returns:
            masks: Tensor of shape (total_rois, num_classes, 28, 28)
        """
        if not boxes or all(len(b) == 0 for b in boxes):
            # No boxes, return empty tensor
            device = next(self.parameters()).device
            return torch.empty(
                0, self.num_classes, 28, 28, device=device
            )

        # Pool features from multi-scale FPN
        # Create a batch index for each box
        box_list = []
        for idx, box in enumerate(boxes):
            if len(box) > 0:
                batch_idx = torch.full(
                    (len(box), 1), idx, dtype=torch.float32, device=box.device
                )
                box_with_idx = torch.cat([batch_idx, box], dim=1)
                box_list.append(box_with_idx)

        if not box_list:
            device = next(self.parameters()).device
            return torch.empty(
                0, self.num_classes, 28, 28, device=device
            )

        all_boxes = torch.cat(box_list, dim=0)

        # RoI Align
        x = self.mask_roi_pool(features, [all_boxes], image_shapes)

        # Mask feature extraction
        x = self.mask_heads(x)

        # Upsample 14×14 → 28×28
        x = self.mask_upsample(x)
        x = torch.relu(x)

        # Predict masks
        masks = self.mask_fcn_logits(x)

        return masks


def build_instance_seg_head(cfg) -> InstanceSegHead:
    """Build instance segmentation head from configuration."""
    return InstanceSegHead(
        in_channels=256,
        roi_output_size=14,
        num_classes=cfg.num_classes,
        hidden_layers=4,
        hidden_channels=256,
        featmap_names=("0", "1", "2", "3"),
        image_size=cfg.min_size,
    )


if __name__ == "__main__":
    # Quick test
    head = InstanceSegHead(in_channels=256, num_classes=91)
    features = {
        "0": torch.randn(2, 256, 200, 200),
        "1": torch.randn(2, 256, 100, 100),
        "2": torch.randn(2, 256, 50, 50),
        "3": torch.randn(2, 256, 25, 25),
    }
    boxes = [
        torch.tensor([[10, 10, 100, 100], [50, 50, 200, 200]], dtype=torch.float32),
        torch.tensor([[20, 20, 150, 150]], dtype=torch.float32),
    ]
    image_shapes = [(800, 800), (800, 600)]
    masks = head(features, boxes, image_shapes)
    print(f"Mask output shape: {masks.shape}")  # Expected: (3, 91, 28, 28)