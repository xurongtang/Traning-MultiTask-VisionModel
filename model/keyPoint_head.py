"""
Keypoint Detection Head
=========================
Keypoint heatmap prediction head for human pose estimation.

Architecture:
    RoI Feature (14×14) → Conv layers → Deconv → 56×56 Heatmaps (17 keypoints)

This head predicts a heatmap for each of the 17 COCO person keypoints,
enabling human pose estimation within each detected instance.

COCO 17 Keypoints:
    0: nose           1: left_eye      2: right_eye
    3: left_ear       4: right_ear     5: left_shoulder
    6: right_shoulder 7: left_elbow    8: right_elbow
    9: left_wrist    10: right_wrist  11: left_hip
    12: right_hip    13: left_knee    14: right_knee
    15: left_ankle   16: right_ankle

Key Components:
    - keypoint_roi_pool: RoIAlign for keypoint feature extraction
    - keypoint_layers: Conv layers for feature processing
    - keypoint_upsample: Deconv layers to upsample heatmaps
    - keypoint_logits: Final prediction (17 keypoint heatmaps)
"""

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNHeads
from torchvision.ops import MultiScaleRoIAlign


class KeypointHead(nn.Module):
    """
    Keypoint detection head that predicts heatmaps for each RoI.

    Uses RoIAlign to extract features from FPN, then processes them
    through conv + deconv layers to produce per-keypoint heatmaps.

    Args:
        in_channels: Number of input channels from FPN (default: 256)
        roi_output_size: RoIAlign output size (default: 14)
        num_keypoints: Number of keypoints (default: 17 for COCO)
        hidden_channels: Number of channels in conv layers (default: 256)
        num_hidden_layers: Number of hidden conv layers (default: 8)
        keypoint_heatmap_size: Output heatmap size (default: 56×56)
        featmap_names: Feature map names for RoI pooling
    """

    def __init__(
        self,
        in_channels: int = 256,
        roi_output_size: int = 14,
        num_keypoints: int = 17,
        hidden_channels: int = 256,
        num_hidden_layers: int = 8,
        keypoint_heatmap_size: int = 56,
        featmap_names: tuple = ("0", "1", "2", "3"),
    ):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.keypoint_heatmap_size = keypoint_heatmap_size

        # RoI Align for keypoint features
        self.keypoint_roi_pool = MultiScaleRoIAlign(
            featmap_names=list(featmap_names),
            output_size=roi_output_size,
            sampling_ratio=2,
        )

        # Keypoint feature extraction using torchvision's KeypointRCNNHeads
        # 8 conv layers with 256 channels each
        self.keypoint_heads = KeypointRCNNHeads(
            in_channels=in_channels,
            layers=[hidden_channels] * num_hidden_layers,
            num_keypoints=num_keypoints,
        )

        # Note: KeypointRCNNHeads already includes the final prediction layer
        # so we don't need a separate logits layer

    @property
    def resolution(self):
        return self.keypoint_heatmap_size

    def forward(
        self,
        features: dict,
        boxes: list,
        image_shapes: list,
    ) -> Tensor:
        """
        Predict keypoint heatmaps for each RoI.

        Args:
            features: FPN feature maps dict from backbone
            boxes: List of box tensors, one per image, shape (N_i, 4)
            image_shapes: List of original image shapes (H, W)

        Returns:
            keypoints: Tensor of shape (total_rois, num_keypoints, H_k, W_k)
                       where H_k = W_k = keypoint_heatmap_size (56)
        """
        if not boxes or all(len(b) == 0 for b in boxes):
            device = next(self.parameters()).device
            return torch.empty(
                0, self.num_keypoints, self.resolution, self.resolution, device=device
            )

        # Create batch-indexed boxes for RoIAlign
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
                0, self.num_keypoints, self.resolution, self.resolution, device=device
            )

        all_boxes = torch.cat(box_list, dim=0)

        # RoI Align
        x = self.keypoint_roi_pool(features, [all_boxes], image_shapes)

        # Keypoint prediction (includes feature extraction + final conv)
        keypoints = self.keypoint_heads(x)

        return keypoints


def build_keypoint_head(cfg) -> KeypointHead:
    """Build keypoint detection head from configuration."""
    return KeypointHead(
        in_channels=256,
        roi_output_size=14,
        num_keypoints=cfg.num_keypoints,
        hidden_channels=256,
        num_hidden_layers=8,
        keypoint_heatmap_size=56,
        featmap_names=("0", "1", "2", "3"),
    )


if __name__ == "__main__":
    # Quick test
    head = KeypointHead(in_channels=256, num_keypoints=17)
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
    keypoints = head(features, boxes, image_shapes)
    print(f"Keypoint output shape: {keypoints.shape}")  # Expected: (3, 17, 56, 56)