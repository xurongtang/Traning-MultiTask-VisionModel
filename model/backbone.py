"""
Shared Backbone with Feature Pyramid Network (FPN)
====================================================
ResNet-50 backbone with FPN for multi-scale feature extraction.
Shared by both Instance Segmentation and Keypoint Detection tasks.

Architecture:
    Input → ResNet-50 → C2, C3, C4, C5
                         ↓
              FPN → P2, P3, P4, P5, P6

The FPN produces feature pyramids at multiple scales, enabling
detection of objects at various sizes - critical for both
instance segmentation and keypoint detection.
"""

import torch
import torch.nn as nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


class Backbone(nn.Module):
    """
    ResNet + FPN backbone for multi-task feature extraction.

    The backbone extracts multi-scale features that are shared between
    the instance segmentation and keypoint detection heads.

    Args:
        backbone_name: Name of the backbone architecture (e.g., 'resnet50')
        pretrained: Whether to use ImageNet pretrained weights
        trainable_layers: Number of trainable (non-frozen) backbone layers (0-5)
        returned_layers: Which ResNet layers to use for FPN (default: [1,2,3,4] → C2-C5)
        extra_blocks: Extra FPN level (default: LastLevelMaxPool → P6)
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        trainable_layers: int = 3,
        returned_layers=None,
        extra_blocks=None,
    ):
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        if returned_layers is None:
            returned_layers = [1, 2, 3, 4]

        # Build ResNet-FPN backbone using torchvision utilities
        self.backbone = resnet_fpn_backbone(
            backbone_name=backbone_name,
            weights="DEFAULT" if pretrained else None,
            trainable_layers=trainable_layers,
            returned_layers=returned_layers,
            extra_blocks=extra_blocks,
        )

        # Feature map channels (same for all FPN levels)
        self.out_channels = 256

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through the backbone.

        Args:
            x: Input images tensor of shape (B, 3, H, W)

        Returns:
            Dictionary of feature maps:
                - '0': P2 features (1/4 scale)
                - '1': P3 features (1/8 scale)
                - '2': P4 features (1/16 scale)
                - '3': P5 features (1/32 scale)
                - 'pool': P6 features (1/64 scale, from LastLevelMaxPool)
        """
        features = self.backbone(x)
        return features

    def freeze_batch_norm(self):
        """Freeze all BatchNorm layers in the backbone."""
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False


def build_backbone(cfg) -> Backbone:
    """Build backbone from configuration."""
    backbone = Backbone(
        backbone_name=cfg.backbone_name,
        pretrained=cfg.pretrained_backbone,
        trainable_layers=cfg.trainable_layers,
    )
    return backbone


if __name__ == "__main__":
    # Quick test
    backbone = Backbone(backbone_name="resnet50", pretrained=False)
    x = torch.randn(1, 3, 800, 800)
    features = backbone(x)
    print("Backbone output feature maps:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
    print(f"Out channels: {backbone.out_channels}")