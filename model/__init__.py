"""
Multi-Task Model Components
=============================
- backbone: Shared feature extraction (ResNet-50 + FPN)
- instanceSeg_head: Instance segmentation mask prediction
- keyPoint_head: Keypoint detection heatmap prediction
- multitask_model: Full multi-task model assembly
"""

from model.backbone import Backbone, build_backbone
from model.instanceSeg_head import InstanceSegHead, build_instance_seg_head
from model.keyPoint_head import KeypointHead, build_keypoint_head
from model.multitask_model import MultiTaskModel, build_multitask_model