"""
Multi-Task Training Configuration
===================================
Configuration for joint Instance Segmentation + Keypoint Detection on COCO2017.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import os


@dataclass
class Config:
    """Multi-task training configuration."""

    # ── Data ────────────────────────────────────────────────
    data_root: str = "/home/hc/XuRongTangProj/coco_dataset"
    train_ann: str = "annotations/instances_train2017.json"
    val_ann: str = "annotations/instances_val2017.json"
    train_kp_ann: str = "annotations/person_keypoints_train2017.json"
    val_kp_ann: str = "annotations/person_keypoints_val2017.json"
    train_images: str = "train2017"
    val_images: str = "val2017"

    batch_size: int = 4
    num_workers: int = 10  # 0 for debug on CPU

    # ── Model ───────────────────────────────────────────────
    backbone_name: str = "resnet50"
    pretrained_backbone: bool = True
    trainable_layers: int = 3  # Number of trainable backbone layers
    num_classes: int = 2   # Person-only: 1 class (person) + 1 background
    num_keypoints: int = 17  # COCO person keypoints
    keypoint_names: tuple = (
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    )
    # Image size
    min_size: int = 800
    max_size: int = 1333

    # ── Training ────────────────────────────────────────────
    learning_rate: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lr_scheduler: str = "step"  # "step" or "cosine"
    lr_step_size: int = 3
    lr_gamma: float = 0.1
    num_epochs: int = 12
    warmup_epochs: int = 1

    # ── Multi-Task Loss Weights ─────────────────────────────
    # These weights scale each task's contribution to the total loss
    loss_weight_rpn: float = 1.0       # RPN classification + regression
    loss_weight_box: float = 1.0       # Box classification + regression
    loss_weight_mask: float = 1.0      # Instance segmentation mask loss
    loss_weight_keypoint: float = 1.0  # Keypoint detection loss

    # ── Logging ─────────────────────────────────────────────
    log_dir: str = "./runs"
    log_interval: int = 10  # Log every N batches
    save_interval: int = 1  # Save checkpoint every N epochs
    output_dir: str = "./output"

    # ── Device ──────────────────────────────────────────────
    device: str = "cuda:1"  # "cpu" for local debug, "cuda" for server

    # ── Visualization ────────────────────────────────────────
    vis_interval: int = 10            # Visualize every N epochs
    vis_num_samples: int = 100        # Number of random val images per task
    vis_score_threshold: float = 0.5  # Min detection score to draw
    vis_max_detections: int = 10      # Max detections per image
    vis_grid_cols: int = 10           # Grid columns (10x10 = 100 images)
    vis_max_display_size: int = 300   # Max image dimension for grid cells

    # ── Misc ────────────────────────────────────────────────
    seed: int = 42
    resume: Optional[str] = None  # Path to checkpoint for resuming training

    def get_full_train_img_dir(self):
        return os.path.join(self.data_root, self.train_images)

    def get_full_val_img_dir(self):
        return os.path.join(self.data_root, self.val_images)

    def get_full_train_ann(self):
        return os.path.join(self.data_root, self.train_ann)

    def get_full_val_ann(self):
        return os.path.join(self.data_root, self.val_ann)

    def get_full_train_kp_ann(self):
        return os.path.join(self.data_root, self.train_kp_ann)

    def get_full_val_kp_ann(self):
        return os.path.join(self.data_root, self.val_kp_ann)