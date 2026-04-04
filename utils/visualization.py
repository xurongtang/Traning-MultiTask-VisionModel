"""
Visualization Engine for Multi-Task Inference Results
======================================================
Generates visual predictions for:
    - Instance Segmentation (masks + bounding boxes + labels)
    - Keypoint Detection (keypoints + skeletons)

Every N epochs during training, randomly samples images from the val set,
runs inference, and saves a grid of visualized predictions to disk.

Output format:
    output/
    └── visualizations/
        ├── epoch_010_instance_seg.png   # 10×10 grid of mask predictions
        ├── epoch_010_keypoints.png      # 10×10 grid of keypoint predictions
        ├── epoch_020_instance_seg.png
        ├── epoch_020_keypoints.png
        └── ...
"""

import os
import random
import numpy as np
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.ops import box_iou
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from typing import List, Dict, Optional, Tuple

# COCO person skeleton connections (17 keypoints)
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6),                                      # shoulders
    (5, 7), (7, 9),                              # left arm
    (6, 8), (8, 10),                             # right arm
    (5, 11), (6, 12),                            # torso
    (11, 12),                                     # hips
    (11, 13), (13, 15),                          # left leg
    (12, 14), (14, 16),                          # right leg
]

# Distinct colors for instances (up to 20)
INSTANCE_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (64, 0, 0), (0, 64, 0), (0, 0, 64), (192, 0, 0),
    (0, 192, 0), (0, 0, 192), (192, 192, 0), (64, 64, 64),
]


class VisualizationEngine:
    """
    Generates and saves visualization grids of model predictions.

    Args:
        output_dir: Directory to save visualization images
        num_samples: Number of random val images per task (default: 100)
        score_threshold: Minimum detection score to visualize (default: 0.5)
        max_detections: Max detections per image to draw (default: 10)
        grid_cols: Number of columns in the output grid (default: 10)
        max_display_size: Max image dimension for display (default: 300)
    """

    def __init__(
        self,
        output_dir: str = "./output/visualizations",
        num_samples: int = 100,
        score_threshold: float = 0.5,
        max_detections: int = 10,
        grid_cols: int = 10,
        max_display_size: int = 300,
    ):
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.grid_cols = grid_cols
        self.max_display_size = max_display_size

        os.makedirs(output_dir, exist_ok=True)

    @torch.no_grad()
    def visualize_epoch(
        self,
        model: torch.nn.Module,
        val_dataset: torch.utils.data.Dataset,
        epoch: int,
        device: torch.device,
    ) -> Dict[str, str]:
        """
        Run inference on random val images and save visualization grids.

        Generates two grids:
            1. Instance Segmentation: original + predicted masks + boxes
            2. Keypoint Detection: original + predicted keypoints + skeletons

        Args:
            model: The multi-task model
            val_dataset: Validation dataset (CocoMultiTaskDataset)
            epoch: Current epoch number
            device: Device for inference

        Returns:
            Dict with paths to saved visualization images:
                {'instance_seg': path, 'keypoints': path}
        """
        model.eval()

        # Randomly sample image indices
        total_images = len(val_dataset)
        num_samples = min(self.num_samples, total_images)
        indices = random.sample(range(total_images), num_samples)

        # Collect predictions
        all_images_for_seg = []
        all_images_for_kp = []

        for idx in indices:
            try:
                img_tensor, target = val_dataset[idx]
            except Exception:
                continue

            # Run inference
            img_batch = [img_tensor.to(device)]
            try:
                predictions = model(img_batch)
            except Exception:
                continue

            if not predictions or len(predictions) == 0:
                continue

            pred = predictions[0]

            # ── Instance Segmentation Visualization ──
            seg_img = self._draw_instance_seg(img_tensor, pred)
            all_images_for_seg.append(seg_img)

            # ── Keypoint Detection Visualization ──
            kp_img = self._draw_keypoints(img_tensor, pred)
            all_images_for_kp.append(kp_img)

        # Build and save grids
        saved_paths = {}

        if all_images_for_seg:
            grid = self._make_grid(all_images_for_seg)
            path = os.path.join(self.output_dir, f"epoch_{epoch:03d}_instance_seg.png")
            grid.save(path)
            saved_paths["instance_seg"] = path
            print(f"  Instance Seg visualization saved: {path} ({len(all_images_for_seg)} images)")

        if all_images_for_kp:
            grid = self._make_grid(all_images_for_kp)
            path = os.path.join(self.output_dir, f"epoch_{epoch:03d}_keypoints.png")
            grid.save(path)
            saved_paths["keypoints"] = path
            print(f"  Keypoint visualization saved: {path} ({len(all_images_for_kp)} images)")

        return saved_paths

    def _draw_instance_seg(
        self, img_tensor: torch.Tensor, pred: Dict
    ) -> Image.Image:
        """
        Draw instance segmentation results on a single image.

        Shows: predicted bounding boxes, segmentation masks, class labels, scores.
        """
        # Convert image for display
        img_uint8 = self._tensor_to_uint8(img_tensor)
        img_pil = self._tensor_to_pil(img_tensor)
        img_display = self._resize_for_display(img_pil)

        scores = pred.get("scores", torch.tensor([]))
        if len(scores) == 0:
            return img_display

        # Filter by score threshold
        keep = scores > self.score_threshold
        keep = keep.nonzero(as_tuple=True)[0][:self.max_detections]

        if len(keep) == 0:
            return img_display

        # Scale factor for resizing
        orig_w, orig_h = img_pil.size
        disp_w, disp_h = img_display.size
        scale_x = disp_w / orig_w
        scale_y = disp_h / orig_h

        # Draw using matplotlib
        fig, ax = plt.subplots(1, figsize=(disp_w / 100, disp_h / 100), dpi=100)
        ax.imshow(img_display)
        ax.axis("off")

        boxes = pred["boxes"][keep].cpu()
        masks = pred.get("masks", None)
        labels = pred.get("labels", torch.tensor([0] * len(keep)))
        scores_kept = scores[keep].cpu()

        # Draw masks
        if masks is not None and len(masks) > 0:
            masks_kept = masks[keep].cpu()
            for i, mask in enumerate(masks_kept):
                color = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
                mask_np = mask[0].numpy()
                # Resize mask to display size
                mask_pil = Image.fromarray((mask_np > 0.5).astype(np.uint8) * 255)
                mask_pil = mask_pil.resize((disp_w, disp_h), Image.NEAREST)
                mask_np = np.array(mask_pil) / 255.0

                # Overlay mask with transparency
                colored_mask = np.zeros((*mask_np.shape, 4))
                colored_mask[mask_np > 0.5, 0] = color[0] / 255.0
                colored_mask[mask_np > 0.5, 1] = color[1] / 255.0
                colored_mask[mask_np > 0.5, 2] = color[2] / 255.0
                colored_mask[mask_np > 0.5, 3] = 0.4
                ax.imshow(colored_mask)

        # Draw boxes and labels
        for i in range(len(boxes)):
            color = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
            x1, y1, x2, y2 = boxes[i].tolist()
            # Scale to display size
            x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y

            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor=(*[c / 255.0 for c in color], 1.0),
                facecolor="none",
            )
            ax.add_patch(rect)

            label_id = labels[i].item() if i < len(labels) else 0
            score = scores_kept[i].item()
            ax.text(
                x1, y1 - 2, f"{label_id}:{score:.2f}",
                fontsize=5, color="white",
                bbox=dict(boxstyle="round,pad=0.1", facecolor=(*[c / 255.0 for c in color], 0.7), alpha=0.8),
            )

        fig.tight_layout(pad=0)
        fig.canvas.draw()

        # Convert to PIL
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        result = Image.frombytes("RGBA", (w, h), buf).convert("RGB")
        plt.close(fig)

        return result

    def _draw_keypoints(
        self, img_tensor: torch.Tensor, pred: Dict
    ) -> Image.Image:
        """
        Draw keypoint detection results on a single image.

        Shows: predicted bounding boxes, keypoints as dots, skeleton connections.
        """
        img_pil = self._tensor_to_pil(img_tensor)
        img_display = self._resize_for_display(img_pil)

        scores = pred.get("scores", torch.tensor([]))
        if len(scores) == 0:
            return img_display

        # Filter by score threshold
        keep = scores > self.score_threshold
        keep = keep.nonzero(as_tuple=True)[0][:self.max_detections]

        if len(keep) == 0:
            return img_display

        # Scale factor
        orig_w, orig_h = img_pil.size
        disp_w, disp_h = img_display.size
        scale_x = disp_w / orig_w
        scale_y = disp_h / orig_h

        # Draw using matplotlib
        fig, ax = plt.subplots(1, figsize=(disp_w / 100, disp_h / 100), dpi=100)
        ax.imshow(img_display)
        ax.axis("off")

        boxes = pred["boxes"][keep].cpu()
        keypoints = pred.get("keypoints", None)
        scores_kept = scores[keep].cpu()

        if keypoints is None or len(keypoints) == 0:
            plt.close(fig)
            return img_display

        keypoints_kept = keypoints[keep].cpu()

        for i in range(len(keypoints_kept)):
            color = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
            kp = keypoints_kept[i]  # (K, 3): x, y, visibility

            # Draw bounding box (lighter)
            x1, y1, x2, y2 = boxes[i].tolist()
            x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1, edgecolor=(*[c / 255.0 for c in color], 0.6),
                facecolor="none", linestyle="--",
            )
            ax.add_patch(rect)

            # Draw skeleton connections
            kp_coords = kp[:, :2].numpy()  # x, y
            kp_vis = kp[:, 2].numpy()      # visibility/confidence

            # Scale keypoints to display size
            kp_scaled = kp_coords.copy()
            kp_scaled[:, 0] *= scale_x
            kp_scaled[:, 1] *= scale_y

            for start, end in COCO_SKELETON:
                if start < len(kp_vis) and end < len(kp_vis):
                    if kp_vis[start] > 0.1 and kp_vis[end] > 0.1:
                        ax.plot(
                            [kp_scaled[start, 0], kp_scaled[end, 0]],
                            [kp_scaled[start, 1], kp_scaled[end, 1]],
                            color=(*[c / 255.0 for c in color], 0.7),
                            linewidth=1.0,
                        )

            # Draw keypoints as dots
            for j in range(len(kp_scaled)):
                if kp_vis[j] > 0.1:
                    ax.plot(
                        kp_scaled[j, 0], kp_scaled[j, 1],
                        "o",
                        markersize=2,
                        markerfacecolor=(1, 0, 0),
                        markeredgecolor="white",
                        markeredgewidth=0.3,
                    )

        fig.tight_layout(pad=0)
        fig.canvas.draw()

        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        result = Image.frombytes("RGBA", (w, h), buf).convert("RGB")
        plt.close(fig)

        return result

    def _make_grid(self, images: List[Image.Image]) -> Image.Image:
        """
        Arrange a list of images into a grid layout.

        Automatically computes rows based on grid_cols.
        Adds a small gap between images.
        """
        if not images:
            return Image.new("RGB", (100, 100), (255, 255, 255))

        # Normalize all images to same size
        cell_w = self.max_display_size
        cell_h = self.max_display_size
        gap = 4  # pixels between images

        images_resized = []
        for img in images:
            img_resized = img.resize((cell_w, cell_h), Image.LANCZOS)
            images_resized.append(img_resized)

        cols = self.grid_cols
        rows = (len(images_resized) + cols - 1) // cols

        # Create canvas
        canvas_w = cols * cell_w + (cols + 1) * gap
        canvas_h = rows * cell_h + (rows + 1) * gap
        canvas = Image.new("RGB", (canvas_w, canvas_h), (240, 240, 240))

        for i, img in enumerate(images_resized):
            row = i // cols
            col = i % cols
            x = gap + col * (cell_w + gap)
            y = gap + row * (cell_h + gap)
            canvas.paste(img, (x, y))

        return canvas

    @staticmethod
    def _tensor_to_uint8(img: torch.Tensor) -> torch.Tensor:
        """Convert [0,1] float tensor to [0,255] uint8 tensor."""
        return (img * 255).clamp(0, 255).to(torch.uint8)

    @staticmethod
    def _tensor_to_pil(img: torch.Tensor) -> Image.Image:
        """Convert a CHW float tensor to PIL Image."""
        return torchvision.transforms.functional.to_pil_image(img)

    def _resize_for_display(self, img_pil: Image.Image) -> Image.Image:
        """Resize image so max dimension <= max_display_size, preserving aspect ratio."""
        w, h = img_pil.size
        if max(w, h) <= self.max_display_size:
            return img_pil
        ratio = self.max_display_size / max(w, h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        return img_pil.resize((new_w, new_h), Image.LANCZOS)


def build_visualizer(cfg) -> VisualizationEngine:
    """Build VisualizationEngine from config."""
    return VisualizationEngine(
        output_dir=os.path.join(cfg.output_dir, "visualizations"),
        num_samples=cfg.vis_num_samples,
        score_threshold=cfg.vis_score_threshold,
        max_detections=cfg.vis_max_detections,
        grid_cols=cfg.vis_grid_cols,
        max_display_size=cfg.vis_max_display_size,
    )