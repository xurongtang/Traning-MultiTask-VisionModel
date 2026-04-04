"""
COCO2017 Dataset for Multi-Task Training
==========================================
Dataset loader that provides:
    - Images
    - Bounding boxes + class labels (for RPN + Box head)
    - Instance segmentation masks (for Instance Segmentation head)
    - Keypoint annotations (for Keypoint Detection head)

Supports COCO2017 format with both instance segmentation and person
keypoint annotations loaded from separate annotation files.

Note:
    COCO provides instance segmentation annotations in 'instances_train2017.json'
    and keypoint annotations in 'person_keypoints_train2017.json'.
    This dataset merges both annotation types for multi-task training.
"""

import os
import torch
import torch.utils.data
import torchvision
from torchvision.io import read_image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple


class CocoMultiTaskDataset(torch.utils.data.Dataset):
    """
    COCO dataset for multi-task learning (Instance Segmentation + Keypoint Detection).

    Loads images with both instance segmentation masks and keypoint annotations.
    For images/instances without keypoint annotations, keypoints are filled with zeros.

    Args:
        img_root: Path to images directory (e.g., '.../train2017')
        ann_file: Path to instance annotation file (e.g., '.../instances_train2017.json')
        kp_ann_file: Path to keypoint annotation file (e.g., '.../person_keypoints_train2017.json')
        transforms: Optional transforms to apply
        num_keypoints: Number of keypoints (default: 17 for COCO)
    """

    def __init__(
        self,
        img_root: str,
        ann_file: str,
        kp_ann_file: str,
        transforms=None,
        num_keypoints: int = 17,
    ):
        super().__init__()

        self.img_root = img_root
        self.transforms = transforms
        self.num_keypoints = num_keypoints

        # Load instance segmentation annotations
        print(f"Loading instance annotations from {ann_file}...")
        self.coco_ins = COCO(ann_file)

        # Load keypoint annotations
        print(f"Loading keypoint annotations from {kp_ann_file}...")
        self.coco_kp = COCO(kp_ann_file)

        # Get all image IDs that have instance annotations
        self.img_ids = sorted(list(self.coco_ins.imgs.keys()))

        # Build mapping: img_id -> keypoint annotations (by ann_id)
        self.kp_by_image = {}
        for img_id in self.img_ids:
            kp_anns = self.coco_kp.loadAnns(
                self.coco_kp.getAnnIds(imgIds=img_id)
            )
            # Map annotation_id -> keypoint annotation
            kp_dict = {}
            for ann in kp_anns:
                if "keypoints" in ann:
                    kp_dict[ann["id"]] = ann
            self.kp_by_image[img_id] = kp_dict

        print(f"Dataset loaded: {len(self.img_ids)} images")

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_id = self.img_ids[idx]
        img_info = self.coco_ins.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_root, img_info["file_name"])

        # ── Debug: image loading ──────────────────────────────
        # print(f"\n[DEBUG] idx={idx}, img_id={img_id}, file={img_info['file_name']}")
        # print(f"[DEBUG] img_info: width={img_info['width']}, height={img_info['height']}")

        # Load image
        img = Image.open(img_path).convert("RGB")
        # print(f"[DEBUG] After PIL open: mode={img.mode}, size={img.size} (W×H)")

        # Get instance annotations for this image
        ann_ids = self.coco_ins.getAnnIds(imgIds=img_id)
        anns = self.coco_ins.loadAnns(ann_ids)

        # Get keypoint annotations for this image
        kp_dict = self.kp_by_image.get(img_id, {})

        # ── Debug: annotation counts ──────────────────────────
        n_crowd = sum(1 for a in anns if a.get("iscrowd", 0))
        # print(f"[DEBUG] Annotations: total={len(anns)}, crowd={n_crowd}, "
        #       f"with_kp={len(kp_dict)}")

        # Build target dict
        target = self._parse_annotations(anns, kp_dict, img_info)

        # ── Debug: parsed target shapes ───────────────────────
        # print(f"[DEBUG] Parsed target: "
        #       f"boxes={target['boxes'].shape}, "
        #       f"labels={target['labels'].shape}, "
        #       f"masks={target['masks'].shape}, "
        #       f"keypoints={target['keypoints'].shape}")

        # Apply transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            # After transform, img is still PIL Image
            # print(f"[DEBUG] After transform: PIL size={img.size} (W×H)")

        # Convert image to tensor
        if not isinstance(img, torch.Tensor):
            img = torchvision.transforms.functional.to_tensor(img)

        # ── Debug: final output ───────────────────────────────
        # print(f"[DEBUG] Final image tensor: shape={img.shape}, dtype={img.dtype}")
        # print(f"[DEBUG] Final target:")
        # for k, v in target.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"[DEBUG]   {k}: shape={v.shape}, dtype={v.dtype}")
        #     else:
        #         print(f"[DEBUG]   {k}: {v}")

        return img, target

    def _parse_annotations(
        self, anns: list, kp_dict: dict, img_info: dict
    ) -> Dict:
        """Parse COCO annotations into target format for Mask R-CNN."""
        w = img_info["width"]
        h = img_info["height"]

        boxes = []
        labels = []
        masks = []
        keypoints = []

        for ann in anns:
            # Skip crowd annotations
            if ann.get("iscrowd", 0):
                continue

            # Bounding box [x, y, w, h] → [x1, y1, x2, y2]
            bbox = ann["bbox"]
            x1, y1, bw, bh = bbox
            x2 = x1 + bw
            y2 = y1 + bh

            # Validate box
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(ann["category_id"])

            # Instance segmentation mask
            if "segmentation" in ann:
                rle = self.coco_ins.annToRLE(ann)
                mask = coco_mask.decode(rle)
                masks.append(mask)
            else:
                masks.append(np.zeros((h, w), dtype=np.uint8))

            # Keypoints (check if this annotation has keypoint data)
            ann_id = ann["id"]
            if ann_id in kp_dict and "keypoints" in kp_dict[ann_id]:
                kp = np.array(kp_dict[ann_id]["keypoints"], dtype=np.float32)
                kp = kp.reshape(-1, 3)  # (num_keypoints, 3) [x, y, visibility]
                keypoints.append(kp)
            else:
                # No keypoint annotation for this instance
                # Fill with zeros (visibility=0 means not labeled)
                keypoints.append(np.zeros((self.num_keypoints, 3), dtype=np.float32))

        # Handle empty annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, h, w), dtype=torch.uint8)
            keypoints = torch.zeros((0, self.num_keypoints, 3), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            keypoints = torch.as_tensor(np.array(keypoints), dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "keypoints": keypoints,
            "image_id": img_info["id"],
            "orig_size": (h, w),
        }

        return target


class CocoTransform:
    """
    Simple transforms for COCO multi-task training.
    Applies random horizontal flip and converts targets accordingly.
    """

    def __init__(self, train: bool = True):
        self.train = train

    def __call__(self, image: Image.Image, target: Dict):
        if self.train:
            # Random horizontal flip (50% probability)
            if torch.rand(1) < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                w = image.width

                # Flip boxes
                if "boxes" in target and len(target["boxes"]) > 0:
                    boxes = target["boxes"]
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                    target["boxes"] = boxes

                # Flip masks
                if "masks" in target and len(target["masks"]) > 0:
                    target["masks"] = target["masks"].flip(-1)

                # Flip keypoints
                if "keypoints" in target and len(target["keypoints"]) > 0:
                    keypoints = target["keypoints"]
                    keypoints[:, :, 0] = w - keypoints[:, :, 0]
                    target["keypoints"] = keypoints

        return image, target

def collate_fn(batch):
    """
    Keep images as a list of tensors instead of stacking them.
    Torchvision detection models require List[Tensor[C,H,W]] as input.
    """
    images, targets = zip(*batch)
    return list(images), list(targets)

def build_dataloaders(cfg, mode="train"):
    """
    Build train and val dataloaders.

    Args:
        cfg: Configuration object
        mode: "train", "val", or "both"

    Returns:
        If mode == "both": (train_loader, val_loader)
        If mode == "train": train_loader
        If mode == "val": val_loader
    """
    datasets = {}

    if mode in ("train", "both"):
        train_dataset = CocoMultiTaskDataset(
            img_root=cfg.get_full_train_img_dir(),
            ann_file=cfg.get_full_train_ann(),
            kp_ann_file=cfg.get_full_train_kp_ann(),
            transforms=CocoTransform(train=True),
            num_keypoints=cfg.num_keypoints,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=True if cfg.device == "cuda" else False,
        )
        datasets["train"] = train_loader

    if mode in ("val", "both"):
        val_dataset = CocoMultiTaskDataset(
            img_root=cfg.get_full_val_img_dir(),
            ann_file=cfg.get_full_val_ann(),
            kp_ann_file=cfg.get_full_val_kp_ann(),
            transforms=CocoTransform(train=False),
            num_keypoints=cfg.num_keypoints,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=True if cfg.device == "cuda" else False,
        )
        datasets["val"] = val_loader

    if mode == "both":
        return datasets["train"], datasets["val"]
    elif mode == "train":
        return datasets["train"]
    else:
        return datasets["val"]


if __name__ == "__main__":
    # Quick test
    from config import Config
    cfg = Config()

    # Check if COCO data exists
    import os
    if os.path.exists(cfg.data_root):
        dataset = CocoMultiTaskDataset(
            img_root=cfg.get_full_train_img_dir(),
            ann_file=cfg.get_full_train_ann(),
            kp_ann_file=cfg.get_full_train_kp_ann(),
            num_keypoints=17,
        )
        if len(dataset) > 0:
            img, target = dataset[0]
            print(f"Image shape: {img.shape}")
            print(f"Boxes: {target['boxes'].shape}")
            print(f"Labels: {target['labels'].shape}")
            print(f"Masks: {target['masks'].shape}")
            print(f"Keypoints: {target['keypoints'].shape}")
    else:
        print(f"COCO data not found at {cfg.data_root}")
        print("Please download COCO2017 dataset first.")