"""
Test for Visualization Engine
================================
Tests the visualization module with synthetic data (no COCO dataset needed).

Verifies:
    - VisualizationEngine initialization
    - _draw_instance_seg with mock predictions
    - _draw_keypoints with mock predictions
    - _make_grid layout
    - build_visualizer from config
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils.visualization import VisualizationEngine, build_visualizer, COCO_SKELETON


def test_engine_init():
    """Test VisualizationEngine initialization."""
    engine = VisualizationEngine(output_dir="./test_output/vis", num_samples=5)
    assert os.path.exists("./test_output/vis"), "Output dir should be created"
    assert engine.num_samples == 5
    assert engine.score_threshold == 0.5
    print("  [PASS] Engine initialization")


def test_tensor_to_pil():
    """Test image conversion helpers."""
    engine = VisualizationEngine(output_dir="./test_output/vis")
    
    # Create a random CHW tensor [0, 1]
    img_tensor = torch.rand(3, 200, 300)
    pil_img = engine._tensor_to_pil(img_tensor)
    assert isinstance(pil_img, Image.Image)
    assert pil_img.size == (300, 200)  # PIL: (W, H)
    print("  [PASS] tensor_to_pil conversion")


def test_resize_for_display():
    """Test image resizing for display."""
    engine = VisualizationEngine(output_dir="./test_output/vis", max_display_size=200)
    
    # Large image
    big_img = Image.new("RGB", (800, 600))
    resized = engine._resize_for_display(big_img)
    assert max(resized.size) <= 200, f"Max dimension should be <= 200, got {max(resized.size)}"
    
    # Small image (should not be enlarged)
    small_img = Image.new("RGB", (100, 80))
    same = engine._resize_for_display(small_img)
    assert same.size == (100, 80), "Small images should not be resized"
    print("  [PASS] resize_for_display")


def test_draw_instance_seg():
    """Test instance segmentation visualization with mock predictions."""
    engine = VisualizationEngine(
        output_dir="./test_output/vis",
        score_threshold=0.3,
        max_display_size=200,
    )
    
    # Synthetic image
    img_tensor = torch.rand(3, 400, 600)
    
    # Mock predictions (2 detections)
    pred = {
        "boxes": torch.tensor([[50.0, 50.0, 200.0, 300.0], [250.0, 100.0, 450.0, 350.0]]),
        "labels": torch.tensor([1, 2]),
        "scores": torch.tensor([0.9, 0.7]),
        "masks": torch.rand(2, 1, 400, 600),
    }
    
    result = engine._draw_instance_seg(img_tensor, pred)
    assert isinstance(result, Image.Image)
    assert result.size[0] > 0 and result.size[1] > 0
    print(f"  [PASS] draw_instance_seg → {result.size}")


def test_draw_instance_seg_empty():
    """Test instance seg with no predictions above threshold."""
    engine = VisualizationEngine(output_dir="./test_output/vis", score_threshold=0.99)
    
    img_tensor = torch.rand(3, 200, 300)
    pred = {
        "boxes": torch.tensor([[50.0, 50.0, 200.0, 300.0]]),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.3]),  # Below threshold
        "masks": torch.rand(1, 1, 200, 300),
    }
    
    result = engine._draw_instance_seg(img_tensor, pred)
    assert isinstance(result, Image.Image)
    print("  [PASS] draw_instance_seg (empty predictions)")


def test_draw_keypoints():
    """Test keypoint visualization with mock predictions."""
    engine = VisualizationEngine(
        output_dir="./test_output/vis",
        score_threshold=0.3,
        max_display_size=200,
    )
    
    img_tensor = torch.rand(3, 400, 600)
    
    # Mock predictions with 17 COCO keypoints
    num_kp = 17
    keypoints = torch.zeros(1, num_kp, 3)
    # Set some visible keypoints
    keypoints[0, 0, :] = torch.tensor([100.0, 50.0, 1.0])   # nose
    keypoints[0, 5, :] = torch.tensor([80.0, 150.0, 1.0])    # left_shoulder
    keypoints[0, 6, :] = torch.tensor([120.0, 150.0, 1.0])   # right_shoulder
    keypoints[0, 11, :] = torch.tensor([85.0, 280.0, 1.0])   # left_hip
    keypoints[0, 12, :] = torch.tensor([115.0, 280.0, 1.0])  # right_hip
    
    pred = {
        "boxes": torch.tensor([[50.0, 30.0, 200.0, 350.0]]),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.85]),
        "keypoints": keypoints,
    }
    
    result = engine._draw_keypoints(img_tensor, pred)
    assert isinstance(result, Image.Image)
    assert result.size[0] > 0 and result.size[1] > 0
    print(f"  [PASS] draw_keypoints → {result.size}")


def test_make_grid():
    """Test grid layout generation."""
    engine = VisualizationEngine(
        output_dir="./test_output/vis",
        grid_cols=3,
        max_display_size=100,
    )
    
    # Create 7 test images of different sizes
    images = [Image.new("RGB", (150 + i * 20, 120 + i * 10), color=(i * 30, 100, 200)) for i in range(7)]
    
    grid = engine._make_grid(images)
    assert isinstance(grid, Image.Image)
    # 3 cols, 3 rows (ceil(7/3)=3), cell=100, gap=4
    expected_w = 3 * 100 + 4 * 4  # 316
    expected_h = 3 * 100 + 4 * 4  # 316
    assert grid.size == (expected_w, expected_h), f"Expected ({expected_w}, {expected_h}), got {grid.size}"
    print(f"  [PASS] make_grid (7 images → 3x3 grid) → {grid.size}")


def test_make_grid_empty():
    """Test grid with empty image list."""
    engine = VisualizationEngine(output_dir="./test_output/vis")
    grid = engine._make_grid([])
    assert isinstance(grid, Image.Image)
    print("  [PASS] make_grid (empty list)")


def test_build_visualizer():
    """Test building visualizer from config."""
    cfg = Config()
    cfg.output_dir = "./test_output"
    cfg.vis_num_samples = 50
    cfg.vis_score_threshold = 0.6
    cfg.vis_max_detections = 5
    cfg.vis_grid_cols = 8
    cfg.vis_max_display_size = 256
    
    vis = build_visualizer(cfg)
    assert isinstance(vis, VisualizationEngine)
    assert vis.num_samples == 50
    assert vis.score_threshold == 0.6
    assert vis.max_detections == 5
    assert vis.grid_cols == 8
    assert vis.max_display_size == 256
    print("  [PASS] build_visualizer from config")


def test_full_visualize_save():
    """Test full visualization pipeline saving to disk."""
    engine = VisualizationEngine(
        output_dir="./test_output/vis_full",
        num_samples=3,
        max_display_size=150,
        grid_cols=3,
    )
    
    # Create a mock dataset
    class MockDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 5
        def __getitem__(self, idx):
            img = torch.rand(3, 300, 400)
            target = {
                "boxes": torch.tensor([[10.0, 10.0, 200.0, 250.0]]),
                "labels": torch.tensor([1]),
                "masks": torch.zeros(1, 300, 400, dtype=torch.uint8),
                "keypoints": torch.zeros(1, 17, 3),
                "image_id": idx,
                "orig_size": (300, 400),
            }
            return img, target
    
    # Create a mock model
    class MockModel(torch.nn.Module):
        def train(self, mode=True):
            return super().train(mode)
        def eval(self):
            return super().eval()
        def __call__(self, images, targets=None):
            return self.forward(images, targets)
        def forward(self, images, targets=None):
            results = []
            for img in images:
                h, w = img.shape[1], img.shape[2]
                results.append({
                    "boxes": torch.tensor([[20.0, 20.0, 180.0, 220.0]]),
                    "labels": torch.tensor([1]),
                    "scores": torch.tensor([0.88]),
                    "masks": torch.rand(1, 1, h, w),
                    "keypoints": torch.zeros(1, 17, 3),
                })
            return results
    
    model = MockModel()
    dataset = MockDataset()
    
    paths = engine.visualize_epoch(
        model=model,
        val_dataset=dataset,
        epoch=1,
        device=torch.device("cpu"),
    )
    
    assert "instance_seg" in paths, "Should save instance seg visualization"
    assert "keypoints" in paths, "Should save keypoint visualization"
    assert os.path.exists(paths["instance_seg"]), "File should exist on disk"
    assert os.path.exists(paths["keypoints"]), "File should exist on disk"
    
    # Verify the saved images are valid
    seg_img = Image.open(paths["instance_seg"])
    kp_img = Image.open(paths["keypoints"])
    assert seg_img.size[0] > 0
    assert kp_img.size[0] > 0
    
    print(f"  [PASS] Full pipeline: seg={seg_img.size}, kp={kp_img.size}")


def test_coco_skeleton():
    """Verify COCO skeleton connections are valid for 17 keypoints."""
    for start, end in COCO_SKELETON:
        assert 0 <= start < 17, f"Invalid start keypoint: {start}"
        assert 0 <= end < 17, f"Invalid end keypoint: {end}"
        assert start != end, f"Self-connection at {start}"
    print(f"  [PASS] COCO skeleton ({len(COCO_SKELETON)} connections, 17 keypoints)")


if __name__ == "__main__":
    print("=" * 50)
    print("Visualization Engine Tests")
    print("=" * 50)
    
    tests = [
        test_engine_init,
        test_tensor_to_pil,
        test_resize_for_display,
        test_draw_instance_seg,
        test_draw_instance_seg_empty,
        test_draw_keypoints,
        test_make_grid,
        test_make_grid_empty,
        test_build_visualizer,
        test_full_visualize_save,
        test_coco_skeleton,
    ]
    
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")
    
    # Cleanup test output
    # import shutil
    # if os.path.exists("./test_output"):
    #     shutil.rmtree("./test_output")