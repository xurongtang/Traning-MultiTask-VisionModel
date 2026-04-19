"""
测试脚本：加载训练好的 .pth 权重并对测试图片进行推理
============================================================
功能：
    1. 构建 MultiTaskModel 并加载 checkpoint 权重
    2. 读取测试图片并执行推理
    3. 打印检测结果（边界框、置信度、关键点）
    4. 可视化并保存结果到 test_output/
"""

import os
import torch
import cv2
import numpy as np
import torchvision.transforms.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import Config
from model.multitask_model import build_multitask_model

# ── 路径配置 ──────────────────────────────────────────────
pth_path = "/Users/xurongtang/workerFolder/MultiTaskTraining/asset/0418_best_model.pth"
test_path = "/Users/xurongtang/workerFolder/MultiTaskTraining/asset/image_test.png"
output_dir = "/Users/xurongtang/workerFolder/MultiTaskTraining/test_output"

# ── COCO 人物骨架连接定义 ──────────────────────────────────
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

INSTANCE_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
]


def load_checkpoint(model, ckpt_path, device):
    """加载 checkpoint 权重到模型中。"""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint 文件未找到: {ckpt_path}")

    print(f"[INFO] 正在加载 checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 处理不同的 checkpoint 格式
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "?")
        best_loss = checkpoint.get("best_loss", "?")
        print(f"  Epoch: {epoch}, Best Loss: {best_loss}")
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # 加载权重（strict=False 允许部分匹配）
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print("[INFO] Checkpoint 加载成功！\n")


def load_image(image_path, device):
    """加载图片并转为模型输入张量。"""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"测试图片未找到: {image_path}")

    # cv2 读取 BGR -> RGB
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"无法读取图片: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # 转为 float32 张量 [C, H, W], 值域 [0, 1]
    img_tensor = F.to_tensor(img_rgb).to(device)

    print(f"[INFO] 图片加载成功: {image_path}")
    print(f"  原始尺寸: {img_rgb.shape[1]}x{img_rgb.shape[0]}")
    print(f"  张量尺寸: {img_tensor.shape}\n")

    return img_rgb, img_tensor


def run_inference(model, img_tensor):
    """运行模型推理。"""
    model.eval()
    with torch.no_grad():
        results = model([img_tensor])
    return results[0]


def print_predictions(pred):
    """打印检测结果摘要。"""
    scores = pred["scores"]
    print("=" * 60)
    print("检测结果摘要")
    print("=" * 60)
    print(f"  Prediction keys: {list(pred.keys())}")
    print(f"  检测到的实例数: {len(scores)}")
    print(f"  包含关键点: {'keypoints' in pred}")
    print(f"  包含掩码: {'masks' in pred}")

    if len(scores) == 0:
        print("\n[WARN] 未检测到任何目标！模型可能需要更多训练或更换测试图片。")
        return

    # 打印 Top-5 检测
    topk = min(5, len(scores))
    top_scores, top_indices = scores.topk(topk)
    print(f"\n  Top-{topk} 检测结果:")
    print("-" * 60)
    for rank, (score, idx) in enumerate(zip(top_scores, top_indices)):
        label = pred["labels"][idx].item()
        box = pred["boxes"][idx].tolist()
        print(f"  [{rank}] label={label}, score={score:.4f}, "
              f"box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")

        if "keypoints" in pred:
            kp = pred["keypoints"][idx]  # (K, 3)
            visible = kp[:, 2] > 0.1
            num_visible = visible.sum().item()
            print(f"       关键点: {num_visible}/{kp.shape[0]} 可见")

            # 打印每个可见关键点的名称和坐标
            for j in range(kp.shape[0]):
                if kp[j, 2] > 0.1:
                    x, y, v = kp[j].tolist()
                    print(f"         {KEYPOINT_NAMES[j]:>16s}: ({x:.1f}, {y:.1f}), vis={v:.2f}")

        if "masks" in pred:
            mask = pred["masks"][idx, 0]
            mask_area = (mask > 0.5).sum().item()
            print(f"       掩码像素数: {mask_area}")
    print()


def visualize_predictions(img_rgb, pred, save_dir, score_threshold=0.5):
    """可视化并保存检测结果。"""
    os.makedirs(save_dir, exist_ok=True)
    scores = pred["scores"]
    keep = scores > score_threshold
    keep_indices = keep.nonzero(as_tuple=True)[0]

    if len(keep_indices) == 0:
        print(f"[WARN] 没有置信度 > {score_threshold} 的检测结果，尝试降低阈值...")
        if len(scores) > 0:
            top_score = scores.max().item()
            print(f"  最高置信度: {top_score:.4f}")
            # 如果最高置信度都低于阈值，取 top-1
            keep_indices = scores.topk(min(3, len(scores))).indices
        else:
            print("  无检测结果可可视化。")
            return

    # ── 绘制实例分割 + 关键点 ──────────────────────────────
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(img_rgb)
    ax.set_title("Instance Segmentation + Keypoints Detection", fontsize=14)
    ax.axis("off")

    for i in keep_indices:
        i = i.item() if isinstance(i, torch.Tensor) else i
        color = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
        color_norm = tuple(c / 255.0 for c in color)

        # 绘制边界框
        x1, y1, x2, y2 = pred["boxes"][i].tolist()
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color_norm, facecolor="none",
        )
        ax.add_patch(rect)

        label_id = pred["labels"][i].item()
        score_val = pred["scores"][i].item()
        ax.text(
            x1, y1 - 5,
            f"cls={label_id} score={score_val:.2f}",
            fontsize=9, color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=(*color_norm, 0.7)),
        )

        # 绘制实例分割掩码
        if "masks" in pred:
            mask = pred["masks"][i, 0].cpu().numpy()
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask > 0.5, 0] = color[0] / 255.0
            colored_mask[mask > 0.5, 1] = color[1] / 255.0
            colored_mask[mask > 0.5, 2] = color[2] / 255.0
            colored_mask[mask > 0.5, 3] = 0.4
            ax.imshow(colored_mask)

        # 绘制关键点 + 骨架
        if "keypoints" in pred:
            kp = pred["keypoints"][i].cpu()  # (K, 3)
            kp_coords = kp[:, :2].numpy()
            kp_vis = kp[:, 2].numpy()

            # 骨架连线
            for start, end in SKELETON:
                if start < len(kp_vis) and end < len(kp_vis):
                    if kp_vis[start] > 0.1 and kp_vis[end] > 0.1:
                        ax.plot(
                            [kp_coords[start, 0], kp_coords[end, 0]],
                            [kp_coords[start, 1], kp_coords[end, 1]],
                            color=color_norm, linewidth=2,
                        )

            # 关键点圆点
            for j in range(len(kp_coords)):
                if kp_vis[j] > 0.1:
                    ax.plot(
                        kp_coords[j, 0], kp_coords[j, 1], "o",
                        markersize=5, markerfacecolor="red",
                        markeredgecolor="white", markeredgewidth=0.5,
                    )

    fig.tight_layout()
    save_path = os.path.join(save_dir, "test_savepth_result.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] 可视化结果已保存到: {save_path}")

    # ── 单独保存掩码图 ─────────────────────────────────────
    if "masks" in pred and len(keep_indices) > 0:
        fig_mask, axes = plt.subplots(1, len(keep_indices), figsize=(6 * len(keep_indices), 6))
        if len(keep_indices) == 1:
            axes = [axes]

        for ax_i, idx in enumerate(keep_indices):
            idx = idx.item() if isinstance(idx, torch.Tensor) else idx
            mask = pred["masks"][idx, 0].cpu().numpy()
            axes[ax_i].imshow(mask, cmap="gray")
            score_val = pred["scores"][idx].item()
            axes[ax_i].set_title(f"Mask #{idx} (score={score_val:.3f})")
            axes[ax_i].axis("off")

        fig_mask.tight_layout()
        mask_save_path = os.path.join(save_dir, "test_savepth_masks.png")
        fig_mask.savefig(mask_save_path, dpi=150, bbox_inches="tight")
        plt.close(fig_mask)
        print(f"[INFO] 掩码结果已保存到: {mask_save_path}")

    # ── 单独保存关键点图 ───────────────────────────────────
    if "keypoints" in pred and len(keep_indices) > 0:
        fig_kp, ax_kp = plt.subplots(1, figsize=(12, 12))
        ax_kp.imshow(img_rgb)
        ax_kp.set_title("Keypoints Detection", fontsize=14)
        ax_kp.axis("off")

        for i in keep_indices:
            i = i.item() if isinstance(i, torch.Tensor) else i
            color = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
            color_norm = tuple(c / 255.0 for c in color)

            kp = pred["keypoints"][i].cpu()
            kp_coords = kp[:, :2].numpy()
            kp_vis = kp[:, 2].numpy()

            # 骨架
            for start, end in SKELETON:
                if start < len(kp_vis) and end < len(kp_vis):
                    if kp_vis[start] > 0.1 and kp_vis[end] > 0.1:
                        ax_kp.plot(
                            [kp_coords[start, 0], kp_coords[end, 0]],
                            [kp_coords[start, 1], kp_coords[end, 1]],
                            color=color_norm, linewidth=2,
                        )

            # 关键点 + 标签
            for j in range(len(kp_coords)):
                if kp_vis[j] > 0.1:
                    ax_kp.plot(
                        kp_coords[j, 0], kp_coords[j, 1], "o",
                        markersize=5, markerfacecolor="red",
                        markeredgecolor="white", markeredgewidth=0.5,
                    )
                    ax_kp.annotate(
                        KEYPOINT_NAMES[j],
                        (kp_coords[j, 0], kp_coords[j, 1]),
                        fontsize=6, color="yellow",
                        xytext=(5, 5), textcoords="offset points",
                    )

        fig_kp.tight_layout()
        kp_save_path = os.path.join(save_dir, "test_savepth_keypoints.png")
        fig_kp.savefig(kp_save_path, dpi=150, bbox_inches="tight")
        plt.close(fig_kp)
        print(f"[INFO] 关键点结果已保存到: {kp_save_path}")


def main():
    print("=" * 60)
    print("MultiTaskModel 推理测试脚本")
    print("=" * 60)

    # ── 1. 设备选择 ────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}\n")

    # ── 2. 构建模型 ────────────────────────────────────────
    cfg = Config()
    cfg.pretrained_backbone = False  # 从 checkpoint 加载，不需要预训练权重
    print("[INFO] 构建模型...")
    model = build_multitask_model(cfg)

    # ── 3. 加载 checkpoint ─────────────────────────────────
    load_checkpoint(model, pth_path, device)
    model.to(device)

    # ── 4. 检查关键点头状态 ─────────────────────────────────
    print("[DEBUG] 模型关键点头状态:")
    kp_predictor = model.model.roi_heads.keypoint_predictor
    kp_roi_pool = model.model.roi_heads.keypoint_roi_pool
    print(f"  keypoint_predictor is None: {kp_predictor is None}")
    print(f"  keypoint_roi_pool is None: {kp_roi_pool is None}")
    if kp_predictor is not None:
        for name, param in kp_predictor.named_parameters():
            print(f"  keypoint_predictor.{name}: shape={param.shape}")
    print()

    # ── 5. 加载测试图片 ────────────────────────────────────
    img_rgb, img_tensor = load_image(test_path, device)

    # ── 6. 运行推理 ────────────────────────────────────────
    print("[INFO] 正在运行推理...")
    pred = run_inference(model, img_tensor)
    print("[INFO] 推理完成！\n")

    # ── 7. 打印结果 ────────────────────────────────────────
    print_predictions(pred)

    # ── 8. 可视化保存 ──────────────────────────────────────
    visualize_predictions(img_rgb, pred, output_dir, score_threshold=0.5)

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()