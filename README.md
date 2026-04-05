# MultiTaskTraining

基于 PyTorch 的多任务学习框架，在 COCO2017 数据集上联合训练 **实例分割（Instance Segmentation）** 和 **关键点检测（Keypoint Detection）** 任务。

## 项目简介

本项目基于 torchvision 的 Mask R-CNN 框架进行扩展，实现了一个端到端的多任务学习模型。模型共享 ResNet-50 + FPN 骨干网络，同时进行目标检测、实例分割和人体关键点检测，支持可配置的多任务损失权重。

### 模型架构

```
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
```

### 损失函数

总损失由四部分加权组成：

```
L_total = w_rpn × (L_rpn_cls + L_rpn_reg)
        + w_box × (L_box_cls + L_box_reg)
        + w_mask × L_mask
        + w_kp  × L_keypoint
```

| 损失组件 | 描述 |
|----------|------|
| RPN Loss | 区域候选网络的目标分类 + 边界框回归损失 |
| Box Loss | 目标分类 + 边界框回归损失 |
| Mask Loss | 实例分割的逐像素二值交叉熵损失 |
| Keypoint Loss | 关键点热图的交叉熵损失（17个 COCO 人体关键点）|

## 项目结构

```
MultiTaskTraining/
├── config.py                  # 训练配置（数据类，所有超参数）
├── train.py                   # 主训练脚本
├── requirements.txt           # Python 依赖
├── model/
│   ├── __init__.py
│   ├── backbone.py            # 骨干网络定义
│   ├── multitask_model.py     # 多任务模型（Mask R-CNN + Keypoint）
│   ├── instanceSeg_head.py    # 实例分割头
│   └── keyPoint_head.py       # 关键点检测头
├── datasets/
│   ├── __init__.py
│   └── coco_dataset.py        # COCO 数据集加载与预处理
├── loss/
│   ├── __init__.py
│   └── multitask_loss.py      # 多任务损失加权与动态调度
├── logger/
│   ├── __init__.py
│   └── training_logger.py     # TensorBoard 训练日志记录
├── utils/
│   ├── __init__.py
│   └── visualization.py       # 可视化工具（分割掩码、关键点绘制）
└── test_output/               # 测试/可视化输出目录
```

## 环境要求

- Python >= 3.8
- PyTorch >= 1.12.0
- torchvision >= 0.13.0
- CUDA（推荐，支持多 GPU 训练）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

下载 COCO2017 数据集并解压，目录结构应如下：

```
coco_dataset/
├── annotations/
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── train2017/
└── val2017/
```

### 3. 开始训练

使用默认配置训练：

```bash
python train.py
```

自定义参数训练：

```bash
python train.py \
    --data_root /path/to/coco_dataset \
    --epochs 24 \
    --batch_size 4 \
    --lr 0.005 \
    --device cuda:0 \
    --num_workers 8
```

从检查点恢复训练：

```bash
python train.py --resume ./output/checkpoint_epoch_10.pth
```

多 GPU 训练：

```bash
python train.py --device cuda:0,1
```

### 4. 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_root` | `/home/hc/XuRongTangProj/coco_dataset` | COCO 数据集根目录 |
| `--batch_size` | 4 | 批量大小 |
| `--epochs` | 12 | 训练轮数 |
| `--lr` | 0.005 | 学习率 |
| `--device` | `cuda:1` | 训练设备（`cpu` / `cuda` / `cuda:0,1`）|
| `--num_workers` | 10 | DataLoader 工作线程数 |
| `--resume` | None | 检查点路径（用于恢复训练）|
| `--log_interval` | 10 | 每 N 个 batch 记录一次日志 |
| `--seed` | 42 | 随机种子 |

## 配置说明

所有配置项在 `config.py` 的 `Config` 数据类中定义，主要包含以下部分：

- **数据配置**：数据集路径、批量大小、数据加载线程数
- **模型配置**：骨干网络（ResNet-50）、预训练权重、可训练层数、类别数（91）、关键点数（17）
- **训练配置**：学习率、动量、权重衰减、学习率调度器（Step / Cosine）、训练轮数、预热轮数
- **多任务损失权重**：RPN、Box、Mask、Keypoint 四个任务的损失权重
- **日志与可视化**：TensorBoard 日志目录、可视化间隔和参数

详细配置请参考 `config.py`。

## 多任务损失动态调度

项目支持动态损失权重调度策略（`DynamicLossWeightScheduler`）：

| 策略 | 说明 |
|------|------|
| `constant` | 固定权重，全程不变 |
| `linear_warmup` | 从 0 线性增长到初始权重值 |
| `alternating` | 在不同 epoch 交替聚焦于分割和关键点任务 |

## 日志与可视化

- **TensorBoard**：训练和验证的所有损失指标自动记录到 TensorBoard
  ```bash
  tensorboard --logdir ./runs
  ```
- **可视化输出**：定期生成验证集的实例分割和关键点检测结果网格图，保存在 `test_output/` 目录

## 检查点管理

训练过程中自动保存：

- `checkpoint_epoch_N.pth`：每个 epoch 的检查点
- `best_model.pth`：损失最低的最优模型
- `final_model.pth`：训练结束时的最终模型

每个检查点包含：模型权重、优化器状态、学习率调度器状态、epoch 号、损失字典和配置。

## License

This project is for research and educational purposes.

## 修复日志

### 1. 修复无法得到特征点的模型 Bug（2026-04-05）

**问题**：模型推理结果中缺少 `keypoints` 和 `keypoints_scores`，`roi_heads.keypoint_predictor` 始终为 `None`。

**根因**：`model/multitask_model.py` 中使用 `MaskRCNN` 构造模型，但 torchvision v0.23.0 的 `MaskRCNN.__init__` 不接受也不传递 `keypoint_*` 参数到 `RoIHeads`（仅处理 `mask_*` 参数），导致关键点检测头未被创建。

**修复方案**：在 `MaskRCNN` 构造完成后，手动创建并分配关键点检测组件：

```python
from torchvision.models.detection.keypoint_rcnn import (
    KeypointRCNNHeads, KeypointRCNNPredictor
)

# 使用 torchvision 标准 KeypointRCNN 架构
keypoint_head = KeypointRCNNHeads(out_channels, (512,) * 8)
keypoint_predictor = KeypointRCNNPredictor(in_channels=512, num_keypoints=17)

self.model.roi_heads.keypoint_roi_pool = keypoint_roi_pooler
self.model.roi_heads.keypoint_head = keypoint_head
self.model.roi_heads.keypoint_predictor = keypoint_predictor
```

**验证**：修复后推理输出包含 `keypoints`（shape: `[N, 17, 3]`）和 `keypoints_scores`，关键点检测正常工作。
