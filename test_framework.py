"""Quick integration test for the multi-task framework."""
import torch
from config import Config
from model.multitask_model import build_multitask_model
from loss.multitask_loss import build_multitask_loss
from logger.training_logger import TrainingLogger
import tempfile

# Test 1: Config
cfg = Config()
cfg.pretrained_backbone = False
print('[TEST 1] Config OK')

# Test 2: Build model
model = build_multitask_model(cfg)
print('[TEST 2] Model built OK')

# Test 3: Loss
loss_fn = build_multitask_loss(cfg)
print('[TEST 3] Loss OK')

# Test 4: Training forward pass
model.train()
images = [torch.rand(3, 200, 300)]
# Create keypoints with visibility=2 (labeled) to trigger keypoint loss
kp = torch.zeros((1, 17, 3), dtype=torch.float32)
kp[:, :, 0] = torch.linspace(15, 75, 17)  # x coords
kp[:, :, 1] = torch.linspace(15, 75, 17)  # y coords
kp[:, :, 2] = 2  # visibility=2 (labeled and visible)
targets = [{
    'boxes': torch.tensor([[10.0, 10.0, 80.0, 80.0]], dtype=torch.float32),
    'labels': torch.tensor([1], dtype=torch.int64),
    'masks': torch.zeros((1, 200, 300), dtype=torch.uint8),
    'keypoints': kp,
}]

loss_dict = model(images, targets)
total_loss = loss_fn.compute_total_loss(loss_dict)
print(f'[TEST 4] Training forward OK, total_loss={total_loss.item():.4f}')
for k, v in loss_dict.items():
    print(f'       {k}: {v.item():.4f}')

# Test 5: Eval forward pass
model.eval()
with torch.no_grad():
    preds = model(images)
print(f'[TEST 5] Eval forward OK, keys={list(preds[0].keys())}')

# Test 6: Logger
logger = TrainingLogger(experiment_name='test')
weighted = loss_fn.get_weighted_losses(loss_dict)
logger.log_losses(weighted, 0)
logger.close()
print('[TEST 6] Logger OK')

print()
print('=' * 50)
print('ALL TESTS PASSED!')
print('=' * 50)