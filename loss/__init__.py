"""
Loss Components
=================
- multitask_loss: Multi-task loss weighting and scheduling
"""

from loss.multitask_loss import MultiTaskLoss, DynamicLossWeightScheduler, build_multitask_loss