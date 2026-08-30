"""
DeepScript Training Package
===========================
Supervised transfer learning, backbone fine-tuning, and checkpoint management.
"""

from .trainer import Trainer
from .train import train_deepscript, build_optimizer_and_scheduler

__all__ = [
    "Trainer",
    "train_deepscript",
    "build_optimizer_and_scheduler",
]
