"""
DeepScript Evaluation Package
=============================
Metrics computation, test set evaluation, and performance reporting.
"""

from .evaluate import evaluate_checkpoint

__all__ = [
    "evaluate_checkpoint",
]
