"""
DeepScript Models Package
=========================
Modular deep learning architectures for ancient Indian script recognition.
"""

from .backbone import ViTFeatureExtractor, EmbeddingProjector
from .classifier import (
    PrototypicalHead,
    CosineSimilarityHead,
    LinearClassificationHead,
)
from .deepscript_model import DeepScriptModel, get_deepscript_model

__all__ = [
    "ViTFeatureExtractor",
    "EmbeddingProjector",
    "PrototypicalHead",
    "CosineSimilarityHead",
    "LinearClassificationHead",
    "DeepScriptModel",
    "get_deepscript_model",
]
