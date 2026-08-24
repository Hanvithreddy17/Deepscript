"""
DeepScript — Unified Vision Transformer & Few-Shot Model
========================================================
Combines Vision Transformer backbones, embedding projectors, and modular
classification heads (Cosine, Linear, or Few-Shot Prototypical) into an
integrated end-to-end deep learning model for ancient script identification.
"""

from typing import Dict, Any, Optional, Tuple, Union, List
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ViTFeatureExtractor, EmbeddingProjector
from .classifier import PrototypicalHead, CosineSimilarityHead, LinearClassificationHead

logger = logging.getLogger("deepscript.models.deepscript_model")


class DeepScriptModel(nn.Module):
    """
    Unified DeepScript Architecture for Ancient Script Recognition.

    Pipeline:
        Input Image [B, 3, 224, 224]
               │
               ▼
        [ ViT Backbone ]        ──► Dense Features [B, 768]
               │
               ▼
        [ Embedding Projector ] ──► Metric Embeddings [B, embedding_dim] (L2-normalized)
               │
               ▼
        [ Classification Head ] ──► Class Logits [B, num_classes] & Confidence Scores
    """

    def __init__(
        self,
        backbone_name: str = "vit_b_16",
        num_classes: int = 62,
        embedding_dim: int = 256,
        head_type: str = "cosine",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        projector_type: str = "mlp",
        dropout: float = 0.1,
    ):
        """
        Args:
            backbone_name: Name of vision transformer ('vit_b_16', 'vit_b_32', 'swin_t', etc.).
            num_classes: Number of target script classes (62 classes).
            embedding_dim: Dimensionality of metric representation space.
            head_type: Classification head ('cosine', 'linear', or 'prototypical').
            pretrained: Whether to load ImageNet-1k pretrained backbone weights.
            freeze_backbone: Whether to freeze backbone weights initially.
            projector_type: Projector structure ('mlp' or 'linear').
            dropout: Dropout probability in projector and head.
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.head_type = head_type.lower()

        # 1. Feature Extractor Backbone
        self.backbone = ViTFeatureExtractor(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

        # 2. Metric Embedding Projector
        self.projector = EmbeddingProjector(
            in_features=self.backbone.feature_dim,
            embedding_dim=embedding_dim,
            projector_type=projector_type,
            dropout=dropout,
            normalize=True,
        )

        # 3. Task Classification Head
        if self.head_type == "cosine":
            self.head = CosineSimilarityHead(
                in_features=embedding_dim,
                num_classes=num_classes,
                initial_scale=16.0,
                learnable_scale=True,
            )
        elif self.head_type == "linear":
            self.head = LinearClassificationHead(
                in_features=embedding_dim,
                num_classes=num_classes,
                dropout=dropout,
            )
        elif self.head_type == "prototypical":
            self.head = PrototypicalHead(
                metric="euclidean",
                temperature=1.0,
            )
        else:
            raise ValueError(
                f"Unsupported head_type: '{head_type}'. Supported: 'cosine', 'linear', 'prototypical'"
            )

        # Cache for reference class prototypes (used in few-shot inference)
        self.register_buffer("cached_prototypes", None)
        self.register_buffer("cached_classes", None)

    def extract_features(self, x: torch.Tensor, project: bool = True) -> torch.Tensor:
        """
        Extracts feature embeddings from input images.

        Args:
            x: Input image tensor [B, 3, 224, 224].
            project: If True, returns projected [B, embedding_dim] embeddings.
                     If False, returns raw backbone [B, 768] feature vectors.

        Returns:
            Feature tensor of shape [B, D].
        """
        raw_feats = self.backbone(x)
        if project:
            return self.projector(raw_feats)
        return raw_feats

    def set_cached_prototypes(
        self,
        prototypes: torch.Tensor,
        class_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Caches reference script prototypes for few-shot metric classification.

        Args:
            prototypes: Tensor [N_classes, embedding_dim] of class centroid vectors.
            class_indices: Tensor [N_classes] mapping prototype row to class integer index.
        """
        self.cached_prototypes = prototypes
        if class_indices is None:
            self.cached_classes = torch.arange(prototypes.size(0), device=prototypes.device)
        else:
            self.cached_classes = class_indices

    def clear_cached_prototypes(self) -> None:
        """Clears cached few-shot prototypes."""
        self.cached_prototypes = None
        self.cached_classes = None

    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the full DeepScript model.

        Args:
            x: Input image tensor [batch_size, 3, 224, 224].
            return_embeddings: If True, also returns normalized embeddings.

        Returns:
            - If return_embeddings is False: Logits tensor [batch_size, num_classes].
            - If return_embeddings is True: Tuple of (Logits, Embeddings).
        """
        # 1. Backbone feature extraction
        backbone_feats = self.backbone(x)

        # 2. Projection & Normalization
        embeddings = self.projector(backbone_feats)

        # 3. Head Logit Computation
        if self.head_type == "prototypical":
            if self.cached_prototypes is None:
                raise RuntimeError(
                    "Prototypical head requires cached prototypes for standard forward pass. "
                    "Call set_cached_prototypes() or use compute_episodic_loss()."
                )
            logits = self.head(embeddings, self.cached_prototypes)
        else:
            logits = self.head(embeddings)

        if return_embeddings:
            return logits, embeddings
        return logits

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Inference routine computing predictions, confidence scores, and top-k candidates.

        Args:
            x: Input image tensor [1, 3, 224, 224] or batch [B, 3, 224, 224].
            top_k: Number of highest-probability candidate classes to return.

        Returns:
            Dictionary containing:
                - 'predicted_class': Int tensor of top prediction index
                - 'confidence': Float tensor of highest softmax probability
                - 'probabilities': Tensor of all class probabilities
                - 'top_k_indices': Tensor of top-k class indices
                - 'top_k_probabilities': Tensor of top-k probabilities
                - 'embeddings': L2-normalized feature embeddings
        """
        self.eval()
        if x.dim() == 3:
            x = x.unsqueeze(0)

        logits, embeddings = self.forward(x, return_embeddings=True)
        probs = F.softmax(logits, dim=-1)

        confidences, predictions = torch.max(probs, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)

        return {
            "predicted_class": predictions,
            "confidence": confidences,
            "probabilities": probs,
            "top_k_indices": top_k_indices,
            "top_k_probabilities": top_k_probs,
            "embeddings": embeddings,
        }

    def get_parameter_summary(self) -> Dict[str, int]:
        """Returns parameter counts categorized by backbone, projector, and head."""
        backbone_total = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

        projector_total = sum(p.numel() for p in self.projector.parameters())
        projector_trainable = sum(p.numel() for p in self.projector.parameters() if p.requires_grad)

        head_total = sum(p.numel() for p in self.head.parameters())
        head_trainable = sum(p.numel() for p in self.head.parameters() if p.requires_grad)

        total = backbone_total + projector_total + head_total
        trainable = backbone_trainable + projector_trainable + head_trainable

        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "backbone_parameters": backbone_total,
            "projector_parameters": projector_total,
            "head_parameters": head_total,
        }


def get_deepscript_model(
    backbone_name: str = "vit_b_16",
    num_classes: int = 62,
    embedding_dim: int = 256,
    head_type: str = "cosine",
    pretrained: bool = True,
    freeze_backbone: bool = False,
    projector_type: str = "mlp",
    dropout: float = 0.1,
) -> DeepScriptModel:
    """
    Factory function for instantiating a pre-configured DeepScript model.

    Args:
        backbone_name: Architecture name ('vit_b_16', 'vit_b_32', 'swin_t', etc.).
        num_classes: Target classes (default 62).
        embedding_dim: Dimensionality of metric representation space (default 256).
        head_type: Classifier type ('cosine', 'linear', or 'prototypical').
        pretrained: Whether to initialize with ImageNet pretrained weights.
        freeze_backbone: Whether to freeze backbone parameters.
        projector_type: Projector structure ('mlp' or 'linear').
        dropout: Regularization dropout rate.

    Returns:
        Instantiated DeepScriptModel instance.
    """
    return DeepScriptModel(
        backbone_name=backbone_name,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        head_type=head_type,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        projector_type=projector_type,
        dropout=dropout,
    )
