"""
DeepScript — Vision Transformer Backbone & Feature Extractor
============================================================
Provides modular feature extraction backbones using Vision Transformers (ViT)
and hierarchical Swin architectures, along with multi-layer embedding
projectors for few-shot metric learning and downstream classification.
"""

from typing import Optional, Union, Tuple, List
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

logger = logging.getLogger("deepscript.models.backbone")


class ViTFeatureExtractor(nn.Module):
    """
    Vision Transformer (ViT) Feature Extractor Backbone.

    Wraps standard vision transformer architectures, replacing their top-level
    task classification heads with identity mappings to produce dense,
    richly contextualized visual representation vectors.

    Supported Backbones:
        - 'vit_b_16' (Default: ViT-Base, 16x16 patch size, 768-d features)
        - 'vit_b_32' (ViT-Base, 32x32 patch size, 768-d features)
        - 'vit_l_16' (ViT-Large, 16x16 patch size, 1024-d features)
        - 'swin_t'   (Swin Transformer Tiny, 768-d features)
        - 'swin_s'   (Swin Transformer Small, 768-d features)
        - 'resnet50' (ResNet-50 CNN baseline, 2048-d features)
    """

    FEATURE_DIMS = {
        "vit_b_16": 768,
        "vit_b_32": 768,
        "vit_l_16": 1024,
        "swin_t": 768,
        "swin_s": 768,
        "resnet50": 2048,
    }

    def __init__(
        self,
        backbone_name: str = "vit_b_16",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            backbone_name: Name of the vision architecture to instantiate.
            pretrained: Whether to load ImageNet-1k pretrained weights.
            freeze_backbone: Whether to freeze backbone parameters (requires_grad=False).
        """
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.pretrained = pretrained

        if self.backbone_name not in self.FEATURE_DIMS:
            valid_names = list(self.FEATURE_DIMS.keys())
            raise ValueError(
                f"Unsupported backbone: '{backbone_name}'. Supported options: {valid_names}"
            )

        self._feature_dim = self.FEATURE_DIMS[self.backbone_name]
        self.backbone = self._initialize_backbone(self.backbone_name, pretrained)

        if freeze_backbone:
            self.freeze()

    @property
    def feature_dim(self) -> int:
        """Returns the output feature embedding dimension of the backbone."""
        return self._feature_dim

    def _initialize_backbone(self, name: str, pretrained: bool) -> nn.Module:
        """Instantiates the underlying torchvision vision backbone without classification head."""
        try:
            if name == "vit_b_16":
                weights = tv_models.ViT_B_16_Weights.DEFAULT if pretrained else None
                model = tv_models.vit_b_16(weights=weights)
                # Replace classification head with Identity
                model.heads.head = nn.Identity()
                return model

            elif name == "vit_b_32":
                weights = tv_models.ViT_B_32_Weights.DEFAULT if pretrained else None
                model = tv_models.vit_b_32(weights=weights)
                model.heads.head = nn.Identity()
                return model

            elif name == "vit_l_16":
                weights = tv_models.ViT_L_16_Weights.DEFAULT if pretrained else None
                model = tv_models.vit_l_16(weights=weights)
                model.heads.head = nn.Identity()
                return model

            elif name == "swin_t":
                weights = tv_models.Swin_T_Weights.DEFAULT if pretrained else None
                model = tv_models.swin_t(weights=weights)
                model.head = nn.Identity()
                return model

            elif name == "swin_s":
                weights = tv_models.Swin_S_Weights.DEFAULT if pretrained else None
                model = tv_models.swin_s(weights=weights)
                model.head = nn.Identity()
                return model

            elif name == "resnet50":
                weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
                model = tv_models.resnet50(weights=weights)
                model.fc = nn.Identity()
                return model

            else:
                raise ValueError(f"Unknown backbone: {name}")

        except Exception as e:
            if pretrained:
                logger.warning(
                    f"Failed to load pretrained weights for {name} ({e}). "
                    f"Falling back to randomly initialized weights."
                )
                return self._initialize_backbone(name, pretrained=False)
            raise

    def freeze(self) -> None:
        """Freezes all backbone parameters so gradients are not computed."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreezes all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_n_blocks(self, n: int = 2) -> None:
        """
        Unfreezes the last N transformer encoder blocks while keeping early layers frozen.

        Args:
            n: Number of final transformer blocks/layers to unfreeze.
        """
        self.freeze()
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layers"):
            layers = self.backbone.encoder.layers
            total_layers = len(layers)
            for i in range(max(0, total_layers - n), total_layers):
                for param in layers[i].parameters():
                    param.requires_grad = True
            # Also unfreeze encoder norm if present
            if hasattr(self.backbone.encoder, "ln"):
                for param in self.backbone.encoder.ln.parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts visual feature representations.

        Args:
            x: Input image tensor of shape [batch_size, 3, H, W] (e.g., [B, 3, 224, 224]).

        Returns:
            Tensor of shape [batch_size, feature_dim] containing extracted visual features.
        """
        return self.backbone(x)


class EmbeddingProjector(nn.Module):
    """
    Feature Embedding Projector.

    Projects high-dimensional backbone features (e.g., 768-d from ViT) into a
    compact metric embedding space (e.g., 256-d or 512-d) optimized for
    distance computation, few-shot prototypical clustering, and script classification.
    """

    def __init__(
        self,
        in_features: int = 768,
        embedding_dim: int = 256,
        projector_type: str = "mlp",
        dropout: float = 0.1,
        normalize: bool = True,
    ):
        """
        Args:
            in_features: Dimensionality of input feature vectors from the backbone.
            embedding_dim: Output dimension of the metric embedding space.
            projector_type: Projection architecture ('linear' or 'mlp').
            dropout: Dropout probability.
            normalize: Whether to apply L2 normalization to output embeddings.
        """
        super().__init__()
        self.in_features = in_features
        self.embedding_dim = embedding_dim
        self.projector_type = projector_type.lower()
        self.normalize = normalize

        if self.projector_type == "linear":
            self.net = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(in_features, embedding_dim),
            )
        elif self.projector_type == "mlp":
            hidden_dim = max(in_features, embedding_dim * 2)
            self.net = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, embedding_dim),
            )
        else:
            raise ValueError(
                f"Unsupported projector_type: '{projector_type}'. Supported: 'linear', 'mlp'"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects input features to the metric embedding space.

        Args:
            x: Input feature tensor [batch_size, in_features].

        Returns:
            Projected embedding tensor [batch_size, embedding_dim].
        """
        embeddings = self.net(x)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
