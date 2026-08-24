"""
DeepScript — Classification & Metric Learning Heads
===================================================
Provides modular classification layers for ancient Indian script recognition:
1. PrototypicalHead: Metric-learning centroid classifier for Few-Shot / Episodic tasks.
2. CosineSimilarityHead: Angular metric classifier with learnable temperature scaling.
3. LinearClassificationHead: Standard linear probe with regularization.
"""

from typing import Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalHead(nn.Module):
    """
    Prototypical Network Metric-Learning Head (Snell et al.).

    Computes class prototypes (centroids) from a support set of script embeddings
    and classifies query embeddings based on minimum Euclidean or Cosine distance.
    Supports episodic training, prototype caching, and few-shot inference.
    """

    def __init__(self, metric: str = "euclidean", temperature: float = 1.0):
        """
        Args:
            metric: Distance metric to evaluate ('euclidean' or 'cosine').
            temperature: Scaling temperature factor for softmax logits.
        """
        super().__init__()
        metric = metric.lower()
        if metric not in ("euclidean", "cosine"):
            raise ValueError(f"Unsupported metric: '{metric}'. Supported: 'euclidean', 'cosine'")
        self.metric = metric
        self.temperature = nn.Parameter(torch.tensor(float(temperature)), requires_grad=False)

    def compute_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the class centroid prototypes from support set embeddings.

        Args:
            support_embeddings: Tensor [N_support, embedding_dim].
            support_labels: Tensor [N_support] of integer class indices.

        Returns:
            Tuple of:
                - prototypes: Tensor [N_classes, embedding_dim]
                - unique_classes: Tensor [N_classes] of sorted class indices
        """
        unique_classes = torch.unique(support_labels).sort()[0]
        n_classes = len(unique_classes)
        embedding_dim = support_embeddings.size(-1)

        prototypes = torch.zeros(
            (n_classes, embedding_dim),
            dtype=support_embeddings.dtype,
            device=support_embeddings.device,
        )

        for i, cls_idx in enumerate(unique_classes):
            mask = support_labels == cls_idx
            cls_samples = support_embeddings[mask]
            prototypes[i] = cls_samples.mean(dim=0)

        if self.metric == "cosine":
            prototypes = F.normalize(prototypes, p=2, dim=-1)

        return prototypes, unique_classes

    def compute_distances(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes pairwise distances between query embeddings and class prototypes.

        Args:
            query_embeddings: Tensor [N_query, embedding_dim].
            prototypes: Tensor [N_classes, embedding_dim].

        Returns:
            Distance matrix tensor [N_query, N_classes].
        """
        if self.metric == "euclidean":
            # (q - p)^2 = ||q||^2 + ||p||^2 - 2 q.p
            # torch.cdist computes pairwise Euclidean distance directly and stably
            return torch.cdist(query_embeddings, prototypes, p=2.0)
        else:
            # Cosine distance: 1 - cos_sim(q, p)
            q_norm = F.normalize(query_embeddings, p=2, dim=-1)
            p_norm = F.normalize(prototypes, p=2, dim=-1)
            similarity = torch.mm(q_norm, p_norm.t())
            return 1.0 - similarity

    def forward(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes classification logits for query samples against prototypes.

        Args:
            query_embeddings: Tensor [N_query, embedding_dim].
            prototypes: Tensor [N_classes, embedding_dim].

        Returns:
            Logits tensor [N_query, N_classes] (negative scaled distance).
        """
        distances = self.compute_distances(query_embeddings, prototypes)
        # Convert distances to logits: higher logit = closer / higher probability
        logits = -distances / self.temperature
        return logits

    def compute_episodic_loss(
        self,
        query_embeddings: torch.Tensor,
        query_labels: torch.Tensor,
        prototypes: torch.Tensor,
        unique_classes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes episodic Cross-Entropy loss on query predictions.

        Args:
            query_embeddings: Tensor [N_query, embedding_dim].
            query_labels: Tensor [N_query] with original class labels.
            prototypes: Tensor [N_classes, embedding_dim].
            unique_classes: Tensor [N_classes] mapping prototype indices to original class labels.

        Returns:
            Tuple of:
                - loss: Scalar Cross-Entropy loss tensor.
                - accuracy: Classification accuracy on query set in [0.0, 1.0].
        """
        logits = self.forward(query_embeddings, prototypes)

        # Map query labels to local prototype indices [0 .. N_classes-1]
        target_indices = torch.zeros(
            query_labels.size(0), dtype=torch.long, device=query_labels.device
        )
        for i, cls_idx in enumerate(unique_classes):
            target_indices[query_labels == cls_idx] = i

        loss = F.cross_entropy(logits, target_indices)
        preds = logits.argmax(dim=-1)
        acc = (preds == target_indices).float().mean()
        return loss, acc


class CosineSimilarityHead(nn.Module):
    """
    Cosine Similarity Normalized Classification Head.

    Maintains normalized weight vectors per script class and computes scaled
    cosine similarities. This provides superior feature clustering and angle-based
    separation for fine-grained character recognition.
    """

    def __init__(
        self,
        in_features: int = 256,
        num_classes: int = 62,
        initial_scale: float = 16.0,
        learnable_scale: bool = True,
    ):
        """
        Args:
            in_features: Dimensionality of input feature vectors.
            num_classes: Number of script classes (62 target classes).
            initial_scale: Initial inverse temperature scale s = 1 / tau.
            learnable_scale: Whether the scaling factor s is learnable during training.
        """
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        # Class weight vectors [num_classes, in_features]
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(initial_scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(initial_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature embeddings [batch_size, in_features].

        Returns:
            Logits tensor [batch_size, num_classes].
        """
        # Normalize input features and class weights
        x_norm = F.normalize(x, p=2, dim=-1)
        w_norm = F.normalize(self.weight, p=2, dim=-1)

        # Compute cosine similarity
        cosine_sim = F.linear(x_norm, w_norm)  # [B, num_classes]

        # Apply scale (inverse temperature)
        logits = cosine_sim * self.scale
        return logits


class LinearClassificationHead(nn.Module):
    """
    Standard Linear Probe Classification Head.

    Includes LayerNorm, Dropout, and Linear projection layer to map
    embeddings to class probability logits.
    """

    def __init__(
        self,
        in_features: int = 256,
        num_classes: int = 62,
        dropout: float = 0.2,
    ):
        """
        Args:
            in_features: Dimensionality of input feature vectors.
            num_classes: Number of output script classes.
            dropout: Dropout rate applied before linear layer.
        """
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [batch_size, in_features].

        Returns:
            Logits tensor [batch_size, num_classes].
        """
        return self.net(x)
