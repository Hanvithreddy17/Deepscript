"""
DeepScript — Few-Shot Episodic Evaluation Engine (Prototypical Networks)
=======================================================================
Implements metric-based Few-Shot episodic benchmarking on ancient Indian script classes
using the trained Vision Transformer (ViT-B/16) and 256-D metric embeddings.

Workflow:
  1. Sample N random script classes (N-way) per episode.
  2. For each class, sample K support examples (K-shot) and Q query examples.
  3. Guarantee zero overlap between Support and Query sets (S ∩ Q = ∅).
  4. Extract 256-D L2-normalized embeddings via trained ViT + Metric Projector.
  5. Compute class prototype centroids: c_k = (1/K) * sum(z_i).
  6. Classify query embeddings based on nearest prototype distance (Euclidean or Cosine).
  7. Compute mean episode accuracy and 95% confidence intervals over E episodes.
"""

import sys
import os
import argparse
import json
import csv
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import (
    AncientScriptDataset,
    get_eval_transforms,
    create_stratified_splits,
)
from models import get_deepscript_model
from models.classifier import PrototypicalHead


class FewShotEpisodeSampler:
    """
    Episodic Sampler for N-way K-shot tasks.
    Organizes dataset indices by class and constructs balanced episodes
    with strictly disjoint support and query partitions.
    """

    def __init__(
        self,
        dataset: AncientScriptDataset,
        random_seed: int = 42,
    ):
        self.dataset = dataset
        self.rng = random.Random(random_seed)
        self.np_rng = np.random.RandomState(random_seed)

        # Build class -> list of sample indices index
        self.class_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            sample = dataset.samples[idx]
            # AncientScriptDataset sample is (image_path, class_name, label_idx) or (image_path, label_idx)
            label = sample[2] if len(sample) >= 3 else sample[1]
            self.class_to_indices[label].append(idx)

        # Filter classes that have enough samples
        self.available_classes = sorted(list(self.class_to_indices.keys()))

    def sample_episode(
        self,
        n_way: int = 5,
        k_shot: int = 5,
        q_query: int = 3,
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Samples a single N-way K-shot episode.

        Returns:
            Tuple of:
                - support_indices: Sample indices for support set (size: N * K)
                - support_labels: Original class labels for support set
                - query_indices: Sample indices for query set (size: N * Q)
                - query_labels: Original class labels for query set
        """
        # Filter classes that have at least (K + q_query) samples
        valid_classes = [
            c for c in self.available_classes
            if len(self.class_to_indices[c]) >= (k_shot + q_query)
        ]

        if len(valid_classes) < n_way:
            # Fallback: classes with at least (K + 1) samples if Q is large
            valid_classes = [
                c for c in self.available_classes
                if len(self.class_to_indices[c]) >= (k_shot + 1)
            ]
            if len(valid_classes) < n_way:
                raise ValueError(
                    f"Not enough classes with >={k_shot+1} samples to form {n_way}-way task. "
                    f"Available qualifying classes: {len(valid_classes)}"
                )

        selected_classes = self.rng.sample(valid_classes, n_way)

        support_indices = []
        support_labels = []
        query_indices = []
        query_labels = []

        for cls_label in selected_classes:
            cls_pool = list(self.class_to_indices[cls_label])
            self.rng.shuffle(cls_pool)

            cls_support = cls_pool[:k_shot]
            cls_query = cls_pool[k_shot : k_shot + q_query]

            support_indices.extend(cls_support)
            support_labels.extend([cls_label] * len(cls_support))
            query_indices.extend(cls_query)
            query_labels.extend([cls_label] * len(cls_query))

        # Explicit safety assertion guaranteeing strict zero leakage between support and query
        overlap = set(support_indices).intersection(set(query_indices))
        if overlap:
            raise RuntimeError(f"Episodic data leakage detected: Support and Query sets overlap by {len(overlap)} samples!")

        return support_indices, support_labels, query_indices, query_labels


def extract_all_embeddings(
    model: torch.nn.Module,
    dataset: AncientScriptDataset,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts 256-D metric embeddings for all samples in the dataset.

    Returns:
        Tuple of (embeddings [N, 256], labels [N])
    """
    exec_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(exec_device)
            embeddings = model.extract_features(images, project=True)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(targets)

    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def evaluate_few_shot_episodic(
    model: torch.nn.Module,
    dataset: AncientScriptDataset,
    n_way_list: List[int] = [5, 10, 20],
    k_shot_list: List[int] = [1, 5],
    q_query: int = 3,
    num_episodes: int = 100,
    metric: str = "euclidean",
    random_seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Runs multi-episode Few-Shot evaluation across multiple (N-way, K-shot) configurations.
    """
    exec_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proto_head = PrototypicalHead(metric=metric, temperature=1.0)
    proto_head.to(exec_device)

    print(f"\n[Few-Shot] Extracting metric embeddings across {len(dataset)} evaluation samples...", flush=True)
    embeddings, labels = extract_all_embeddings(model, dataset, device=exec_device)
    embeddings = embeddings.to(exec_device)
    labels = labels.to(exec_device)

    sampler = FewShotEpisodeSampler(dataset, random_seed=random_seed)

    results_summary = {
        "metric": metric,
        "num_episodes": num_episodes,
        "q_query": q_query,
        "configurations": {},
    }

    for n_way in n_way_list:
        for k_shot in k_shot_list:
            config_name = f"{n_way}_way_{k_shot}_shot"
            episode_accuracies = []

            for ep_idx in range(num_episodes):
                supp_idx, supp_lbls, query_idx, query_lbls = sampler.sample_episode(
                    n_way=n_way, k_shot=k_shot, q_query=q_query
                )

                supp_embeds = embeddings[supp_idx]
                supp_labels_t = torch.tensor(supp_lbls, device=exec_device)

                query_embeds = embeddings[query_idx]
                query_labels_t = torch.tensor(query_lbls, device=exec_device)

                # 1. Compute class prototypes
                prototypes, unique_classes = proto_head.compute_prototypes(
                    supp_embeds, supp_labels_t
                )

                # 2. Compute distances & accuracy
                logits = proto_head(query_embeds, prototypes)
                preds = logits.argmax(dim=-1)

                # Map original query labels to local prototype indices [0..N-1]
                target_local = torch.zeros(
                    query_labels_t.size(0), dtype=torch.long, device=exec_device
                )
                for i, cls_id in enumerate(unique_classes):
                    target_local[query_labels_t == cls_id] = i

                acc = (preds == target_local).float().mean().item()
                episode_accuracies.append(acc)

            acc_array = np.array(episode_accuracies)
            mean_acc = float(acc_array.mean())
            std_acc = float(acc_array.std())
            # 95% Confidence Interval: 1.96 * std / sqrt(num_episodes)
            ci95 = float(1.96 * std_acc / np.sqrt(num_episodes))

            results_summary["configurations"][config_name] = {
                "n_way": n_way,
                "k_shot": k_shot,
                "mean_accuracy": round(mean_acc, 4),
                "std_accuracy": round(std_acc, 4),
                "ci95": round(ci95, 4),
                "min_accuracy": round(float(acc_array.min()), 4),
                "max_accuracy": round(float(acc_array.max()), 4),
            }

            print(
                f"  • {n_way:2d}-Way {k_shot:2d}-Shot : Mean Acc = {mean_acc * 100:6.2f}% ± {ci95 * 100:4.2f}% (Std: {std_acc * 100:4.2f}%)",
                flush=True,
            )

    return results_summary


def run_few_shot_benchmark(
    checkpoint_path: Union[str, Path] = "checkpoints/best_vit_model.pth",
    dataset_root: str = "dataset/dataset",
    results_dir: Union[str, Path] = "results",
    num_episodes: int = 100,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    CLI / API entrypoint to run few-shot benchmark and export results.
    """
    ckpt_file = Path(checkpoint_path)
    res_dir = Path(results_dir) / "evaluation"
    res_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75, flush=True)
    print(" DEEPSCRIPT: FEW-SHOT PROTOTYPICAL LEARNING BENCHMARK", flush=True)
    print(f" Checkpoint : {ckpt_file.resolve()}", flush=True)
    print(f" Episodes   : {num_episodes} per configuration", flush=True)
    print(f" Seed       : {random_seed}", flush=True)
    print("=" * 75, flush=True)

    # 1. Load Preprocessing & Dataset
    raw_dataset = AncientScriptDataset(
        root_dir=dataset_root,
        transform=get_eval_transforms(image_size=(224, 224)),
    )
    num_classes = len(raw_dataset.get_classes())

    train_ds, val_ds, test_ds = create_stratified_splits(
        dataset=raw_dataset,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=random_seed,
    )

    # 2. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_deepscript_model(
        backbone_name="vit_b_16",
        num_classes=num_classes,
        embedding_dim=256,
        head_type="cosine",
        pretrained=False,
    )
    checkpoint = torch.load(ckpt_file, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Evaluate Few-Shot on Held-Out Test Partition
    print(f"\n--- Benchmarking on Held-Out Test Set ({len(test_ds)} isolated samples) ---", flush=True)
    few_shot_results = evaluate_few_shot_episodic(
        model=model,
        dataset=test_ds,
        n_way_list=[5, 10, 20],
        k_shot_list=[1, 5],
        q_query=3,
        num_episodes=num_episodes,
        metric="euclidean",
        random_seed=random_seed,
        device=device,
    )

    # 4. Export JSON
    json_path = res_dir / "few_shot_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(few_shot_results, f, indent=2)
    print(f"\n  >>> Few-shot results JSON saved to: {json_path}", flush=True)

    # 5. Export CSV
    csv_path = res_dir / "few_shot_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["configuration", "n_way", "k_shot", "mean_accuracy", "ci95_margin", "std", "min_acc", "max_acc"])
        for cfg_name, data in few_shot_results["configurations"].items():
            writer.writerow([
                cfg_name,
                data["n_way"],
                data["k_shot"],
                data["mean_accuracy"],
                data["ci95"],
                data["std_accuracy"],
                data["min_accuracy"],
                data["max_accuracy"],
            ])
    print(f"  >>> Few-shot results CSV saved to: {csv_path}", flush=True)

    return few_shot_results


def main():
    parser = argparse.ArgumentParser(description="DeepScript: Few-Shot Prototypical Evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_vit_model.pth")
    parser.add_argument("--dataset_root", type=str, default="dataset/dataset")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_few_shot_benchmark(
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
        results_dir=args.results_dir,
        num_episodes=args.episodes,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
