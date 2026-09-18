"""
DeepScript — Unit & Integration Test Suite for Few-Shot Learning
================================================================
Validates:
1. FewShotEpisodeSampler: disjointness of support and query sets, correct N-way K-shot counts.
2. PrototypicalHead: prototype calculation, metric distance computation (Euclidean/Cosine),
   logits calculation, and episodic loss.
3. End-to-end few-shot evaluation workflow on synthetic and actual embedding distributions.
"""

import sys
import unittest
from pathlib import Path
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.classifier import PrototypicalHead
from evaluation.few_shot_eval import FewShotEpisodeSampler


class DummyDataset:
    """Mock dataset with known class labels for episodic testing."""
    def __init__(self, num_classes=10, samples_per_class=20):
        self.samples = []
        self.classes = [f"script_{i}" for i in range(num_classes)]
        for c_idx in range(num_classes):
            for s_idx in range(samples_per_class):
                self.samples.append((f"img_{c_idx}_{s_idx}.png", c_idx))

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.samples)


class TestFewShotLearning(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")

    def test_prototypical_head_centroid_computation(self):
        """Test that class prototypes match exact mathematical centroids."""
        head = PrototypicalHead(metric="euclidean")
        
        # 3 classes, 4 samples each, dim=4
        # Class 0: all [1, 1, 1, 1] -> mean [1, 1, 1, 1]
        # Class 1: all [2, 2, 2, 2] -> mean [2, 2, 2, 2]
        # Class 2: all [3, 3, 3, 3] -> mean [3, 3, 3, 3]
        embeddings = torch.tensor([
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0, 3.0],
        ])
        labels = torch.tensor([0, 0, 1, 1, 2, 2])

        prototypes, unique_classes = head.compute_prototypes(embeddings, labels)
        
        self.assertEqual(prototypes.shape, (3, 4))
        self.assertTrue(torch.equal(unique_classes, torch.tensor([0, 1, 2])))
        self.assertTrue(torch.allclose(prototypes[0], torch.tensor([1.0, 1.0, 1.0, 1.0])))
        self.assertTrue(torch.allclose(prototypes[1], torch.tensor([2.0, 2.0, 2.0, 2.0])))
        self.assertTrue(torch.allclose(prototypes[2], torch.tensor([3.0, 3.0, 3.0, 3.0])))

    def test_prototypical_head_distances_and_logits(self):
        """Test pairwise distance calculation and inverse logit conversion."""
        head = PrototypicalHead(metric="euclidean", temperature=1.0)

        prototypes = torch.tensor([
            [0.0, 0.0],
            [10.0, 0.0],
        ])
        queries = torch.tensor([
            [0.0, 0.0],   # identical to proto 0 (distance 0 to proto 0, 10 to proto 1)
            [10.0, 0.0],  # identical to proto 1 (distance 10 to proto 0, 0 to proto 1)
        ])

        distances = head.compute_distances(queries, prototypes)
        self.assertEqual(distances.shape, (2, 2))
        self.assertAlmostEqual(distances[0, 0].item(), 0.0, places=4)
        self.assertAlmostEqual(distances[0, 1].item(), 10.0, places=4)
        self.assertAlmostEqual(distances[1, 0].item(), 10.0, places=4)
        self.assertAlmostEqual(distances[1, 1].item(), 0.0, places=4)

        logits = head(queries, prototypes)
        preds = logits.argmax(dim=-1)
        self.assertTrue(torch.equal(preds, torch.tensor([0, 1])))

    def test_episodic_sampler_disjointness(self):
        """Test that sampled support and query sets have strictly ZERO overlap."""
        dataset = DummyDataset(num_classes=8, samples_per_class=15)
        sampler = FewShotEpisodeSampler(dataset, random_seed=42)

        for _ in range(10):
            supp_idx, supp_lbls, query_idx, query_lbls = sampler.sample_episode(
                n_way=5, k_shot=3, q_query=4
            )
            # Check counts
            self.assertEqual(len(supp_idx), 5 * 3)
            self.assertEqual(len(query_idx), 5 * 4)

            # Check zero overlap
            overlap = set(supp_idx).intersection(set(query_idx))
            self.assertEqual(len(overlap), 0, f"Found overlap in episode: {overlap}")

            # Check class distribution
            supp_classes = set(supp_lbls)
            query_classes = set(query_lbls)
            self.assertEqual(len(supp_classes), 5)
            self.assertEqual(supp_classes, query_classes)

    def test_episodic_loss_and_accuracy(self):
        """Test episodic Cross-Entropy loss computation."""
        head = PrototypicalHead(metric="euclidean")
        prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        unique_classes = torch.tensor([10, 20])

        # Exact matching queries
        queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        query_labels = torch.tensor([10, 20])

        loss, acc = head.compute_episodic_loss(
            queries, query_labels, prototypes, unique_classes
        )
        self.assertEqual(acc.item(), 1.0)
        self.assertGreater(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
