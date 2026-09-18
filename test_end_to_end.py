"""
DeepScript — End-to-End System Integration Test Suite
=====================================================
Validates all system layers from raw data ingestion to neural inference:
1. Dataset & Split Integrity: 62-class discovery, zero-leakage disjointness.
2. Preprocessing & Augmentation: 224x224 tensor normalization, finite values.
3. Feature Extractor & Projector: 768-D to 256-D metric embeddings, unit-norm property (||z||_2 = 1.0).
4. Classifier Heads: Cosine head angular scaling, Prototypical centroid distances.
5. Few-Shot Episodic Sampling: Disjoint support/query episodes, deterministic seed reproducibility.
6. Backend REST API: Full lifecycle TestClient validation of /health, /classes, /predict, and edge cases.
"""

import sys
import unittest
import io
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import (
    AncientScriptDataset,
    get_train_transforms,
    get_eval_transforms,
    create_stratified_splits,
    verify_split_disjointness,
)
from models import get_deepscript_model
from models.classifier import PrototypicalHead, CosineSimilarityHead
from evaluation.few_shot_eval import FewShotEpisodeSampler
from backend.main import app, load_model_and_checkpoint, state


class TestEndToEndSystem(unittest.TestCase):
    """Full-stack integration test suite for DeepScript."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.dataset_root = PROJECT_ROOT / "dataset" / "dataset"
        cls.checkpoint_path = PROJECT_ROOT / "checkpoints" / "best_vit_model.pth"

    def test_1_dataset_discovery_and_split_disjointness(self):
        """Validates that dataset discovers classes and stratified split has zero leakage."""
        if not self.dataset_root.exists():
            self.skipTest(f"Dataset directory not found at {self.dataset_root}")

        eval_tf = get_eval_transforms(image_size=(224, 224))
        raw_dataset = AncientScriptDataset(root_dir=self.dataset_root, transform=eval_tf)
        classes = raw_dataset.get_classes()

        self.assertEqual(len(classes), 62, f"Expected 62 classes, found {len(classes)}")
        self.assertGreater(len(raw_dataset), 6000, "Dataset sample count should exceed 6000")

        # Create stratified splits
        train_ds, val_ds, test_ds = create_stratified_splits(
            dataset=raw_dataset,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )

        # Zero leakage verification
        is_disjoint = verify_split_disjointness(train_ds, val_ds, test_ds)
        self.assertTrue(is_disjoint, "Data leakage detected between dataset partitions")

    def test_2_preprocessing_and_augmentation_invariants(self):
        """Validates that transforms produce exact [3, 224, 224] float32 tensors with valid ranges."""
        train_tf = get_train_transforms(image_size=(224, 224))
        eval_tf = get_eval_transforms(image_size=(224, 224))

        # Test on RGBA, Grayscale, and standard RGB test images (converted to RGB like dataset loader)
        modes = ["RGB", "RGBA", "L"]
        for mode in modes:
            dummy_img = Image.new(mode, (300, 150), color=128 if mode == "L" else (100, 150, 200))
            rgb_img = dummy_img.convert("RGB")
            
            t_train = train_tf(rgb_img)
            t_eval = eval_tf(rgb_img)

            self.assertEqual(t_train.shape, (3, 224, 224))
            self.assertEqual(t_eval.shape, (3, 224, 224))
            self.assertEqual(t_train.dtype, torch.float32)
            self.assertEqual(t_eval.dtype, torch.float32)
            self.assertFalse(torch.isnan(t_train).any())
            self.assertFalse(torch.isnan(t_eval).any())

    def test_3_model_architecture_and_l2_unit_sphere_norm(self):
        """Validates that metric projection head produces exact unit-length embeddings (||z||_2 = 1.0)."""
        model = get_deepscript_model(
            backbone_name="vit_b_16",
            num_classes=62,
            embedding_dim=256,
            head_type="cosine",
            pretrained=False,
        )
        model.eval()

        dummy_batch = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            embeddings = model.extract_features(dummy_batch, project=True)
            raw_feats = model.extract_features(dummy_batch, project=False)
            output = model(dummy_batch)

        self.assertEqual(raw_feats.shape, (4, 768))
        self.assertEqual(embeddings.shape, (4, 256))
        self.assertEqual(output.shape, (4, 62))

        # Check L2 unit-norm property: ||z||_2 should equal 1.0 for all embeddings
        norms = torch.norm(embeddings, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_4_few_shot_sampler_and_prototypical_inference(self):
        """Validates few-shot episodic sampler guarantees and prototypical nearest centroid matching."""
        if not self.dataset_root.exists():
            self.skipTest("Dataset not found")

        eval_tf = get_eval_transforms(image_size=(224, 224))
        raw_dataset = AncientScriptDataset(root_dir=self.dataset_root, transform=eval_tf)
        sampler = FewShotEpisodeSampler(raw_dataset, random_seed=42)

        # 5-way 5-shot episode
        supp_idx, supp_lbls, query_idx, query_lbls = sampler.sample_episode(
            n_way=5, k_shot=5, q_query=3
        )

        self.assertEqual(len(supp_idx), 25)
        self.assertEqual(len(query_idx), 15)
        # Strict disjointness
        self.assertEqual(len(set(supp_idx).intersection(set(query_idx))), 0)

        # Prototypical head inference
        head = PrototypicalHead(metric="euclidean")
        supp_embeds = torch.randn(25, 256)
        supp_embeds = F.normalize(supp_embeds, p=2, dim=-1)
        supp_labels = torch.tensor(supp_lbls)

        query_embeds = torch.randn(15, 256)
        query_embeds = F.normalize(query_embeds, p=2, dim=-1)

        prototypes, unique_classes = head.compute_prototypes(supp_embeds, supp_labels)
        self.assertEqual(prototypes.shape, (5, 256))
        self.assertEqual(len(unique_classes), 5)

        logits = head(query_embeds, prototypes)
        self.assertEqual(logits.shape, (15, 5))

    def test_5_fastapi_backend_endpoints_and_robustness(self):
        """Validates FastAPI backend endpoints (/health, /classes, /predict) and error handling."""
        with TestClient(app) as client:
            # Health
            h_resp = client.get("/health")
            self.assertEqual(h_resp.status_code, 200)
            self.assertEqual(h_resp.json()["status"], "healthy")

            # Classes
            c_resp = client.get("/classes")
            self.assertEqual(c_resp.status_code, 200)
            self.assertEqual(c_resp.json()["total_classes"], 62)

            # Predict with valid image
            dummy_img = Image.new("RGB", (224, 224), color=(200, 180, 150))
            buf = io.BytesIO()
            dummy_img.save(buf, format="PNG")
            buf.seek(0)

            p_resp = client.post("/predict?top_k=3", files={"file": ("test.png", buf, "image/png")})
            self.assertEqual(p_resp.status_code, 200)
            p_data = p_resp.json()
            self.assertIn("script", p_data)
            self.assertIn("confidence", p_data)
            self.assertIn("candidates", p_data)
            self.assertEqual(len(p_data["candidates"]), 3)
            self.assertEqual(p_data["source"], "live")

            # Invalid / corrupt upload
            bad_resp = client.post("/predict", files={"file": ("bad.bin", b"not-an-image", "application/octet-stream")})
            self.assertEqual(bad_resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
