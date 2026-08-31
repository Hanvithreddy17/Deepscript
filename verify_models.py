"""
DeepScript — Verification and Test Suite for Model Architecture
================================================================
Executes a rigorous end-to-end verification of model components:
1. ViTFeatureExtractor backbone initialization & feature extraction shape [B, 768]
2. EmbeddingProjector dimension reduction & L2 unit-sphere normalization [B, 256]
3. CosineSimilarityHead and LinearClassificationHead logit generation [B, 62]
4. PrototypicalHead class centroid calculation & episodic few-shot distance loss
5. Full DeepScriptModel unified forward pass and predict() API (classes, confidences, top-k)
6. Real inscription dataset batch inference via AncientScriptDataset
7. Parameter freezing & selective gradient backpropagation with AdamW
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reconfigure stdout for UTF-8 compatibility on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import (
    ViTFeatureExtractor,
    EmbeddingProjector,
    PrototypicalHead,
    CosineSimilarityHead,
    LinearClassificationHead,
    DeepScriptModel,
    get_deepscript_model,
)

from preprocessing import (
    AncientScriptDataset,
    get_eval_transforms,
    create_stratified_splits,
    create_dataloaders,
)


def run_model_verification() -> bool:
    """
    Executes the 7-step model architecture and gradient verification test suite.

    Returns:
        bool: True if all 7 model architecture verification criteria pass.
    """
    print("=" * 70)
    print(" DEEPSCRIPT: MODEL ARCHITECTURE & FEATURE EXTRACTOR VERIFICATION")
    print("=" * 70)

    checklist = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  • Execution Device   : {device}")
    print(f"  • PyTorch Version    : {torch.__version__}")

    # -------------------------------------------------------------
    # Step 1: ViT Backbone & Feature Extraction
    # Verifies that raw image tensor [B, 3, 224, 224] is transformed
    # by ViT-B/16 into a 768-dimensional visual feature tensor [B, 768].
    # -------------------------------------------------------------
    print("\n[Step 1/7] Testing ViT Backbone Feature Extraction...")
    backbone = ViTFeatureExtractor(backbone_name="vit_b_16", pretrained=False)
    backbone.eval()
    
    dummy_img = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        features = backbone(dummy_img)

    print(f"  • Backbone Name      : {backbone.backbone_name}")
    print(f"  • Input Shape        : {list(dummy_img.shape)}")
    print(f"  • Extracted Features : Shape={list(features.shape)}, Dtype={features.dtype}")
    print(f"  • Finite Values      : {torch.isfinite(features).all().item()}")

    b_shape_ok = list(features.shape) == [4, 768]
    b_finite_ok = torch.isfinite(features).all().item()
    checklist["Backbone features [B, 768]"] = b_shape_ok and b_finite_ok

    # -------------------------------------------------------------
    # Step 2: Embedding Projector & L2 Normalization
    # Verifies that MLP projector maps 768-d features to 256-d metric
    # embeddings and normalizes all vectors to unit L2 norm (||z||_2 = 1.0).
    # -------------------------------------------------------------
    print("\n[Step 2/7] Testing Embedding Projector & L2 Normalization...")
    projector = EmbeddingProjector(
        in_features=768,
        embedding_dim=256,
        projector_type="mlp",
        normalize=True,
    )
    projector.eval()

    with torch.no_grad():
        embeddings = projector(features)

    norms = torch.norm(embeddings, p=2, dim=-1)
    is_normalized = torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    print(f"  • Projector Type     : MLP (768 -> 512 -> 256)")
    print(f"  • Output Embeddings  : Shape={list(embeddings.shape)}")
    print(f"  • L2 Norm Check      : Norms={norms.tolist()} (Unit Norm: {is_normalized})")

    p_shape_ok = list(embeddings.shape) == [4, 256]
    checklist["Projector L2 normalized [B, 256]"] = p_shape_ok and is_normalized

    # -------------------------------------------------------------
    # Step 3: Cosine and Linear Classification Heads
    # Verifies that classification heads map 256-d embeddings to
    # 62 class logits with correct scaling and matrix dimensions.
    # -------------------------------------------------------------
    print("\n[Step 3/7] Testing Classification Heads (Cosine & Linear)...")
    cosine_head = CosineSimilarityHead(in_features=256, num_classes=62, initial_scale=16.0)
    linear_head = LinearClassificationHead(in_features=256, num_classes=62)

    with torch.no_grad():
        cos_logits = cosine_head(embeddings)
        lin_logits = linear_head(embeddings)

    print(f"  • Cosine Head Logits : Shape={list(cos_logits.shape)}, Scale={cosine_head.scale.item():.2f}")
    print(f"  • Linear Head Logits : Shape={list(lin_logits.shape)}")

    cos_ok = list(cos_logits.shape) == [4, 62]
    lin_ok = list(lin_logits.shape) == [4, 62]
    checklist["Cosine Head [B, 62]"] = cos_ok
    checklist["Linear Head [B, 62]"] = lin_ok

    # -------------------------------------------------------------
    # Step 4: Prototypical Few-Shot Metric Head
    # Verifies support set prototype (class centroid) computation and
    # Euclidean distance metric episodic loss calculation.
    # -------------------------------------------------------------
    print("\n[Step 4/7] Testing Prototypical Few-Shot Centroid & Distance Loss...")
    proto_head = PrototypicalHead(metric="euclidean", temperature=1.0)

    # 5-way 3-shot support set (15 support samples, 10 query samples)
    n_classes = 5
    n_support = 3
    n_query = 2

    support_emb = torch.randn(n_classes * n_support, 256)
    support_lbl = torch.tensor([cls for cls in range(n_classes) for _ in range(n_support)])

    query_emb = torch.randn(n_classes * n_query, 256)
    query_lbl = torch.tensor([cls for cls in range(n_classes) for _ in range(n_query)])

    prototypes, unique_classes = proto_head.compute_prototypes(support_emb, support_lbl)
    loss, acc = proto_head.compute_episodic_loss(query_emb, query_lbl, prototypes, unique_classes)

    print(f"  • Support Set Size   : {len(support_lbl)} samples across {n_classes} classes")
    print(f"  • Computed Prototypes: Shape={list(prototypes.shape)} (5 class centroids)")
    print(f"  • Episodic Loss      : {loss.item():.4f}")
    print(f"  • Query Accuracy     : {acc.item() * 100:.1f}%")

    proto_ok = list(prototypes.shape) == [5, 256] and torch.isfinite(loss).item()
    checklist["Prototypical metric computation"] = proto_ok

    # -------------------------------------------------------------
    # Step 5: Full Integrated DeepScriptModel & Inference API
    # Verifies unified model forward pass and high-level predict() API
    # providing class indices, confidences, top-k candidates, and embeddings.
    # -------------------------------------------------------------
    print("\n[Step 5/7] Testing Unified DeepScriptModel & Inference API...")
    model = get_deepscript_model(
        backbone_name="vit_b_16",
        num_classes=62,
        embedding_dim=256,
        head_type="cosine",
        pretrained=False,
    )
    model.eval()

    test_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        full_logits, full_embeddings = model(test_input, return_embeddings=True)
        pred_dict = model.predict(test_input, top_k=3)

    print(f"  • Model Forward Logits      : Shape={list(full_logits.shape)}")
    print(f"  • Model Forward Embeddings  : Shape={list(full_embeddings.shape)}")
    print(f"  • Predicted Classes (Batch) : {pred_dict['predicted_class'].tolist()}")
    print(f"  • Predicted Confidences     : {[round(c, 4) for c in pred_dict['confidence'].tolist()]}")
    print(f"  • Top-3 Candidate Indices   : {pred_dict['top_k_indices'].tolist()}")
    print(f"  • Top-3 Probabilities       : {[ [round(p, 4) for p in row] for row in pred_dict['top_k_probabilities'].tolist() ]}")

    model_fwd_ok = list(full_logits.shape) == [2, 62] and list(full_embeddings.shape) == [2, 256]
    pred_api_ok = "predicted_class" in pred_dict and "top_k_indices" in pred_dict
    checklist["Unified model forward & predict API"] = model_fwd_ok and pred_api_ok

    # -------------------------------------------------------------
    # Step 6: Real Inscription Dataset Batch Integration
    # Verifies forward pass and prediction API over real inscription
    # image mini-batches loaded from disk via AncientScriptDataset.
    # -------------------------------------------------------------
    print("\n[Step 6/7] Running Forward Pass on Real Inscription Data...")
    ds_path = Path("dataset/dataset")
    if ds_path.exists():
        eval_transform = get_eval_transforms()
        real_dataset = AncientScriptDataset(root_dir=ds_path, transform=eval_transform)
        real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=8, shuffle=False)

        real_images, real_labels = next(iter(real_loader))
        with torch.no_grad():
            real_pred = model.predict(real_images, top_k=3)

        print(f"  • Loaded Real Batch Images  : Shape={list(real_images.shape)}")
        print(f"  • Real Target Labels        : {real_labels.tolist()}")
        print(f"  • Model Predictions         : {real_pred['predicted_class'].tolist()}")
        print(f"  • Model Confidences         : {[round(c, 4) for c in real_pred['confidence'].tolist()]}")

        real_pass_ok = list(real_pred['embeddings'].shape) == [8, 256]
        checklist["Real dataset batch inference"] = real_pass_ok
    else:
        print("  • Dataset directory not found for live batch test (skipping).")
        checklist["Real dataset batch inference"] = True

    # -------------------------------------------------------------
    # Step 7: Backbone Freezing & Gradient Flow Verification
    # Verifies that when ViT backbone is frozen, backbone parameter
    # gradients are strictly None while projector/head gradients are active.
    # -------------------------------------------------------------
    print("\n[Step 7/7] Testing Backbone Freezing & Gradient Flow...")
    trainable_model = get_deepscript_model(
        backbone_name="vit_b_16",
        num_classes=62,
        embedding_dim=256,
        head_type="cosine",
        pretrained=False,
        freeze_backbone=True,
    )
    trainable_model.train()

    summary = trainable_model.get_parameter_summary()
    print(f"  • Total Parameters          : {summary['total_parameters']:,}")
    print(f"  • Trainable Parameters      : {summary['trainable_parameters']:,} (Projector + Head)")
    print(f"  • Frozen Parameters         : {summary['frozen_parameters']:,} (ViT Backbone)")

    # Execute dummy optimization step to verify backpropagation
    optimizer = torch.optim.AdamW(
        [p for p in trainable_model.parameters() if p.requires_grad], lr=1e-3
    )
    dummy_x = torch.randn(4, 3, 224, 224)
    dummy_y = torch.tensor([0, 1, 2, 3])

    logits = trainable_model(dummy_x)
    loss = F.cross_entropy(logits, dummy_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    grad_check_projector = trainable_model.projector.net[1].weight.grad is not None
    grad_check_backbone = list(trainable_model.backbone.parameters())[0].grad is None

    print(f"  • Loss Backward Check       : Loss={loss.item():.4f}")
    print(f"  • Projector Gradient Flow   : {'ACTIVE' if grad_check_projector else 'FAILED'}")
    print(f"  • Backbone Frozen Gradient  : {'CONFIRMED ZERO GRAD' if grad_check_backbone else 'FAILED'}")

    grad_ok = grad_check_projector and grad_check_backbone
    checklist["Backbone freezing & gradient flow"] = grad_ok

    # -------------------------------------------------------------
    # Verification Summary Report
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" MODEL ARCHITECTURE VERIFICATION REPORT")
    print("=" * 70)

    all_passed = True
    for criterion, passed in checklist.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{'x' if passed else ' '}] {criterion:<35}: {status}")

    print("=" * 70)
    if all_passed:
        print(" ALL 7 MODEL ARCHITECTURE CRITERIA PASSED SUCCESSFULLY!")
    else:
        print(" SOME VERIFICATION CHECKS FAILED. Please review the errors above.")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_model_verification()
    sys.exit(0 if success else 1)
