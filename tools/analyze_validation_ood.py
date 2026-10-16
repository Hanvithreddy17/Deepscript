"""
Analyze validation split statistics and OOD scoring mechanisms for DeepScript 6-Class ViT-B/16
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.dataset import AncientScriptDataset
from preprocessing.splits import create_stratified_splits
from preprocessing.transforms import get_eval_transforms
from models import get_deepscript_model

SCREENSHOT_PATH = Path(r"C:\Users\hanvi\OneDrive\Pictures\Screenshots\Screenshot 2026-10-13 114513.png")

def main():
    print("=" * 80)
    print(" VALIDATION SPLIT & OPEN-SET METRICS ANALYSIS")
    print("=" * 80)
    
    # 1. Load Dataset and Stratified Splits
    full_ds = AncientScriptDataset(root_dir=PROJECT_ROOT / "dataset" / "whole_inscriptions")
    train_ds, val_ds, test_ds = create_stratified_splits(
        full_ds,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )
    
    print(f"Total samples: {len(full_ds)}")
    print(f"Train split: {len(train_ds)} samples")
    print(f"Val split:   {len(val_ds)} samples")
    print(f"Test split:  {len(test_ds)} samples")
    
    print("\nVal Split Samples by Class:")
    val_class_counts = {}
    for path, cls, idx in val_ds.samples:
        val_class_counts[cls] = val_class_counts.get(cls, 0) + 1
        print(f"  [{cls}] {path.name}")
    print(f"Val counts: {val_class_counts}")
    
    # 2. Load Model
    ckpt_path = PROJECT_ROOT / "checkpoints" / "six_class_vit" / "best_vit_model.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    
    model = get_deepscript_model(
        backbone_name="vit_b_16",
        num_classes=len(classes),
        embedding_dim=256,
        head_type="cosine",
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    scale = model.head.scale.item()
    head_weights = F.normalize(model.head.weight, p=2, dim=1) # [6, 256]
    
    # 3. Compute Class Prototype Embeddings from Training Set
    # In addition to the learned linear/cosine weights in head.weight,
    # let's compute empirical class mean embeddings (centroids) from the training set:
    transform = get_eval_transforms((224, 224))
    
    train_embeddings_by_class = {c: [] for c in classes}
    for path, cls, idx in train_ds.samples:
        t_img = transform(Image.open(path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            emb = F.normalize(model.extract_features(t_img, project=True), p=2, dim=1)
            train_embeddings_by_class[cls].append(emb)
            
    class_centroids = {}
    for c in classes:
        stack = torch.cat(train_embeddings_by_class[c], dim=0)
        class_centroids[c] = F.normalize(stack.mean(dim=0, keepdim=True), p=2, dim=1) # [1, 256]
        
    centroids_tensor = torch.cat([class_centroids[c] for c in classes], dim=0) # [6, 256]
    
    # 4. Evaluate Validation Set (IN-DISTRIBUTION)
    print("\n" + "-" * 80)
    print(" IN-DISTRIBUTION VALIDATION SET METRICS (CALIBRATION POOL)")
    print("-" * 80)
    
    val_metrics = []
    for path, cls, true_idx in val_ds.samples:
        t_img = transform(Image.open(path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            feat = F.normalize(model.extract_features(t_img, project=True), p=2, dim=1)
            # 1. Cosine similarity with head weights
            cos_to_weights = torch.mm(feat, head_weights.t())[0] # [6]
            # 2. Cosine similarity with empirical class centroids
            cos_to_centroids = torch.mm(feat, centroids_tensor.t())[0] # [6]
            
            logits = cos_to_weights * scale
            probs = F.softmax(logits, dim=-1)
            
            top_prob, pred_idx = probs.max(dim=-1)
            sorted_cos, sort_idx = cos_to_weights.sort(descending=True)
            top1_cos = sorted_cos[0].item()
            top2_cos = sorted_cos[1].item()
            cos_margin = top1_cos - top2_cos
            
            sorted_cent, _ = cos_to_centroids.sort(descending=True)
            top1_cent = sorted_cent[0].item()
            top2_cent = sorted_cent[1].item()
            cent_margin = top1_cent - top2_cent
            
            # Distance to nearest training exemplar
            all_train_embs = torch.cat([torch.cat(train_embeddings_by_class[c]) for c in classes], dim=0)
            max_sim_to_exemplar = torch.mm(feat, all_train_embs.t())[0].max().item()
            
            correct = (pred_idx.item() == true_idx)
            val_metrics.append({
                "path": path.name,
                "true_class": cls,
                "pred_class": classes[pred_idx.item()],
                "correct": correct,
                "top1_cos_weight": top1_cos,
                "top2_cos_weight": top2_cos,
                "margin_weight": cos_margin,
                "top1_cent": top1_cent,
                "cent_margin": cent_margin,
                "max_sim_exemplar": max_sim_to_exemplar,
                "prob": top_prob.item()
            })
            
            print(f"  {cls:13s} | Pred: {classes[pred_idx.item()]:13s} (Correct={correct}) | Top1Cos={top1_cos:.4f} | Margin={cos_margin:.4f} | Top1Centroid={top1_cent:.4f} | MaxExemplar={max_sim_to_exemplar:.4f}")

    print("\nSummary of In-Distribution Validation Set:")
    top1_cos_list = [m["top1_cos_weight"] for m in val_metrics]
    margin_list = [m["margin_weight"] for m in val_metrics]
    top1_cent_list = [m["top1_cent"] for m in val_metrics]
    exemplar_list = [m["max_sim_exemplar"] for m in val_metrics]
    
    print(f"  Top1 Cosine Sim to Weight: Min={min(top1_cos_list):.4f}, Mean={sum(top1_cos_list)/len(top1_cos_list):.4f}, Max={max(top1_cos_list):.4f}")
    print(f"  Cosine Margin (Top1 - Top2): Min={min(margin_list):.4f}, Mean={sum(margin_list)/len(margin_list):.4f}, Max={max(margin_list):.4f}")
    print(f"  Top1 Cosine Sim to Centroid: Min={min(top1_cent_list):.4f}, Mean={sum(top1_cent_list)/len(top1_cent_list):.4f}, Max={max(top1_cent_list):.4f}")
    print(f"  Max Sim to Any Training Exemplar: Min={min(exemplar_list):.4f}, Mean={sum(exemplar_list)/len(exemplar_list):.4f}, Max={max(exemplar_list):.4f}")

    # 5. Evaluate OUT-OF-DISTRIBUTION SAMPLES
    print("\n" + "-" * 80)
    print(" OUT-OF-DISTRIBUTION (OOD) SAMPLES COMPARISON")
    print("-" * 80)
    
    ood_samples = []
    if SCREENSHOT_PATH.exists():
        ood_samples.append(("User Uploaded Medieval Tamil Screenshot", SCREENSHOT_PATH))
        
    for label, path in ood_samples:
        t_img = transform(Image.open(path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            feat = F.normalize(model.extract_features(t_img, project=True), p=2, dim=1)
            cos_to_weights = torch.mm(feat, head_weights.t())[0]
            cos_to_centroids = torch.mm(feat, centroids_tensor.t())[0]
            
            sorted_cos, sort_idx = cos_to_weights.sort(descending=True)
            top1_cos = sorted_cos[0].item()
            top2_cos = sorted_cos[1].item()
            cos_margin = top1_cos - top2_cos
            
            sorted_cent, _ = cos_to_centroids.sort(descending=True)
            top1_cent = sorted_cent[0].item()
            top2_cent = sorted_cent[1].item()
            cent_margin = top1_cent - top2_cent
            
            all_train_embs = torch.cat([torch.cat(train_embeddings_by_class[c]) for c in classes], dim=0)
            max_sim_to_exemplar = torch.mm(feat, all_train_embs.t())[0].max().item()
            
            probs = F.softmax(cos_to_weights * scale, dim=-1)
            
            print(f"  [{label}]")
            print(f"    Predicted Class: {classes[sort_idx[0].item()]} (Prob={probs[sort_idx[0].item()].item():.4f})")
            print(f"    Top-1 Cosine Sim to Head Weight: {top1_cos:.4f}")
            print(f"    Cosine Margin (Top1 - Top2):     {cos_margin:.4f}")
            print(f"    Top-1 Cosine Sim to Centroid:    {top1_cent:.4f}")
            print(f"    Centroid Margin (Top1 - Top2):   {cent_margin:.4f}")
            print(f"    Max Sim to Any Training Exemplar:{max_sim_to_exemplar:.4f}")
            print(f"    Sim to Individual Centroids:")
            for i, c in enumerate(classes):
                print(f"      {c:15s}: Centroid={cos_to_centroids[i].item():+.4f} | Weight={cos_to_weights[i].item():+.4f}")

if __name__ == "__main__":
    main()
