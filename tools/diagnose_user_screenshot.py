r"""
Deep diagnosis of the user's uploaded screenshot:
C:\Users\hanvi\OneDrive\Pictures\Screenshots\Screenshot 2026-10-13 114513.png
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import get_deepscript_model
from preprocessing import get_eval_transforms
from backend.vit_inference import EPIGRAPHIC_DOSSIER_DB

SCREENSHOT_PATH = Path(r"C:\Users\hanvi\OneDrive\Pictures\Screenshots\Screenshot 2026-10-13 114513.png")

def diagnose():
    print("=" * 80)
    print(" DIAGNOSIS OF USER UPLOADED SCREENSHOT")
    print(f" File: {SCREENSHOT_PATH}")
    print("=" * 80)
    
    if not SCREENSHOT_PATH.exists():
        print(f"ERROR: {SCREENSHOT_PATH} does not exist!")
        return

    img = Image.open(SCREENSHOT_PATH)
    print(f"Image Format: {img.format}, Mode: {img.mode}, Size: {img.size} (Width x Height)")
    
    transform = get_eval_transforms(image_size=(224, 224))
    img_rgb = img.convert("RGB")
    tensor = transform(img_rgb).unsqueeze(0)
    
    # 1. Evaluate on 6-Class Model
    ckpt_6_path = PROJECT_ROOT / "checkpoints" / "six_class_vit" / "best_vit_model.pth"
    print(f"\n[1] Evaluating on 6-Class Model ({ckpt_6_path}):")
    ckpt_6 = torch.load(ckpt_6_path, map_location="cpu", weights_only=False)
    classes_6 = ckpt_6["classes"]
    
    model_6 = get_deepscript_model(
        backbone_name="vit_b_16",
        num_classes=len(classes_6),
        embedding_dim=256,
        head_type="cosine",
        pretrained=False,
    )
    model_6.load_state_dict(ckpt_6["model_state_dict"])
    model_6.eval()
    
    with torch.no_grad():
        features_6 = model_6.extract_features(tensor, project=True)
        scale_6 = model_6.head.scale.item()
        head_weight_6 = model_6.head.weight # [6, 256]
        # Cosine similarity is dot product of normalized feature and normalized weights
        norm_feat_6 = F.normalize(features_6, p=2, dim=1)
        norm_weights_6 = F.normalize(head_weight_6, p=2, dim=1)
        cos_sims_6 = torch.mm(norm_feat_6, norm_weights_6.t())[0] # [6]
        logits_6 = model_6(tensor)[0]
        probs_6 = F.softmax(logits_6, dim=-1)
        
    print(f"  Head scale parameter: {scale_6:.4f}")
    print("  Per-Class Scores:")
    for i, cls in enumerate(classes_6):
        dossier = EPIGRAPHIC_DOSSIER_DB.get(cls, {})
        display = dossier.get("name", cls)
        print(f"    [{i}] {cls:15s} ({display:35s}) -> CosineSim: {cos_sims_6[i].item():+.4f} | Logit: {logits_6[i].item():+8.3f} | Softmax Prob: {probs_6[i].item()*100:6.2f}%")
        
    top3_p, top3_i = torch.topk(probs_6, 3)
    print(f"\n  Top Predicted Class: {classes_6[top3_i[0]]} ({probs_6[top3_i[0]].item()*100:.2f}%)")
    print(f"  2nd Candidate:       {classes_6[top3_i[1]]} ({probs_6[top3_i[1]].item()*100:.2f}%)")
    print(f"  3rd Candidate:       {classes_6[top3_i[2]]} ({probs_6[top3_i[2]].item()*100:.2f}%)")

    # 2. Check training dataset comparison
    print(f"\n[2] Feature Distance from Class Prototype Centers in 6-Class Model:")
    # Compute prototype center for brahmi and tamil_brahmi using dataset images
    dataset_tb_dir = PROJECT_ROOT / "dataset" / "whole_inscriptions" / "tamil_brahmi"
    dataset_br_dir = PROJECT_ROOT / "dataset" / "whole_inscriptions" / "brahmi"
    
    tb_embeddings = []
    for p in sorted(dataset_tb_dir.glob("*.jpg")):
        t_img = transform(Image.open(p).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            feat = F.normalize(model_6.extract_features(t_img, project=True), p=2, dim=1)
            tb_embeddings.append(feat)
            
    br_embeddings = []
    for p in sorted(dataset_br_dir.glob("*.jpg")):
        t_img = transform(Image.open(p).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            feat = F.normalize(model_6.extract_features(t_img, project=True), p=2, dim=1)
            br_embeddings.append(feat)
            
    tb_stack = torch.cat(tb_embeddings, dim=0) # [12, 256]
    br_stack = torch.cat(br_embeddings, dim=0) # [10, 256]
    
    tb_mean = F.normalize(tb_stack.mean(dim=0, keepdim=True), p=2, dim=1)
    br_mean = F.normalize(br_stack.mean(dim=0, keepdim=True), p=2, dim=1)
    
    sim_to_tb_mean = torch.mm(norm_feat_6, tb_mean.t())[0, 0].item()
    sim_to_br_mean = torch.mm(norm_feat_6, br_mean.t())[0, 0].item()
    print(f"  Cosine Sim of User Screenshot to Tamil-Brahmi Training Mean: {sim_to_tb_mean:+.4f}")
    print(f"  Cosine Sim of User Screenshot to Ashokan Brahmi Training Mean: {sim_to_br_mean:+.4f}")

    # Also compute max similarity to individual dataset exemplars
    sims_to_tb_all = torch.mm(norm_feat_6, tb_stack.t())[0]
    sims_to_br_all = torch.mm(norm_feat_6, br_stack.t())[0]
    print(f"  Max Cosine Sim to Any Tamil-Brahmi Training Image: {sims_to_tb_all.max().item():+.4f}")
    print(f"  Max Cosine Sim to Any Ashokan Brahmi Training Image: {sims_to_br_all.max().item():+.4f}")

if __name__ == "__main__":
    diagnose()
