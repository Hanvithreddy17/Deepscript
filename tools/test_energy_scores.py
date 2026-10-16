"""
Energy and open-set scoring exploration on validation set and user screenshot
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import get_eval_transforms
from preprocessing.dataset import AncientScriptDataset
from preprocessing.splits import create_stratified_splits
from models import get_deepscript_model

ckpt = torch.load(PROJECT_ROOT / 'checkpoints/six_class_vit/best_vit_model.pth', map_location='cpu')
m = get_deepscript_model('vit_b_16', 6, 256, 'cosine', False)
m.load_state_dict(ckpt['model_state_dict'])
m.eval()
transform = get_eval_transforms((224, 224))

full_ds = AncientScriptDataset(root_dir=PROJECT_ROOT / 'dataset/whole_inscriptions')
train_ds, val_ds, test_ds = create_stratified_splits(full_ds, 0.70, 0.15, 0.15, 42)
head_w = F.normalize(m.head.weight, p=2, dim=1)

def compute_metrics(img):
    t = transform(img.convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        f = F.normalize(m.extract_features(t), p=2, dim=1)
        cos = torch.mm(f, head_w.t())[0] # [6]
        max_cos = cos.max().item()
        sorted_cos, _ = cos.sort(descending=True)
        margin = (sorted_cos[0] - sorted_cos[1]).item()
        free_energy = -torch.logsumexp(cos, dim=-1).item()
        p = F.softmax(cos * 16.0, dim=-1)
        entropy = -(p * torch.log(p + 1e-12)).sum().item()
        return {
            'max_cos': max_cos,
            'margin': margin,
            'free_energy': free_energy,
            'entropy': entropy
        }

print('VALIDATION SET SCORES:')
for p, cls, idx in val_ds.samples:
    res = compute_metrics(Image.open(p))
    print(f"  {cls:13s} | MaxCos={res['max_cos']:.3f} | Margin={res['margin']:.3f} | FreeEnergy={res['free_energy']:+.3f} | Entropy={res['entropy']:.4f}")

p_user = Path(r"C:\Users\hanvi\OneDrive\Pictures\Screenshots\Screenshot 2026-10-13 114513.png")
if p_user.exists():
    res_user = compute_metrics(Image.open(p_user))
    print("\nUSER SCREENSHOT (MEDIEVAL TAMIL):")
    print(f"  MaxCos={res_user['max_cos']:.3f} | Margin={res_user['margin']:.3f} | FreeEnergy={res_user['free_energy']:+.3f} | Entropy={res_user['entropy']:.4f}")
