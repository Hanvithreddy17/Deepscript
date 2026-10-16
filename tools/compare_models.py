import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.vit_inference import ViTInferenceEngine

# Test 1: Original 5-class model
engine_5 = ViTInferenceEngine(checkpoint_path="checkpoints/whole_inscription_vit/best_vit_model.pth")
engine_5.load_model()

# Test 2: New 6-class model
engine_6 = ViTInferenceEngine(checkpoint_path="checkpoints/six_class_vit/best_vit_model.pth")
engine_6.load_model()

# Test images:
# 1. Mangulam Tamil-Brahmi inscription (TBSI_036_original.jpg)
img_mangulam = Image.open("dataset/whole_inscriptions/tamil_brahmi/TBSI_036_original.jpg")
# 2. Sittannavasal Tamil-Brahmi inscription (TBSI_068_original.jpg)
img_sittan = Image.open("dataset/whole_inscriptions/tamil_brahmi/TBSI_068_original.jpg")
# 3. Ashokan Brahmi Lumbini inscription
img_lumbini = Image.open("dataset/whole_inscriptions/brahmi/Devanampiyena Piyadasi in the Lumbini Edict of Ashoka.jpg")
# 4. Ashokan Brahmi Sarnath inscription
img_sarnath = Image.open("dataset/whole_inscriptions/brahmi/Brahmi pillar inscription in Sarnath.jpg")

print("=================================================================")
print(" ORIGINAL 5-CLASS MODEL INFERENCE (checkpoints/whole_inscription_vit)")
print("=================================================================")
for label, img, fn in [
    ("Mangulam Tamil-Brahmi", img_mangulam, "TBSI_036_original.jpg"),
    ("Sittannavasal Tamil-Brahmi", img_sittan, "TBSI_068_original.jpg"),
    ("Lumbini Ashokan Brahmi", img_lumbini, "Lumbini_Ashoka.jpg"),
    ("Sarnath Ashokan Brahmi", img_sarnath, "Sarnath_Ashoka.jpg"),
]:
    res = engine_5.predict(img, filename=fn)
    cand_str = ", ".join([f"{c['class']}: {c['confidence']:.2%}" for c in res.get("candidates", [])[:3]])
    print(f"[{label}] -> Status: {res.get('status')}, Raw: {res.get('script')}, Family: {res.get('script_family')}, Conf: {res.get('confidence'):.2%}")
    print(f"   Top-3: {cand_str}")

print("\n=================================================================")
print(" NEW 6-CLASS MODEL INFERENCE (checkpoints/six_class_vit)")
print("=================================================================")
for label, img, fn in [
    ("Mangulam Tamil-Brahmi", img_mangulam, "TBSI_036_original.jpg"),
    ("Sittannavasal Tamil-Brahmi", img_sittan, "TBSI_068_original.jpg"),
    ("Lumbini Ashokan Brahmi", img_lumbini, "Lumbini_Ashoka.jpg"),
    ("Sarnath Ashokan Brahmi", img_sarnath, "Sarnath_Ashoka.jpg"),
]:
    res = engine_6.predict(img, filename=fn)
    cand_str = ", ".join([f"{c['class']}: {c['confidence']:.2%}" for c in res.get("candidates", [])[:3]])
    print(f"[{label}] -> Status: {res.get('status')}, Raw: {res.get('script')}, Family: {res.get('script_family')}, Conf: {res.get('confidence'):.2%}")
    print(f"   Top-3: {cand_str}")
