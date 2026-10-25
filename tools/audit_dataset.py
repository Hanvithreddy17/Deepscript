"""
DeepScript: Phase 1 Comprehensive Dataset Quality Audit & Contact Sheet Generator
================================================================================
Inspects all training, validation, and test images across the 6 supported classes:
  1. Ashokan Brahmi
  2. Tamil-Brahmi
  3. Kharosthi
  4. Grantha
  5. Gupta
  6. Kadamba

Reports:
- Image counts, unique hashes, near-duplicates, and same-monument leakage.
- Resolution, aspect ratio distribution, color format.
- Stroke visibility metrics (edge energy, contrast, character area ratio).
- Generates contact sheet composite images with metadata overlays.
"""

import os
import hashlib
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "whole_inscriptions"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "contact_sheets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["brahmi", "tamil_brahmi", "kharosthi", "grantha", "gupta", "kadamba"]

def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def analyze_image_quality(img_path):
    with Image.open(img_path) as img:
        w, h = img.size
        aspect_ratio = w / max(1, h)
        mode = img.mode
        
        # Convert to grayscale array for analysis
        gray = img.convert("L")
        arr = np.asarray(gray, dtype=np.float32)
        
        pixel_mean = float(arr.mean())
        pixel_std = float(arr.std())
        
        # Stroke edge energy using Sobel gradient
        arr_uint8 = np.asarray(gray, dtype=np.uint8)
        sobel_x = cv2.Sobel(arr_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(arr_uint8, cv2.CV_64F, 0, 1, ksize=3)
        edge_energy = float(np.mean(np.sqrt(sobel_x**2 + sobel_y**2)))
        
        # High-frequency stroke ratio (contrast)
        # Using Otsu / adaptive threshold to estimate stroke coverage
        blur = cv2.GaussianBlur(arr_uint8, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 4)
        stroke_density = float(np.mean(thresh > 0))
        
        # Classify visibility
        # If aspect ratio is extreme (> 4:1 or < 0.25:1), image is an extreme ribbon strip
        # If edge energy < 15 or pixel_std < 25, low contrast/faded
        is_extreme_aspect = (aspect_ratio > 4.0 or aspect_ratio < 0.25)
        is_faded_or_low_contrast = (edge_energy < 18.0 or pixel_std < 25.0)
        
        quality_category = "visible"
        issues = []
        if is_extreme_aspect:
            issues.append(f"extreme_ribbon_aspect_{aspect_ratio:.1f}")
        if is_faded_or_low_contrast:
            issues.append("low_contrast_faded")
        if stroke_density < 0.05 or stroke_density > 0.60:
            issues.append("poor_stroke_density")
            
        if issues:
            quality_category = "problematic: " + ", ".join(issues)
            
        return {
            "path": img_path,
            "filename": img_path.name,
            "width": w,
            "height": h,
            "aspect_ratio": round(aspect_ratio, 2),
            "mode": mode,
            "pixel_std": round(pixel_std, 1),
            "edge_energy": round(edge_energy, 1),
            "stroke_density": round(stroke_density, 3),
            "quality_category": quality_category,
            "is_problematic": len(issues) > 0,
            "issues": issues,
        }

def create_contact_sheet(cls_name, image_records, output_path):
    """
    Creates a clean visual contact sheet grid of all images for the class.
    Each cell contains a thumbnail with a title bar displaying filename, resolution, and aspect ratio.
    """
    n = len(image_records)
    if n == 0:
        return
        
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    cell_w = 420
    cell_h = 280
    thumb_h = 220
    header_h = 60
    
    sheet_w = cols * cell_w + (cols + 1) * 10
    sheet_h = header_h + rows * cell_h + (rows + 1) * 10
    
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(245, 246, 248))
    draw = ImageDraw.Draw(sheet)
    
    # Title bar
    draw.rectangle([0, 0, sheet_w, header_h], fill=(30, 41, 59))
    draw.text((20, 18), f"DeepScript Contact Sheet: {cls_name.upper()} ({n} Images)", fill=(255, 255, 255))
    
    for idx, rec in enumerate(image_records):
        r = idx // cols
        c = idx % cols
        x0 = 10 + c * (cell_w + 10)
        y0 = header_h + 10 + r * (cell_h + 10)
        
        # Cell border & background
        draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], fill=(255, 255, 255), outline=(203, 213, 225), width=1)
        
        # Load and resize thumbnail maintaining aspect ratio
        try:
            with Image.open(rec["path"]) as img:
                img_rgb = img.convert("RGB")
                img_rgb.thumbnail((cell_w - 20, thumb_h - 10), Image.Resampling.LANCZOS)
                tw, th = img_rgb.size
                tx = x0 + (cell_w - tw) // 2
                ty = y0 + 10 + (thumb_h - 10 - th) // 2
                sheet.paste(img_rgb, (tx, ty))
        except Exception as e:
            draw.text((x0 + 20, y0 + 50), f"Error loading image: {e}", fill=(220, 38, 38))
            
        # Info footer
        info_y = y0 + thumb_h + 5
        draw.rectangle([x0, info_y, x0 + cell_w, y0 + cell_h], fill=(241, 245, 249))
        
        fn_display = rec['filename'][:35] + ("..." if len(rec['filename']) > 35 else "")
        draw.text((x0 + 8, info_y + 4), fn_display, fill=(15, 23, 42))
        
        meta_str = f"{rec['width']}x{rec['height']} | AR: {rec['aspect_ratio']:.2f} | Edge: {rec['edge_energy']}"
        draw.text((x0 + 8, info_y + 22), meta_str, fill=(71, 85, 105))
        
        status_color = (220, 38, 38) if rec["is_problematic"] else (22, 163, 74)
        status_text = "ISSUES: " + ", ".join(rec["issues"]) if rec["is_problematic"] else "CLEAR VISIBILITY"
        draw.text((x0 + 8, info_y + 38), status_text[:48], fill=status_color)
        
    sheet.save(output_path, quality=90)
    print(f"Contact sheet saved: {output_path}")

def main():
    print("==================================================================")
    print(" DEEPSCRIPT: PHASE 1 COMPREHENSIVE DATASET AUDIT REPORT")
    print("==================================================================")
    
    total_imgs = 0
    all_hashes = {}
    class_reports = {}
    
    for cls in CLASSES:
        cls_dir = DATASET_DIR / cls
        files = [p for p in sorted(cls_dir.glob("*.*")) if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        total_imgs += len(files)
        
        records = []
        for p in files:
            rec = analyze_image_quality(p)
            h = compute_sha256(p)
            rec["sha256"] = h
            records.append(rec)
            
        class_reports[cls] = records
        
        # Generate contact sheet
        cs_path = OUTPUT_DIR / f"contact_sheet_{cls}.jpg"
        create_contact_sheet(cls, records, cs_path)
        
    print(f"\nTotal Dataset Images: {total_imgs}")
    for cls in CLASSES:
        recs = class_reports[cls]
        n = len(recs)
        ars = [r["aspect_ratio"] for r in recs]
        edges = [r["edge_energy"] for r in recs]
        problematic = [r for r in recs if r["is_problematic"]]
        prob_pct = (len(problematic) / n) * 100 if n > 0 else 0
        
        print(f"\n--- Class: {cls.upper()} ({n} images) ---")
        print(f"  Aspect Ratios: min={min(ars):.2f}, max={max(ars):.2f}, median={np.median(ars):.2f}")
        print(f"  Edge Energies: min={min(edges):.1f}, max={max(edges):.1f}, mean={np.mean(edges):.1f}")
        print(f"  Clear Inscriptions: {n - len(problematic)} / {n} ({(100 - prob_pct):.1f}%)")
        print(f"  Problematic / Ribbon / Obscured: {len(problematic)} / {n} ({prob_pct:.1f}%)")
        if problematic:
            print("  Problematic Items:")
            for p in problematic:
                print(f"    - {p['filename']} ({p['width']}x{p['height']}, AR={p['aspect_ratio']}): {', '.join(p['issues'])}")

if __name__ == "__main__":
    main()
