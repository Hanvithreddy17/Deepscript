#!/usr/bin/env python3
"""
DeepScript — Generate Interactive ROI Crop Previews for Faculty Demo Failure
============================================================================
Creates:
1. `faculty_coordinate_grid.png`: Full image with precise 50px/10px pixel grid rulers.
2. `candidate_roi_boxes_preview.png`: Five distinct epigraphically informed candidate crop bounding boxes.
Copies outputs to the artifact directory for rendering.
"""

from pathlib import Path
import shutil
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = PROJECT_ROOT / "experiments" / "faculty_failure"
ARTIFACT_DIR = Path(r"C:\Users\hanvi\.gemini\antigravity-ide\brain\0ea7807c-aab3-4275-a481-251287571760")

def main():
    img_path = EXP_DIR / "faculty_tamil_brahmi.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Missing image at {img_path}")

    img_bgr = cv2.imread(str(img_path))
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. Coordinate Grid Reference Image
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    ax.imshow(img_rgb)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0) # Invert Y for image coordinate convention

    # Set grid ticks every 50 pixels
    ax.set_xticks(range(0, w + 1, 50))
    ax.set_yticks(range(0, h + 1, 50))
    ax.set_xticks(range(0, w + 1, 10), minor=True)
    ax.set_yticks(range(0, h + 1, 10), minor=True)

    ax.grid(which="major", color="#00E5FF", linestyle="-", linewidth=0.7, alpha=0.7)
    ax.grid(which="minor", color="#00E5FF", linestyle=":", linewidth=0.3, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=8, colors="white", top=True, labeltop=True)
    ax.set_title(f"Faculty Failure Image: Pixel Coordinate Grid (Width: {w}px, Height: {h}px)", fontsize=11, weight="bold", pad=15)

    grid_path = EXP_DIR / "faculty_coordinate_grid.png"
    plt.tight_layout()
    plt.savefig(grid_path, bbox_inches="tight", facecolor="#141418")
    plt.close(fig)

    # 2. Candidate ROI Bounding Boxes Preview
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    ax.imshow(img_rgb)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    candidates = [
        {
            "id": "ROI-A",
            "name": "Candidate A: Full Text Field (Lines 1-7, excluding top stone ledge)",
            "box": (10, 20, 730, 365),
            "color": "#00E5FF", # Cyan
        },
        {
            "id": "ROI-B",
            "name": "Candidate B: Upper Text Body (Lines 1-4)",
            "box": (10, 20, 730, 220),
            "color": "#76FF03", # Green
        },
        {
            "id": "ROI-C",
            "name": "Candidate C: Central Core Block (Lines 2-5)",
            "box": (20, 65, 720, 275),
            "color": "#FFD600", # Yellow
        },
        {
            "id": "ROI-D",
            "name": "Candidate D: Lower Text Body (Lines 4-7)",
            "box": (10, 175, 730, 365),
            "color": "#FF3D00", # Orange-Red
        },
        {
            "id": "ROI-E",
            "name": "Candidate E: Single Text Line Ribbon (Line 3)",
            "box": (10, 115, 730, 180),
            "color": "#D500F9", # Magenta
        },
    ]

    for cand in candidates:
        x1, y1, x2, y2 = cand["box"]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2.0, edgecolor=cand["color"], facecolor="none", linestyle="-")
        ax.add_patch(rect)
        ax.text(x1 + 8, y1 + 18, f"{cand['id']}: {cand['box']}", color="black", fontsize=8, weight="bold",
                bbox=dict(facecolor=cand["color"], alpha=0.9, edgecolor="none", pad=2))

    ax.set_title("Faculty Failure Image: Candidate Inscription ROI Bounding Boxes", fontsize=11, weight="bold", pad=12)
    ax.axis("off")

    roi_preview_path = EXP_DIR / "candidate_roi_boxes_preview.png"
    plt.tight_layout()
    plt.savefig(roi_preview_path, bbox_inches="tight", facecolor="#141418")
    plt.close(fig)

    # Copy files to artifacts directory
    if ARTIFACT_DIR.exists():
        shutil.copy2(grid_path, ARTIFACT_DIR / grid_path.name)
        shutil.copy2(roi_preview_path, ARTIFACT_DIR / roi_preview_path.name)

    print(f"Generated:")
    print(f"  {grid_path}")
    print(f"  {roi_preview_path}")

if __name__ == "__main__":
    main()
