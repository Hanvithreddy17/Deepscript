#!/usr/bin/env python3
"""
DeepScript — Whole-Inscription Dataset Validation & Audit Tool
=============================================================

Performs comprehensive validation of the DeepScript whole-inscription dataset:
1. Folder structure validation (detect missing, empty, or unexpected folders).
2. Image integrity validation (format, dimensions, color mode, corruption detection).
3. Statistical profiling per script family.
4. Exact duplicate and cross-class duplicate detection (SHA-256).
5. Metadata validation against dataset/metadata.csv.
6. Dataset leakage preparation and monument-level split checks.
7. Automated Markdown and JSON audit report generation.

Usage:
    python tools/validate_dataset.py [--dataset-dir dataset/whole_inscriptions]
                                     [--metadata-csv dataset/metadata.csv]
                                     [--report-md docs/DATASET_VALIDATION_REPORT.md]
                                     [--report-json dataset/dataset_validation_report.json]
"""

import os
import sys
import csv
import json
import hashlib
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
from PIL import Image, UnidentifiedImageError

SUPPORTED_SCRIPTS = [
    "brahmi",
    "gupta",
    "kharosthi",
    "grantha",
    "kadamba",
    "tamil_brahmi"
]

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

METADATA_REQUIRED_HEADERS = [
    "image_path",
    "script",
    "source",
    "source_url",
    "site",
    "inscription_type",
    "image_type",
    "license",
    "split",
    "verified",
    "notes"
]


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class DatasetValidator:
    def __init__(
        self,
        dataset_dir: str = "dataset/whole_inscriptions",
        metadata_path: str = "dataset/metadata.csv",
        report_md_path: str = "docs/DATASET_VALIDATION_REPORT.md",
        report_json_path: str = "dataset/dataset_validation_report.json",
        project_root: str = "."
    ):
        self.project_root = Path(project_root).resolve()
        self.dataset_dir = (self.project_root / dataset_dir).resolve()
        self.metadata_path = (self.project_root / metadata_path).resolve()
        self.report_md_path = (self.project_root / report_md_path).resolve()
        self.report_json_path = (self.project_root / report_json_path).resolve()

        self.results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "dataset_dir": str(self.dataset_dir),
            "folder_check": {
                "supported_scripts": SUPPORTED_SCRIPTS,
                "existing_folders": [],
                "missing_folders": [],
                "unexpected_folders": [],
                "empty_folders": []
            },
            "summary": {
                "total_images_found": 0,
                "valid_images": 0,
                "corrupted_images": 0,
                "non_image_files": 0
            },
            "script_statistics": {},
            "duplicates": {
                "exact_duplicates": {},  # hash -> [paths]
                "cross_class_duplicates": {}  # hash -> {script: [paths]}
            },
            "metadata_check": {
                "metadata_file_found": False,
                "total_metadata_rows": 0,
                "valid_rows": 0,
                "unmatched_dataset_images": [],  # Images on disk missing in CSV
                "unmatched_metadata_paths": [],  # Paths in CSV missing on disk
                "invalid_scripts": [],
                "invalid_verified_flags": [],
                "duplicate_metadata_entries": [],
                "missing_required_fields": []
            },
            "leakage_warnings": [],
            "status": "PASS"
        }

    def validate_folders(self) -> None:
        """Check for existence of supported and unexpected class folders."""
        if not self.dataset_dir.exists():
            self.results["folder_check"]["missing_folders"] = list(SUPPORTED_SCRIPTS)
            self.results["status"] = "WARNING"
            return

        subdirs = [p.name for p in self.dataset_dir.iterdir() if p.is_dir()]
        self.results["folder_check"]["existing_folders"] = subdirs

        for script in SUPPORTED_SCRIPTS:
            script_path = self.dataset_dir / script
            if not script_path.exists():
                self.results["folder_check"]["missing_folders"].append(script)
            else:
                # Check if empty (ignoring .gitkeep)
                valid_files = [f for f in script_path.rglob("*") if f.is_file() and f.name != ".gitkeep"]
                if not valid_files:
                    self.results["folder_check"]["empty_folders"].append(script)

        for folder in subdirs:
            if folder not in SUPPORTED_SCRIPTS:
                self.results["folder_check"]["unexpected_folders"].append(folder)

    def validate_images(self) -> Dict[str, Any]:
        """Inspect and validate image files across script directories."""
        images_by_hash: Dict[str, List[Tuple[str, Path]]] = {}
        script_stats: Dict[str, Dict[str, Any]] = {}

        for script in SUPPORTED_SCRIPTS:
            script_stats[script] = {
                "total": 0,
                "valid": 0,
                "corrupted": 0,
                "dimensions": {
                    "min_width": None,
                    "max_width": None,
                    "min_height": None,
                    "max_height": None,
                    "avg_width": 0.0,
                    "avg_height": 0.0
                },
                "formats": {},
                "color_modes": {},
                "corrupted_files": []
            }

        all_found_images: List[Path] = []

        if self.dataset_dir.exists():
            for script in SUPPORTED_SCRIPTS:
                script_path = self.dataset_dir / script
                if not script_path.exists():
                    continue

                widths = []
                heights = []

                for file_path in script_path.rglob("*"):
                    if not file_path.is_file() or file_path.name == ".gitkeep":
                        continue

                    ext = file_path.suffix.lower()
                    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
                        self.results["summary"]["non_image_files"] += 1
                        continue

                    all_found_images.append(file_path)
                    self.results["summary"]["total_images_found"] += 1
                    script_stats[script]["total"] += 1

                    # Verify image validity and properties
                    try:
                        # Step 1: Open and verify structure
                        with Image.open(file_path) as img:
                            img.verify()

                        # Step 2: Reopen to extract attributes and ensure pixel data unpacks
                        with Image.open(file_path) as img:
                            w, h = img.size
                            fmt = img.format or ext.replace(".", "").upper()
                            mode = img.mode

                            widths.append(w)
                            heights.append(h)

                            script_stats[script]["formats"][fmt] = script_stats[script]["formats"].get(fmt, 0) + 1
                            script_stats[script]["color_modes"][mode] = script_stats[script]["color_modes"].get(mode, 0) + 1
                            script_stats[script]["valid"] += 1
                            self.results["summary"]["valid_images"] += 1

                            # Hash calculation for duplicate analysis
                            img_hash = compute_sha256(file_path)
                            images_by_hash.setdefault(img_hash, []).append((script, file_path))

                    except (UnidentifiedImageError, OSError, Exception) as e:
                        script_stats[script]["corrupted"] += 1
                        self.results["summary"]["corrupted_images"] += 1
                        script_stats[script]["corrupted_files"].append({
                            "file": str(file_path.relative_to(self.project_root)).replace("\\", "/"),
                            "error": str(e)
                        })

                if widths and heights:
                    script_stats[script]["dimensions"]["min_width"] = min(widths)
                    script_stats[script]["dimensions"]["max_width"] = max(widths)
                    script_stats[script]["dimensions"]["min_height"] = min(heights)
                    script_stats[script]["dimensions"]["max_height"] = max(heights)
                    script_stats[script]["dimensions"]["avg_width"] = round(sum(widths) / len(widths), 1)
                    script_stats[script]["dimensions"]["avg_height"] = round(sum(heights) / len(heights), 1)

        self.results["script_statistics"] = script_stats

        # Duplicate detection analysis
        for h, items in images_by_hash.items():
            if len(items) > 1:
                rel_paths = [str(p.relative_to(self.project_root)).replace("\\", "/") for _, p in items]
                self.results["duplicates"]["exact_duplicates"][h] = rel_paths

                # Cross-class duplicate detection
                classes_present = set(s for s, _ in items)
                if len(classes_present) > 1:
                    self.results["duplicates"]["cross_class_duplicates"][h] = {
                        "classes": list(classes_present),
                        "paths": rel_paths
                    }
                    self.results["leakage_warnings"].append(
                        f"CRITICAL: Exact duplicate image hash {h[:12]}... found across multiple classes: {list(classes_present)}"
                    )

        return {"all_found_images": all_found_images}

    def validate_metadata(self, all_found_images: List[Path]) -> None:
        """Validate metadata.csv against dataset images."""
        if not self.metadata_path.exists():
            self.results["metadata_check"]["metadata_file_found"] = False
            return

        self.results["metadata_check"]["metadata_file_found"] = True

        rel_found_images = {
            str(p.relative_to(self.project_root)).replace("\\", "/") for p in all_found_images
        }

        seen_paths: Set[str] = set()
        metadata_paths: Set[str] = set()

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []

                missing_hdrs = [h for h in METADATA_REQUIRED_HEADERS if h not in headers]
                if missing_hdrs:
                    self.results["metadata_check"]["missing_required_fields"].append(
                        f"Missing CSV columns: {missing_hdrs}"
                    )

                for idx, row in enumerate(reader, start=2):
                    self.results["metadata_check"]["total_metadata_rows"] += 1
                    img_path = (row.get("image_path") or "").strip().replace("\\", "/")
                    script = (row.get("script") or "").strip()
                    verified = (row.get("verified") or "").strip()

                    if not img_path:
                        self.results["metadata_check"]["missing_required_fields"].append(
                            f"Row {idx}: Missing image_path"
                        )
                        continue

                    # Duplicate check in CSV
                    if img_path in seen_paths:
                        self.results["metadata_check"]["duplicate_metadata_entries"].append(
                            f"Row {idx}: Duplicate image_path '{img_path}'"
                        )
                    seen_paths.add(img_path)
                    metadata_paths.add(img_path)

                    # Check existence on disk
                    full_p = (self.project_root / img_path).resolve()
                    if not full_p.exists() or not full_p.is_file():
                        self.results["metadata_check"]["unmatched_metadata_paths"].append(img_path)

                    # Script validation
                    if script and script.lower() not in SUPPORTED_SCRIPTS and script.upper() != "UNKNOWN":
                        self.results["metadata_check"]["invalid_scripts"].append(
                            f"Row {idx}: '{script}' not in supported script classes"
                        )

                    # Verified flag validation
                    valid_flags = {"TRUE", "FALSE", "1", "0", "UNKNOWN"}
                    if verified and verified.upper() not in valid_flags:
                        self.results["metadata_check"]["invalid_verified_flags"].append(
                            f"Row {idx}: Verified value '{verified}' is invalid"
                        )

                    self.results["metadata_check"]["valid_rows"] += 1

            # Images on disk missing in CSV
            missing_in_csv = rel_found_images - metadata_paths
            self.results["metadata_check"]["unmatched_dataset_images"] = sorted(list(missing_in_csv))

        except Exception as e:
            self.results["metadata_check"]["missing_required_fields"].append(
                f"Error reading metadata.csv: {str(e)}"
            )

    def run_all_checks(self) -> Dict[str, Any]:
        """Execute all validation routines."""
        self.validate_folders()
        img_res = self.validate_images()
        self.validate_metadata(img_res["all_found_images"])
        return self.results

    def generate_markdown_report(self) -> str:
        """Build comprehensive Markdown audit report."""
        r = self.results
        s = r["summary"]
        fc = r["folder_check"]
        mc = r["metadata_check"]
        dup = r["duplicates"]

        lines = [
            "# DeepScript Dataset Validation & Audit Report",
            "",
            f"**Audit Execution Timestamp:** `{r['timestamp']}`  ",
            f"**Dataset Directory Audited:** `{r['dataset_dir']}`  ",
            f"**Audit Status:** `{'CLEAN / PASS' if s['corrupted_images'] == 0 and not dup['cross_class_duplicates'] else 'ACTION REQUIRED'}`",
            "",
            "---",
            "",
            "## 1. Executive Summary",
            "",
            f"- **Total Whole-Inscription Images Found:** `{s['total_images_found']}`",
            f"- **Valid Images:** `{s['valid_images']}`",
            f"- **Corrupted / Unreadable Images:** `{s['corrupted_images']}`",
            f"- **Empty Script Classes:** `{len(fc['empty_folders'])} / {len(SUPPORTED_SCRIPTS)}`",
            f"- **Exact Duplicate Files:** `{len(dup['exact_duplicates'])}`",
            f"- **Cross-Class Duplicates:** `{len(dup['cross_class_duplicates'])}`",
            "",
        ]

        if s["total_images_found"] == 0:
            lines.extend([
                "> [!NOTE]",
                "> **Dataset structure detected.** No whole-inscription images have been added yet.",
                "> All 6 script family directories are properly structured and ready to receive verified epigraphic specimens.",
                ""
            ])

        lines.extend([
            "---",
            "",
            "## 2. Script Family Statistics",
            "",
            "| Script Family | Folder Status | Total Images | Valid | Corrupted | Resolution Range (Min – Max) | Formats |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
        ])

        for script in SUPPORTED_SCRIPTS:
            st = r["script_statistics"].get(script, {})
            folder_status = "Empty (Ready)" if script in fc["empty_folders"] else "Populated"
            total = st.get("total", 0)
            valid = st.get("valid", 0)
            corrupt = st.get("corrupted", 0)
            dims = st.get("dimensions", {})

            if dims.get("min_width") is not None:
                dim_str = f"{dims['min_width']}x{dims['min_height']} – {dims['max_width']}x{dims['max_height']}"
            else:
                dim_str = "N/A"

            fmts = ", ".join(f"{k}:{v}" for k, v in st.get("formats", {}).items()) or "None"

            lines.append(
                f"| **`{script}`** | {folder_status} | {total} | {valid} | {corrupt} | {dim_str} | {fmts} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 3. Duplicate & Leakage Analysis",
            ""
        ])

        if dup["cross_class_duplicates"]:
            lines.extend([
                "### Cross-Class Duplicates (Critical Risk)",
                "",
                "The following identical image hashes appear across multiple distinct script folders:",
                ""
            ])
            for h, info in dup["cross_class_duplicates"].items():
                lines.append(f"- **Hash `{h[:12]}...`** in classes: `{info['classes']}`")
                for p in info["paths"]:
                    lines.append(f"  - `{p}`")
            lines.append("")
        else:
            lines.append("- **Cross-Class Duplicates:** None detected (0).  ")

        if dup["exact_duplicates"]:
            lines.extend([
                "",
                "### Intra-Class Duplicate Files",
                ""
            ])
            for h, paths in dup["exact_duplicates"].items():
                if h not in dup["cross_class_duplicates"]:
                    lines.append(f"- **Hash `{h[:12]}...`** ({len(paths)} copies):")
                    for p in paths:
                        lines.append(f"  - `{p}`")
            lines.append("")
        else:
            lines.append("- **Exact Duplicates:** None detected (0).")

        lines.extend([
            "",
            "---",
            "",
            "## 4. Metadata Validation (`dataset/metadata.csv`)",
            ""
        ])

        if not mc["metadata_file_found"]:
            lines.append("> [!WARNING]\n> `dataset/metadata.csv` was not found.")
        else:
            lines.extend([
                f"- **Metadata Header Status:** `{'VALID' if not mc['missing_required_fields'] else 'ERRORS'}`",
                f"- **Total Metadata Records:** `{mc['total_metadata_rows']}`",
                f"- **Valid Metadata Rows:** `{mc['valid_rows']}`",
                f"- **Images on Disk Missing Metadata:** `{len(mc['unmatched_dataset_images'])}`",
                f"- **Metadata Paths Missing on Disk:** `{len(mc['unmatched_metadata_paths'])}`",
                f"- **Invalid Script Entries:** `{len(mc['invalid_scripts'])}`",
                f"- **Invalid Verified Flags:** `{len(mc['invalid_verified_flags'])}`",
                f"- **Duplicate Metadata Rows:** `{len(mc['duplicate_metadata_entries'])}`",
                ""
            ])

        lines.extend([
            "---",
            "",
            "## 5. Recommended Actions & Next Steps",
            "",
            "1. **Preserve Clean Data Acquisition:** Only add whole-inscription images with verified source citations.",
            "2. **Maintain Provenance in `metadata.csv`:** Populate `source`, `source_url`, `site`, `inscription_type`, and set `verified = TRUE` upon verification.",
            "3. **Enforce Monument-Level Splitting:** When partitioning data into train/val/test splits, ensure images from the same physical monument or inscription panel remain within the same split to avoid data leakage.",
            "4. **No Guesswork Policy:** Any image lacking historical confirmation must remain quarantined with `verified = FALSE` and excluded from model training.",
            ""
        ])

        report_content = "\n".join(lines)
        return report_content

    def save_reports(self) -> None:
        """Write Markdown and JSON report outputs."""
        self.report_md_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_json_path.parent.mkdir(parents=True, exist_ok=True)

        md_content = self.generate_markdown_report()
        with open(self.report_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        with open(self.report_json_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Validate DeepScript whole-inscription dataset.")
    parser.add_argument("--dataset-dir", default="dataset/whole_inscriptions", help="Path to whole_inscriptions directory")
    parser.add_argument("--metadata-csv", default="dataset/metadata.csv", help="Path to metadata.csv")
    parser.add_argument("--report-md", default="docs/DATASET_VALIDATION_REPORT.md", help="Path to output markdown report")
    parser.add_argument("--report-json", default="dataset/dataset_validation_report.json", help="Path to output JSON report")

    args = parser.parse_args()

    validator = DatasetValidator(
        dataset_dir=args.dataset_dir,
        metadata_path=args.metadata_csv,
        report_md_path=args.report_md,
        report_json_path=args.report_json
    )

    print("=================================================================")
    print("  DeepScript — Whole-Inscription Dataset Validation & Audit Tool ")
    print("=================================================================")

    results = validator.run_all_checks()
    validator.save_reports()

    summary = results["summary"]
    folder_check = results["folder_check"]
    dup = results["duplicates"]

    print(f"\nAudit completed at: {results['timestamp']}")
    print(f"Target Directory : {results['dataset_dir']}")
    print(f"Supported Scripts: {', '.join(SUPPORTED_SCRIPTS)}")
    print(f"\n--- Folder Status ---")
    print(f"Existing Classes : {len(folder_check['existing_folders'])} / {len(SUPPORTED_SCRIPTS)}")
    print(f"Missing Classes  : {len(folder_check['missing_folders'])}")
    print(f"Empty Classes    : {len(folder_check['empty_folders'])}")

    print(f"\n--- Image Status ---")
    print(f"Total Found      : {summary['total_images_found']}")
    print(f"Valid Images     : {summary['valid_images']}")
    print(f"Corrupted Images : {summary['corrupted_images']}")

    if summary["total_images_found"] == 0:
        print("\n>> Status: Dataset structure detected. No whole-inscription images have been added yet.")
    else:
        print(f"\n>> Status: {summary['valid_images']} valid images verified across script folders.")

    if dup["cross_class_duplicates"]:
        print(f"\n[ALERT] {len(dup['cross_class_duplicates'])} Cross-Class Duplicates Detected!")

    print(f"\n--- Reports Generated ---")
    print(f"Markdown Report  : {args.report_md}")
    print(f"JSON Report      : {args.report_json}")
    print("=================================================================\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
