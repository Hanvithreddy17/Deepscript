"""
DeepScript — FastAPI Backend Test Suite
========================================
Comprehensive test suite validating:
  1. Valid, supported ancient inscription images.
  2. Random natural photographs.
  3. Blank or nearly blank images (abstention / rejection).
  4. Corrupted image files (graceful 400 error handling).
  5. Unsupported file formats (graceful 400 error handling).
  6. Missing upload / empty files (graceful 400 / 422 error handling).
  7. Backend inference errors (graceful 500 error handling).
"""

import io
import sys
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
from backend.vit_inference import vit_inference_engine

EXPECTED_CLASSES = vit_inference_engine.classes if vit_inference_engine.classes else ["brahmi", "grantha", "gupta", "kadamba", "kharosthi", "tamil_brahmi"]


def test_root_and_health_endpoints(client: TestClient):
    """Verifies service health check and class discovery."""
    root_resp = client.get("/")
    assert root_resp.status_code == 200, f"Root failed: {root_resp.text}"
    root_data = root_resp.json()
    assert root_data["service"] == "DeepScript Inference API"
    assert root_data["status"] == "online"

    active_classes = vit_inference_engine.classes
    health_resp = client.get("/health")
    assert health_resp.status_code == 200, f"Health failed: {health_resp.text}"
    health_data = health_resp.json()
    assert health_data["status"] == "healthy"
    assert health_data["num_classes"] == len(active_classes)
    assert health_data["vit_loaded"] is True
    assert set(health_data["classes"]) == set(active_classes)

    classes_resp = client.get("/classes")
    assert classes_resp.status_code == 200, f"Classes failed: {classes_resp.text}"
    classes_data = classes_resp.json()
    assert classes_data["count"] == len(active_classes)
    assert set(classes_data["classes"]) == set(active_classes)
    print("✓ Health, root, and classes discovery endpoints passed.")


def test_case_1_valid_supported_inscriptions(client: TestClient):
    """Case 1: Valid, supported whole-inscription images across all supported classes."""
    dataset_dir = PROJECT_ROOT / "dataset" / "whole_inscriptions"
    tested_count = 0

    for target_cls in EXPECTED_CLASSES:
        class_folder = dataset_dir / target_cls
        sample_img_paths = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))
        if not sample_img_paths:
            continue

        sample_path = sample_img_paths[0]
        with open(sample_path, "rb") as img_file:
            files = {"file": (sample_path.name, img_file, "image/jpeg")}
            resp = client.post("/predict?top_k=5", files=files)

        assert resp.status_code == 200, f"Predict failed for {sample_path.name}: {resp.text}"
        data = resp.json()

        assert data["status"] == "success", f"Expected 'success', got: {data.get('status')}"
        assert data["decision"] == "accepted", f"Expected 'accepted', got: {data.get('decision')}"
        assert data["script"] in EXPECTED_CLASSES
        assert data["confidence"] >= 0.25
        assert len(data["candidates"]) == min(5, len(EXPECTED_CLASSES))
        assert "execution_time_ms" in data
        assert data["source"] == "vit_b16_whole_inscription"
        assert "details" in data
        assert "name" in data["details"]

        print(f"  [Case 1: Valid Inscription - {target_cls}] Sample '{sample_path.name[:30]}...' -> '{data['script']}' (Conf: {data['confidence']:.4f}, {data['execution_time_ms']}ms)")
        tested_count += 1

    assert tested_count >= 3, f"Expected at least 3 classes to be tested, got {tested_count}"
    print(f"✓ Case 1 passed: Valid whole-inscriptions recognized across {tested_count} classes.")


def test_case_2_random_natural_photograph(client: TestClient):
    """Case 2: Random natural photograph / synthetic photographic scene."""
    # Generate a photographic scene (gradient sky + natural texture)
    grad = np.linspace(40, 220, 224, dtype=np.uint8)
    photo_arr = np.stack([
        np.tile(grad, (224, 1)),
        np.tile(grad[::-1], (224, 1)),
        np.full((224, 224), 140, dtype=np.uint8)
    ], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(photo_arr).save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("natural_landscape_photo.jpg", buf, "image/jpeg")}
    resp = client.post("/predict?top_k=5", files=files)

    assert resp.status_code == 200, f"Expected 200 OK for readable photo, got: {resp.status_code}"
    data = resp.json()
    assert "status" in data
    assert "confidence" in data
    assert "candidates" in data
    assert "execution_time_ms" in data
    print(f"✓ Case 2 passed: Random natural photograph processed gracefully (Status: {data['status']}, Top: {data.get('script')}, Conf: {data.get('confidence')}).")


def test_case_3_blank_or_nearly_blank_image(client: TestClient):
    """Case 3: Blank or nearly blank images (must return 'Script not detected')."""
    # 3a. Pure White Image
    white_img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    buf_white = io.BytesIO()
    white_img.save(buf_white, format="PNG")
    buf_white.seek(0)

    resp_white = client.post("/predict?top_k=5", files={"file": ("blank_white.png", buf_white, "image/png")})
    assert resp_white.status_code == 200, f"Expected 200 for blank image, got: {resp_white.status_code}"
    data_white = resp_white.json()
    assert data_white["status"] == "script_not_detected"
    assert data_white["decision"] == "rejected"
    assert data_white["script"] == "Script not detected"
    assert data_white["script_family"] == "Script not detected"
    assert "blank/uniform" in data_white["rejection_reason"]

    # 3b. Pure Black Image
    black_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
    buf_black = io.BytesIO()
    black_img.save(buf_black, format="PNG")
    buf_black.seek(0)

    resp_black = client.post("/predict?top_k=5", files={"file": ("blank_black.png", buf_black, "image/png")})
    assert resp_black.status_code == 200
    data_black = resp_black.json()
    assert data_black["status"] == "script_not_detected"
    assert data_black["decision"] == "rejected"
    assert data_black["script"] == "Script not detected"
    print("✓ Case 3 passed: Blank and nearly blank images correctly rejected as 'Script not detected'.")


def test_case_4_corrupted_image_file(client: TestClient):
    """Case 4: Corrupted / damaged image file payload."""
    corrupt_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRcorrupted_garbage_stream_not_decodable"
    files = {"file": ("corrupted_inscription.png", corrupt_bytes, "image/png")}
    resp = client.post("/predict?top_k=5", files=files)

    assert resp.status_code == 400, f"Expected 400 for corrupt image, got: {resp.status_code}"
    detail = resp.json().get("detail", "")
    assert "invalid input image" in detail.lower() or "unreadable" in detail.lower()
    print("✓ Case 4 passed: Corrupted image file correctly rejected with HTTP 400 'Invalid input image'.")


def test_case_5_unsupported_file_format(client: TestClient):
    """Case 5: Unsupported non-image file format."""
    text_bytes = b"This is a plain text transcript, not a whole-inscription image."
    files = {"file": ("inscription_notes.txt", text_bytes, "text/plain")}
    resp = client.post("/predict?top_k=5", files=files)

    assert resp.status_code == 400, f"Expected 400 for text file, got: {resp.status_code}"
    detail = resp.json().get("detail", "")
    assert "invalid input image" in detail.lower()
    print("✓ Case 5 passed: Unsupported file format correctly rejected with HTTP 400 'Invalid input image'.")


def test_case_6_missing_upload(client: TestClient):
    """Case 6: Missing upload and empty (0-byte) file."""
    # 6a. Empty 0-byte file
    empty_files = {"file": ("empty.png", b"", "image/png")}
    resp_empty = client.post("/predict?top_k=5", files=empty_files)
    assert resp_empty.status_code == 400, f"Expected 400 for empty file, got: {resp_empty.status_code}"
    assert "empty" in resp_empty.json()["detail"].lower()

    # 6b. Missing file field entirely
    resp_missing = client.post("/predict?top_k=5")
    assert resp_missing.status_code == 422, f"Expected 422 for missing required file parameter, got: {resp_missing.status_code}"
    print("✓ Case 6 passed: Empty files and missing upload parameters handled gracefully (HTTP 400 / 422).")


def test_case_7_backend_inference_error(client: TestClient):
    """Case 7: Backend inference pipeline error recovery."""
    dummy_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    # Add minimal edge noise so it passes blank check
    arr = np.array(dummy_img)
    arr[50:150, 50:150] = 20
    dummy_img = Image.fromarray(arr)
    buf = io.BytesIO()
    dummy_img.save(buf, format="PNG")
    buf.seek(0)

    # Mock vit_inference_engine.predict to raise an exception
    with patch.object(vit_inference_engine, "predict", side_effect=RuntimeError("Simulated neural engine timeout")):
        resp = client.post("/predict?top_k=5", files={"file": ("test.png", buf, "image/png")})

    assert resp.status_code == 500, f"Expected 500 for backend failure, got: {resp.status_code}"
    detail = resp.json().get("detail", "")
    assert "vit inference error" in detail.lower()

    # Verify service remains fully operational immediately afterwards
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    print("✓ Case 7 passed: Backend inference errors handled gracefully (HTTP 500) without crashing server.")


def run_all_tests():
    """Runs the complete test suite."""
    print("===============================================================")
    print(" DeepScript Backend Verification & Rejection Test Suite")
    print("===============================================================")
    with TestClient(app) as client:
        test_root_and_health_endpoints(client)
        test_case_1_valid_supported_inscriptions(client)
        test_case_2_random_natural_photograph(client)
        test_case_3_blank_or_nearly_blank_image(client)
        test_case_4_corrupted_image_file(client)
        test_case_5_unsupported_file_format(client)
        test_case_6_missing_upload(client)
        test_case_7_backend_inference_error(client)

    print("\n===============================================================")
    print(" ALL 7 TEST CASES PASSED SUCCESSFULLY!")
    print("===============================================================")


if __name__ == "__main__":
    run_all_tests()
