"""
DeepScript — FastAPI Backend Test Suite
========================================
Tests the FastAPI backend endpoints (/health, /classes, /predict)
using FastAPI TestClient and sample whole-inscription dataset images.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app

EXPECTED_CLASSES = ["brahmi", "grantha", "gupta", "kadamba", "kharosthi"]


def test_backend_lifespan_and_endpoints():
    """Validates full backend lifecycle, health check, class discovery, and inference."""
    with TestClient(app) as client:
        # 1. Test Root
        response = client.get("/")
        assert response.status_code == 200, f"Root failed: {response.text}"
        data = response.json()
        assert data["service"] == "DeepScript Inference API"
        assert data["status"] == "online"
        print("Root endpoint test passed.")

        # 2. Test Health Check
        health_resp = client.get("/health")
        assert health_resp.status_code == 200, f"Health failed: {health_resp.text}"
        health_data = health_resp.json()
        assert health_data["status"] == "healthy"
        assert health_data["num_classes"] == 5
        assert health_data["vit_loaded"] is True
        assert set(health_data["classes"]) == set(EXPECTED_CLASSES)
        print(f"Health check passed: {health_data['device']}, 5 MVP classes active.")

        # 3. Test Classes List
        classes_resp = client.get("/classes")
        assert classes_resp.status_code == 200, f"Classes failed: {classes_resp.text}"
        classes_data = classes_resp.json()
        assert classes_data["count"] == 5
        assert set(classes_data["classes"]) == set(EXPECTED_CLASSES)
        for cls in EXPECTED_CLASSES:
            assert cls in classes_data["metadata"]
        print(f"Classes endpoint passed: {classes_data['classes']}.")

        # 4. Test Predict Endpoint with Sample Whole-Inscription Images
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
                predict_resp = client.post("/predict?top_k=5", files=files)

            assert predict_resp.status_code == 200, f"Predict failed for {sample_path.name}: {predict_resp.text}"
            pred_data = predict_resp.json()

            assert "script" in pred_data
            assert pred_data["script"] in EXPECTED_CLASSES
            assert "confidence" in pred_data
            assert 0.0 <= pred_data["confidence"] <= 1.0
            assert "candidates" in pred_data
            assert len(pred_data["candidates"]) == 5
            assert "execution_time_ms" in pred_data
            assert pred_data["source"] == "vit_b16_whole_inscription"
            assert "details" in pred_data
            assert "name" in pred_data["details"]

            print(f"  [Predict: {target_cls}] Sample '{sample_path.name[:35]}...' -> Predicted '{pred_data['script']}' (Conf: {pred_data['confidence']:.4f}, Latency: {pred_data['execution_time_ms']}ms)")
            tested_count += 1

        assert tested_count >= 3, f"Expected at least 3 classes to be tested, got {tested_count}"
        print(f"Tested /predict across {tested_count} inscription classes successfully.")

        # 5. Test Error Handling: Empty File Upload
        empty_resp = client.post("/predict?top_k=5", files={"file": ("empty.png", b"", "image/png")})
        assert empty_resp.status_code == 400, f"Expected 400 for empty file, got: {empty_resp.status_code}"
        assert "empty" in empty_resp.json()["detail"].lower()
        print("Empty file upload test passed (HTTP 400).")

        # 6. Test Error Handling: Corrupt / Non-Image File Upload
        corrupt_resp = client.post("/predict?top_k=5", files={"file": ("corrupt.txt", b"not an image data string", "text/plain")})
        assert corrupt_resp.status_code == 400, f"Expected 400 for corrupt file, got: {corrupt_resp.status_code}"
        print("Corrupt file upload test passed (HTTP 400).")

        # 7. Test Parameter Bounds: Invalid top_k
        invalid_k_resp = client.post("/predict?top_k=0", files={"file": ("dummy.png", b"GIF89a...", "image/png")})
        assert invalid_k_resp.status_code == 422, f"Expected 422 for top_k=0, got: {invalid_k_resp.status_code}"
        print("Invalid top_k validation test passed (HTTP 422).")


if __name__ == "__main__":
    print("Running DeepScript Backend Test Suite...")
    test_backend_lifespan_and_endpoints()
    print("\nAll backend integration tests PASSED successfully!")
