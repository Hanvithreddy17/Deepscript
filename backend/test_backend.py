"""
DeepScript — FastAPI Backend Test Suite
========================================
Tests the FastAPI backend endpoints (/health, /classes, /predict)
using FastAPI TestClient and sample dataset images.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app


def test_backend_lifespan_and_endpoints():
    """Validates full backend lifecycle, health check, class discovery, and inference."""
    with TestClient(app) as client:
        # 1. Test Root
        response = client.get("/")
        assert response.status_code == 200, f"Root failed: {response.text}"
        data = response.json()
        assert data["service"] == "DeepScript Inference API"
        assert data["status"] == "online"

        # 2. Test Health Check
        health_resp = client.get("/health")
        assert health_resp.status_code == 200, f"Health failed: {health_resp.text}"
        health_data = health_resp.json()
        assert health_data["status"] == "healthy"
        assert health_data["num_classes"] == 62
        assert "model_info" in health_data
        print(f"Health check passed: {health_data['device']}, 62 classes active.")

        # 3. Test Classes List
        classes_resp = client.get("/classes")
        assert classes_resp.status_code == 200, f"Classes failed: {classes_resp.text}"
        classes_data = classes_resp.json()
        assert classes_data["total_classes"] == 62
        assert len(classes_data["classes"]) == 62
        assert "ka" in classes_data["classes"]
        assert "a" in classes_data["classes"]
        print(f"Classes endpoint passed: {classes_data['classes'][:5]}...")

        # 4. Test Predict Endpoint with a Sample Dataset Image
        # Find any valid sample image in dataset/dataset/
        dataset_dir = PROJECT_ROOT / "dataset" / "dataset"
        sample_img_paths = list(dataset_dir.glob("*/*.png")) + list(dataset_dir.glob("*/*.jpg"))
        
        if sample_img_paths:
            sample_path = sample_img_paths[0]
            true_class = sample_path.parent.name
            print(f"Testing /predict with sample: {sample_path.name} (ground truth: {true_class})")

            with open(sample_path, "rb") as img_file:
                files = {"file": (sample_path.name, img_file, "image/png")}
                predict_resp = client.post("/predict?top_k=5", files=files)

            assert predict_resp.status_code == 200, f"Predict failed: {predict_resp.text}"
            pred_data = predict_resp.json()

            assert "script" in pred_data
            assert "confidence" in pred_data
            assert "candidates" in pred_data
            assert len(pred_data["candidates"]) == 5
            assert "execution_time_ms" in pred_data
            assert pred_data["source"] == "live"
            assert 0.0 <= pred_data["confidence"] <= 1.0

            print(f"Prediction successful: predicted '{pred_data['script']}' with confidence {pred_data['confidence']:.4f} in {pred_data['execution_time_ms']}ms")
            print(f"Top candidates: {pred_data['candidates']}")
        else:
            print("Warning: No dataset images found for inference test.")


if __name__ == "__main__":
    print("Running DeepScript Backend Test Suite...")
    test_backend_lifespan_and_endpoints()
    print("\nAll backend integration tests PASSED successfully!")
