"""
DeepScript: Comprehensive Verification of Open-Set Rejection Architecture
Tests:
1. All 6 supported classes (sample from each)
2. Uploaded Medieval Tamil stone inscription screenshot
3. Blank images (white, uniform)
4. Ordinary photographs
5. Corrupted files (empty, corrupt bytes, non-image)
"""
import requests
import io
from pathlib import Path
from PIL import Image

BASE_URL = "http://localhost:8000"

def test_health():
    res = requests.get(f"{BASE_URL}/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    data = res.json()
    assert data["num_classes"] == 6, f"Expected 6 classes, got {data['num_classes']}"
    assert "tamil_brahmi" in data["classes"]
    print(f"[PASS] /health -> 6 classes loaded: {data['classes']}")

def test_classes():
    res = requests.get(f"{BASE_URL}/classes")
    assert res.status_code == 200, f"Classes check failed: {res.text}"
    data = res.json()
    assert data["count"] == 6
    print(f"[PASS] /classes -> Count: {data['count']}")

def test_supported_classes():
    print("\n--- Testing Supported Classes ---")
    data_dir = Path("dataset/whole_inscriptions")
    classes = ["brahmi", "tamil_brahmi", "kharosthi", "grantha", "gupta", "kadamba"]
    for cls in classes:
        cls_dir = data_dir / cls
        images = [p for p in cls_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
        assert len(images) > 0, f"No images found for {cls}"
        test_img_path = images[0]
        with open(test_img_path, "rb") as f:
            files = {"file": (test_img_path.name, f.read(), "image/jpeg")}
            res = requests.post(f"{BASE_URL}/predict?top_k=5", files=files)
        assert res.status_code == 200, f"Predict failed for {cls}: {res.text}"
        data = res.json()
        status = data.get("status")
        pred_script = data.get("script")
        conf = data.get("confidence")
        print(f"[{'PASS' if status == 'success' else 'FAIL'}] Class: {cls:13s} -> status={status}, pred={pred_script}, conf={conf:.3f} (file={test_img_path.name[:25]})")
        assert status == "success", f"Class {cls} was unexpectedly rejected: {data}"

def test_user_medieval_tamil():
    print("\n--- Testing Uploaded Medieval Tamil Screenshot ---")
    p_u = Path(r"C:\Users\hanvi\OneDrive\Pictures\Screenshots\Screenshot 2026-10-13 114513.png")
    assert p_u.exists(), f"User screenshot not found at {p_u}"
    with open(p_u, "rb") as f:
        files = {"file": ("Screenshot 2026-10-13 114513.png", f.read(), "image/png")}
        res = requests.post(f"{BASE_URL}/predict?top_k=5", files=files)
    assert res.status_code == 200, f"Predict failed: {res.text}"
    data = res.json()
    status = data.get("status")
    decision = data.get("decision")
    reason = data.get("rejection_reason")
    conf = data.get("confidence")
    print(f"[{'PASS' if status == 'unsupported_script' else 'FAIL'}] Medieval Tamil Screenshot -> status={status}, decision={decision}, conf={conf:.3f}")
    print(f"  Reason: {reason}")
    assert status == "unsupported_script", f"Expected unsupported_script, got {status}"
    assert decision == "rejected"
    assert conf < 0.70, f"Expected calibrated confidence (< 70%), got {conf}"

def test_blank_images():
    print("\n--- Testing Blank / Uniform Images ---")
    img_white = Image.new("RGB", (200, 200), (255, 255, 255))
    buf = io.BytesIO()
    img_white.save(buf, format="PNG")
    files = {"file": ("blank_white.png", buf.getvalue(), "image/png")}
    res = requests.post(f"{BASE_URL}/predict?top_k=5", files=files)
    assert res.status_code == 200
    data = res.json()
    status = data.get("status")
    print(f"[{'PASS' if status == 'script_not_detected' else 'FAIL'}] Blank White -> status={status}, reason={data.get('rejection_reason')}")
    assert status == "script_not_detected"

def test_ordinary_photos():
    print("\n--- Testing Ordinary Non-Inscription Photographs ---")
    pics_dir = Path(r"C:\Users\hanvi\OneDrive\Pictures")
    for name in ["batman-.jpg", "himym.webp", "spidey.jpg"]:
        p = pics_dir / name
        if p.exists():
            with open(p, "rb") as f:
                files = {"file": (p.name, f.read(), "image/jpeg")}
                res = requests.post(f"{BASE_URL}/predict?top_k=5", files=files)
            assert res.status_code == 200
            data = res.json()
            status = data.get("status")
            decision = data.get("decision")
            print(f"[{'PASS' if decision == 'rejected' else 'FAIL'}] Photo: {name:12s} -> status={status}, decision={decision}, reason={data.get('rejection_reason')[:60]}...")
            assert decision == "rejected"

def test_corrupted_and_invalid_files():
    print("\n--- Testing Corrupted / Invalid Files ---")
    # 1. Empty file (0 bytes)
    res_empty = requests.post(f"{BASE_URL}/predict", files={"file": ("empty.png", b"", "image/png")})
    print(f"[{'PASS' if res_empty.status_code == 400 else 'FAIL'}] Empty (0 bytes) -> HTTP {res_empty.status_code}: {res_empty.text}")
    assert res_empty.status_code == 400

    # 2. Corrupted byte stream
    res_corrupt = requests.post(f"{BASE_URL}/predict", files={"file": ("corrupt.jpg", b"NOT_A_VALID_JPEG_HEADER_12345", "image/jpeg")})
    print(f"[{'PASS' if res_corrupt.status_code == 400 else 'FAIL'}] Corrupt bytes -> HTTP {res_corrupt.status_code}: {res_corrupt.text}")
    assert res_corrupt.status_code == 400

    # 3. Degenerate size (<10x10)
    img_tiny = Image.new("RGB", (5, 5), (100, 100, 100))
    buf = io.BytesIO()
    img_tiny.save(buf, format="PNG")
    res_tiny = requests.post(f"{BASE_URL}/predict", files={"file": ("tiny.png", buf.getvalue(), "image/png")})
    print(f"[{'PASS' if res_tiny.status_code == 400 else 'FAIL'}] Tiny 5x5 -> HTTP {res_tiny.status_code}: {res_tiny.text}")
    assert res_tiny.status_code == 400

if __name__ == "__main__":
    test_health()
    test_classes()
    test_supported_classes()
    test_user_medieval_tamil()
    test_blank_images()
    test_ordinary_photos()
    test_corrupted_and_invalid_files()
    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")
