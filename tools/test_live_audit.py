"""
Live Audit Script for DeepScript Tamil-Brahmi Integration
Tests the active running backend at http://localhost:8000 with real inscription images
and rejection test cases.
"""
import sys
import io
import json
import urllib.request
import urllib.error
from pathlib import Path
from PIL import Image
import numpy as np

# Reconfigure stdout for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "http://localhost:8000"


def send_multipart_request(url: str, filename: str, file_bytes: bytes, content_type: str = "image/jpeg"):
    boundary = "----WebKitFormBoundaryDeepScriptTest7MA4YWxkTrZu0gW"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8"))
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    body.extend(file_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(err_body)
        except Exception:
            return e.code, {"detail": err_body}


def run_live_audit():
    print("=" * 80)
    print(" DEEPSCRIPT LIVE INTEGRATION AUDIT")
    print(f" Target Server: {BASE_URL}")
    print("=" * 80)

    # 1. /health
    print("\n--- [1] GET /health ---")
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health") as resp:
            health = json.loads(resp.read().decode("utf-8"))
            print(f"  Status: {health.get('status')}")
            print(f"  Primary Engine: {health.get('primary_engine')}")
            print(f"  ViT Loaded: {health.get('vit_loaded')}")
            print(f"  Loaded Checkpoint: {health.get('checkpoint_path')}")
            print(f"  Num Classes: {health.get('num_classes')}")
            print(f"  Classes: {health.get('classes')}")
    except Exception as e:
        print(f"  FAILED to query /health: {e}")
        return

    # 2. /classes
    print("\n--- [2] GET /classes ---")
    try:
        with urllib.request.urlopen(f"{BASE_URL}/classes") as resp:
            classes_resp = json.loads(resp.read().decode("utf-8"))
            print(f"  Count: {classes_resp.get('count')}")
            print(f"  Classes: {classes_resp.get('classes')}")
            metadata = classes_resp.get("metadata", {})
            for c in classes_resp.get("classes", []):
                meta = metadata.get(c, {})
                print(f"    - Class '{c}': name='{meta.get('name')}', region='{meta.get('region')}'")
    except Exception as e:
        print(f"  FAILED to query /classes: {e}")
        return

    # 3. Test Inscription Images
    print("\n--- [3] Inscription Predictions (/predict) ---")
    test_inscriptions = [
        ("Mangulam Tamil-Brahmi (TBSI_001)", PROJECT_ROOT / "dataset" / "whole_inscriptions" / "tamil_brahmi" / "TBSI_001_original.jpg"),
        ("Sittannavasal Tamil-Brahmi (TBSI_020)", PROJECT_ROOT / "dataset" / "whole_inscriptions" / "tamil_brahmi" / "TBSI_020_original.jpg"),
        ("Lumbini Ashokan Brahmi", PROJECT_ROOT / "dataset" / "whole_inscriptions" / "brahmi" / "Devanampiyena Piyadasi in the Lumbini Edict of Ashoka.jpg"),
        ("Sarnath Ashokan Brahmi", PROJECT_ROOT / "dataset" / "whole_inscriptions" / "brahmi" / "Brahmi pillar inscription in Sarnath.jpg"),
    ]

    for label, img_path in test_inscriptions:
        if not img_path.exists():
            print(f"  [MISSING] {label}: {img_path}")
            continue
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        status_code, resp_data = send_multipart_request(
            f"{BASE_URL}/predict?top_k=5",
            filename=img_path.name,
            file_bytes=img_bytes,
            content_type="image/jpeg"
        )
        print(f"  [{label}]")
        print(f"    HTTP Status: {status_code}")
        print(f"    Predicted Script: {resp_data.get('script')} ({resp_data.get('script_family')})")
        print(f"    Confidence: {resp_data.get('confidence'):.4f} ({resp_data.get('decision')})")
        print(f"    Status: {resp_data.get('status')}")
        print(f"    Execution Time: {resp_data.get('execution_time_ms')} ms")
        cands = resp_data.get("candidates", [])
        cands_str = ", ".join([f"{c.get('class')}: {c.get('confidence'):.4f}" for c in cands[:3]])
        print(f"    Top Candidates: {cands_str}")

    # 4. Rejection and Validation Tests
    print("\n--- [4] Rejection & Validation Tests (/predict) ---")

    # 4a. Random Photograph (synthetic natural scene)
    grad = np.linspace(40, 220, 224, dtype=np.uint8)
    photo_arr = np.stack([
        np.tile(grad, (224, 1)),
        np.tile(grad[::-1], (224, 1)),
        np.full((224, 224), 140, dtype=np.uint8)
    ], axis=-1)
    buf_photo = io.BytesIO()
    Image.fromarray(photo_arr).save(buf_photo, format="JPEG")
    status_code, resp_photo = send_multipart_request(
        f"{BASE_URL}/predict?top_k=5",
        filename="random_landscape_photo.jpg",
        file_bytes=buf_photo.getvalue(),
        content_type="image/jpeg"
    )
    print(f"  [Random Photograph]")
    print(f"    HTTP Status: {status_code}")
    print(f"    Status: {resp_photo.get('status')}")
    print(f"    Decision: {resp_photo.get('decision')}")
    print(f"    Script: {resp_photo.get('script')}")
    print(f"    Rejection Reason: {resp_photo.get('rejection_reason')}")

    # 4b. Blank White Image
    white_img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    buf_white = io.BytesIO()
    white_img.save(buf_white, format="PNG")
    status_code, resp_white = send_multipart_request(
        f"{BASE_URL}/predict?top_k=5",
        filename="blank_white.png",
        file_bytes=buf_white.getvalue(),
        content_type="image/png"
    )
    print(f"  [Blank White Image]")
    print(f"    HTTP Status: {status_code}")
    print(f"    Status: {resp_white.get('status')}")
    print(f"    Decision: {resp_white.get('decision')}")
    print(f"    Script: {resp_white.get('script')}")
    print(f"    Rejection Reason: {resp_white.get('rejection_reason')}")

    # 4c. Corrupted Image File
    corrupt_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRcorrupted_garbage_stream_not_decodable"
    status_code, resp_corrupt = send_multipart_request(
        f"{BASE_URL}/predict?top_k=5",
        filename="corrupted_inscription.png",
        file_bytes=corrupt_bytes,
        content_type="image/png"
    )
    print(f"  [Corrupted Image File]")
    print(f"    HTTP Status: {status_code}")
    print(f"    Detail: {resp_corrupt.get('detail')}")

    # 4d. Unsupported File Format
    text_bytes = b"This is a transcript text file, not an inscription raster image."
    status_code, resp_text = send_multipart_request(
        f"{BASE_URL}/predict?top_k=5",
        filename="transcription.txt",
        file_bytes=text_bytes,
        content_type="text/plain"
    )
    print(f"  [Unsupported File Format (.txt)]")
    print(f"    HTTP Status: {status_code}")
    print(f"    Detail: {resp_text.get('detail')}")

    print("\n" + "=" * 80)
    print(" LIVE AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_live_audit()
