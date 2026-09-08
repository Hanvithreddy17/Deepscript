# DeepScript — Production FastAPI Inference Backend

The DeepScript backend service provides high-performance RESTful API endpoints for ancient Indian script recognition. It serves the fine-tuned **Vision Transformer (ViT-B/16)** model combined with a metric embedding projector and **Cosine Similarity classification head**.

---

## 1. System Architecture

```
[ Inscription Image Upload (multipart/form-data) ]
                        │
                        ▼
[ FastAPI Request Handler & File Validation ]
                        │
                        ▼
[ Preprocessing: RGB Conversion + Resize (224×224) + ImageNet Normalization ]
                        │
                        ▼
[ Vision Transformer (ViT-B/16) Feature Extractor (768-dim) ]
                        │
                        ▼
[ Metric Projector (MLP: 768 → 512 → 256 + L2 Normalization) ]
                        │
                        ▼
[ Cosine Similarity Classification Head (256 → 62 Classes) ]
                        │
                        ▼
[ JSON Output: Top-1 Class, Top-5 Candidates, Latency & Metadata ]
```

---

## 2. API Endpoints

### `GET /health`
Returns the operational health, device status, and model metadata.

**Sample Response:**
```json
{
  "status": "healthy",
  "device": "cpu",
  "gpu_available": false,
  "num_classes": 62,
  "model_info": {
    "checkpoint": "best_vit_model.pth",
    "backbone": "vit_b_16",
    "embedding_dim": 256,
    "head_type": "cosine",
    "num_classes": 62,
    "epoch": 4,
    "val_accuracy": 0.9226
  },
  "uptime_seconds": 12.4
}
```

---

### `GET /classes`
Returns the complete list of all 62 supported ancient Indian script character classes.

**Sample Response:**
```json
{
  "total_classes": 62,
  "classes": ["a", "aa", "ah", "ai", "am", "au", "ba", "bha", "ca", "..."]
}
```

---

### `POST /predict`
Uploads an inscription character image for real-time script recognition.

- **Request Type:** `multipart/form-data`
- **Query Parameter:** `top_k` (optional integer, default = 5, range: 1–20)
- **Form Field:** `file` (binary image data)

**Sample Response:**
```json
{
  "script": "ka",
  "confidence": 0.9652,
  "candidates": [
    { "script": "ka", "score": 0.9652 },
    { "script": "kha", "score": 0.0142 },
    { "script": "ga", "score": 0.0078 },
    { "script": "ksha", "score": 0.0041 },
    { "script": "ta", "score": 0.0029 }
  ],
  "execution_time_ms": 41.8,
  "model": "vit_b_16_cosine",
  "source": "live",
  "image_details": {
    "original_filename": "sample_ka_01.png",
    "content_type": "image/png",
    "dimensions": "224x224"
  }
}
```

---

## 3. Running the Server

### Direct Execution with Python:
```bash
python -m backend.main
```

### Or with Uvicorn CLI:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Interactive API Documentation:
When the server is running, navigate to:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 4. Testing the Backend

Run the automated backend test suite:
```bash
python backend/test_backend.py
```
