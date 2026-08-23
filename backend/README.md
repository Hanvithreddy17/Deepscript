
# Backend Directory

## Directory Purpose

This directory will contain the FastAPI application, RESTful API endpoints, request validation routines, and inference integration logic serving the DeepScript model.

> **Status Notice:** The API backend is **NOT implemented yet**. The architecture and endpoints described below represent the planned design.

---

## Planned API Architecture

```
[ Frontend Client ]
        ↓
[ FastAPI Server ]
        ↓
[ Image Preprocessing Utility ]
        ↓
[ Trained ViT Feature Extractor ]
        ↓
[ Few-Shot Classifier ]
        ↓
[ Prediction & Score Generation ]
        ↓
[ JSON Response to Client ]
```

---

## Planned Endpoints

### `POST /predict`

- **Description:** Accepts an uploaded inscription image and returns the predicted script class along with a confidence/similarity score.
- **Request Format:** `multipart/form-data` containing an image file (e.g., JPEG, PNG).

#### Example Response Format (Planned Specification)

```json
{
  "script": "Tamil-Brahmi",
  "confidence": 0.946
}
```

> **Note:** The above endpoint specification and response format are planned for future development phases and are **not yet implemented**.
