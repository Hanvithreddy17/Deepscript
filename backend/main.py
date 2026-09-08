"""
DeepScript — Production FastAPI Inference Backend
=================================================
Serves real-time inference for Ancient Indian Script Identification using
a fine-tuned Vision Transformer (ViT-B/16) and Cosine Similarity metric head.

Endpoints:
  - GET  /health   : Health status, device info, and model metadata
  - GET  /classes  : List of supported 62 script character classes
  - POST /predict  : Upload image for script identification & top-k candidate scoring
"""

import io
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import get_deepscript_model
from preprocessing import get_eval_transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("deepscript.backend")

# Global state container for model & metadata
state: Dict[str, Any] = {
    "model": None,
    "transforms": None,
    "classes": [],
    "class_to_idx": {},
    "device": None,
    "model_metadata": {},
    "is_ready": False,
    "start_time": time.time(),
}


def load_model_and_checkpoint(
    checkpoint_path: Path = PROJECT_ROOT / "checkpoints" / "best_vit_model.pth",
) -> None:
    """Loads the trained DeepScript model checkpoint and warms up inference engine."""
    logger.info(f"Loading checkpoint from: {checkpoint_path.resolve()}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Trained model checkpoint not found at: {checkpoint_path.resolve()}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device} ({'GPU Acceleration' if device.type == 'cuda' else 'Multi-threaded CPU'})")

    # Load serialized checkpoint dictionary
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        classes = checkpoint.get("classes", [])
        class_to_idx = checkpoint.get("class_to_idx", {})
        epoch = checkpoint.get("epoch", "N/A")
        val_acc = checkpoint.get("val_acc", 0.0)
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        classes = []
        class_to_idx = {}
        epoch = "N/A"
        val_acc = 0.0
    else:
        raise ValueError(f"Unrecognized checkpoint format in {checkpoint_path}")

    num_classes = len(classes) if classes else 62
    logger.info(f"Checkpoint metadata — Classes: {num_classes}, Trained Epoch: {epoch}, Val Accuracy: {val_acc:.4f}")

    # Build model architecture
    model = get_deepscript_model(
        backbone_name="vit_b_16",
        num_classes=num_classes,
        embedding_dim=256,
        head_type="cosine",
        pretrained=False,
    )

    # Load trained weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Preprocessing transformation pipeline
    transforms = get_eval_transforms(image_size=(224, 224))

    # Perform warmup dummy inference pass
    logger.info("Executing warmup inference pass...")
    with torch.no_grad():
        dummy_input = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=device)
        _ = model(dummy_input)

    # Cache in global state
    state["model"] = model
    state["transforms"] = transforms
    state["classes"] = classes
    state["class_to_idx"] = class_to_idx
    state["device"] = device
    state["model_metadata"] = {
        "checkpoint": checkpoint_path.name,
        "backbone": "vit_b_16",
        "embedding_dim": 256,
        "head_type": "cosine",
        "num_classes": num_classes,
        "epoch": epoch,
        "val_accuracy": float(val_acc) if isinstance(val_acc, (int, float)) else val_acc,
    }
    state["is_ready"] = True
    logger.info("DeepScript Vision Transformer inference engine is READY.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to initialize model on startup and clean up on shutdown."""
    try:
        load_model_and_checkpoint()
    except Exception as e:
        logger.error(f"Failed to initialize DeepScript model: {e}", exc_info=True)
    yield
    logger.info("DeepScript backend service shutting down.")


# Instantiate FastAPI application
app = FastAPI(
    title="DeepScript Ancient Indian Script Recognition API",
    description=(
        "Production REST API for identifying ancient Indian script character classes "
        "from visual inscription images using Vision Transformers (ViT-B/16) and Cosine Similarity embeddings."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="Root index")
async def root():
    """Returns basic service descriptor."""
    return {
        "service": "DeepScript Inference API",
        "version": "1.0.0",
        "status": "online" if state["is_ready"] else "initializing",
        "endpoints": {
            "health": "/health",
            "classes": "/classes",
            "predict": "POST /predict",
            "docs": "/docs",
        },
    }


@app.get("/health", summary="Service Health & Model Status")
async def health_check():
    """Returns server health status, device information, and model metadata."""
    uptime_sec = round(time.time() - state["start_time"], 2)
    return {
        "status": "healthy" if state["is_ready"] else "loading",
        "device": str(state["device"]) if state["device"] else "none",
        "gpu_available": torch.cuda.is_available(),
        "num_classes": len(state["classes"]),
        "model_info": state["model_metadata"],
        "uptime_seconds": uptime_sec,
    }


@app.get("/classes", summary="List Supported Script Classes")
async def get_classes():
    """Returns the alphabetical list of all supported 62 script character classes."""
    if not state["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently initializing. Please retry in a few moments.",
        )
    return {
        "total_classes": len(state["classes"]),
        "classes": state["classes"],
    }


@app.post("/predict", summary="Identify Script from Inscription Image")
async def predict(
    file: UploadFile = File(..., description="Inscription image file (PNG, JPG, BMP, WebP, etc.)"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of top candidates to return"),
):
    """
    Classifies an ancient Indian inscription image into one of 62 script symbol classes.

    Returns:
      - `script`: Predicted Top-1 script class name
      - `confidence`: Softmax probability score (0.0 to 1.0)
      - `candidates`: Ranked list of top-k candidates with confidence scores
      - `execution_time_ms`: Total end-to-end inference latency in milliseconds
      - `model`: Architecture identifier
      - `source`: Inference engine source ('live')
    """
    if not state["is_ready"] or state["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is not loaded yet.",
        )

    # 1. Read & Validate Uploaded Image
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = pil_img.convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file could not be parsed as a valid image.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image reading error: {str(e)}",
        )

    # 2. Run Preprocessing & Inference
    try:
        t_start = time.perf_counter()

        # Apply deterministic evaluation transforms (Resize 224x224 + ImageNet Normalization)
        img_tensor = state["transforms"](pil_img)
        img_tensor = img_tensor.unsqueeze(0).to(state["device"])

        # Execute Model Forward Pass
        with torch.no_grad():
            output = state["model"].predict(img_tensor, top_k=top_k)

        t_end = time.perf_counter()
        latency_ms = round((t_end - t_start) * 1000, 2)

        # 3. Format Output
        pred_idx = output["predicted_class"].item()
        confidence = float(output["confidence"].item())
        classes_list = state["classes"]

        top_script_name = classes_list[pred_idx] if pred_idx < len(classes_list) else f"class_{pred_idx}"

        top_k_indices = output["top_k_indices"][0].tolist()
        top_k_probs = output["top_k_probabilities"][0].tolist()

        candidates = []
        for idx, score in zip(top_k_indices, top_k_probs):
            name = classes_list[idx] if idx < len(classes_list) else f"class_{idx}"
            candidates.append({
                "script": name,
                "score": round(float(score), 4),
            })

        return {
            "script": top_script_name,
            "confidence": round(confidence, 4),
            "candidates": candidates,
            "execution_time_ms": latency_ms,
            "model": "vit_b_16_cosine",
            "source": "live",
            "image_details": {
                "original_filename": file.filename,
                "content_type": file.content_type,
                "dimensions": f"{pil_img.width}x{pil_img.height}",
            },
        }

    except Exception as e:
        logger.error(f"Inference pipeline failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
