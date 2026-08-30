"""
DeepScript — Feature Extraction Test
===================================
Pipeline Test: Preprocessed Image -> Pretrained ViT-B/16 -> Feature Embedding

Flow:
1. Load real ancient script image from the existing dataset.
2. Apply the deterministic preprocessing pipeline (Resize -> RGB -> ImageNet Norm).
3. Pass the tensor [1, 3, 224, 224] through pretrained ViT-B/16 (backbone frozen).
4. Extract the feature embedding and verify finite values.
"""

import sys
from pathlib import Path
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import AncientScriptDataset, get_eval_transforms
from models import ViTFeatureExtractor


def run_feature_extraction_test(dataset_root: str = "dataset/dataset") -> bool:
    ds_path = Path(dataset_root)
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset root directory not found at: {ds_path.resolve()}")

    # 1. Connect to existing preprocessing pipeline
    eval_transform = get_eval_transforms(image_size=(224, 224))

    # 2. Load one real image from the existing dataset
    dataset = AncientScriptDataset(root_dir=ds_path, transform=eval_transform)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; no images found.")

    # Select the first real image sample
    sample_index = 0
    sample_info = dataset.get_sample_info(sample_index)
    image_tensor, label = dataset[sample_index]

    # Add batch dimension: [3, 224, 224] -> [1, 3, 224, 224]
    input_tensor = image_tensor.unsqueeze(0)

    # 3. Instantiate pretrained ViT-B/16 with frozen backbone
    model = ViTFeatureExtractor(
        backbone_name="vit_b_16",
        pretrained=True,
        freeze_backbone=True,
    )
    model.eval()

    # 4. Extract feature embedding
    with torch.no_grad():
        embedding = model(input_tensor)

    # 5. Inspect dynamic shapes and finite status (no hardcoding)
    actual_input_shape = list(input_tensor.shape)
    actual_embedding_shape = list(embedding.shape)
    is_finite = bool(torch.isfinite(embedding).all().item())
    is_frozen = model.is_frozen

    # Format output as required
    print(f"Input: {sample_info['filename']}")
    print(f"Input shape: {actual_input_shape}")
    print(f"Model: ViT-B/16")
    print(f"Embedding shape: {actual_embedding_shape}")
    print(f"Finite values: {is_finite}")

    # Assertions for programmatic verification
    assert actual_input_shape == [1, 3, 224, 224], f"Unexpected input shape: {actual_input_shape}"
    assert len(actual_embedding_shape) == 2 and actual_embedding_shape[0] == 1, (
        f"Unexpected embedding shape: {actual_embedding_shape}"
    )
    assert is_finite, "Extracted embedding contains NaN or Inf values!"
    assert is_frozen, "Model backbone parameters are not frozen!"

    return True


if __name__ == "__main__":
    success = run_feature_extraction_test()
    sys.exit(0 if success else 1)
