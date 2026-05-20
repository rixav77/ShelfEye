"""Generate CLIP embeddings for catalogue images and detected crop regions."""

import json
import pickle
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CATALOGUE_CACHE = "catalogue/embeddings.pkl"

_model = None
_processor = None


def load_clip(device: str = None):
    global _model, _processor
    if _model is not None:
        return _model, _processor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[embedder] Loading CLIP ({CLIP_MODEL_ID}) on {device}...")
    _processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    _model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    _model.eval()
    print("[embedder] CLIP loaded.")
    return _model, _processor


def embed_images(images: list[Image.Image], device: str = None) -> np.ndarray:
    """Compute CLIP image embeddings for a list of PIL images.

    Returns (N, 512) normalized embedding matrix.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, processor = load_clip(device)
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        out = model.get_image_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out

    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


def embed_catalogue(
    catalogue_path: str = "catalogue/catalogue.json",
    cache_path: str = CATALOGUE_CACHE,
    force: bool = False,
    device: str = None,
) -> dict:
    """Embed all catalogue reference images.

    Returns dict with:
      - 'embeddings': (N, 512) numpy array
      - 'sku_ids': list of sku_id strings (same order as rows)
      - 'products': list of product dicts from catalogue.json
    """
    cache = Path(cache_path)
    if cache.exists() and not force:
        with open(cache, "rb") as f:
            data = pickle.load(f)
        print(f"[embedder] Loaded cached catalogue embeddings ({len(data['sku_ids'])} products)")
        return data

    with open(catalogue_path) as f:
        products = json.load(f)

    images = []
    sku_ids = []
    valid_products = []

    for p in products:
        img_path = Path(p["image_path"])
        if not img_path.exists():
            print(f"[embedder] WARNING: missing image for {p['sku_id']}, skipping")
            continue
        images.append(Image.open(img_path).convert("RGB"))
        sku_ids.append(p["sku_id"])
        valid_products.append(p)

    embeddings = embed_images(images, device=device)

    data = {
        "embeddings": embeddings,
        "sku_ids": sku_ids,
        "products": valid_products,
    }

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(data, f)

    print(f"[embedder] Embedded {len(sku_ids)} catalogue products -> {cache}")
    return data


def embed_crops(detections: list[dict], device: str = None) -> np.ndarray:
    """Compute CLIP embeddings for cropped detection regions.

    Each detection must have a 'crop' key (BGR numpy array from cropper).
    Returns (N, 512) normalized embedding matrix.
    """
    images = []
    for det in detections:
        bgr = det["crop"]
        rgb = bgr[:, :, ::-1]
        images.append(Image.fromarray(rgb))

    return embed_images(images, device=device)


if __name__ == "__main__":
    cat_data = embed_catalogue(force=True)
    print(f"\nCatalogue: {cat_data['embeddings'].shape}")
    for sku, emb in zip(cat_data["sku_ids"], cat_data["embeddings"]):
        print(f"  {sku}: norm={np.linalg.norm(emb):.4f}")
