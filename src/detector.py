"""
Product detection using Grounding DINO (zero-shot, text-prompted).
Detects bounding boxes of products on retail shelf images.
"""

import torch
from torchvision.ops import nms
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_ID = "IDEA-Research/grounding-dino-tiny"

_model = None
_processor = None


def load_model(device: str = None):
    global _model, _processor
    if _model is not None:
        return _model, _processor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[detector] Loading Grounding DINO ({MODEL_ID}) on {device}...")
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
    _model.eval()
    print("[detector] Model loaded.")
    return _model, _processor


def _filter_boxes(detections: list[dict], image_w: int, image_h: int) -> list[dict]:
    """Remove boxes that are too large (shelf-level) or too small (noise)."""
    image_area = image_w * image_h
    filtered = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        box_w = x2 - x1
        box_h = y2 - y1
        box_area = box_w * box_h
        area_ratio = box_area / image_area

        # Skip boxes covering >10% of image (shelf-level detections)
        if area_ratio > 0.10:
            continue
        # Skip tiny boxes <0.5% of image (noise)
        if area_ratio < 0.005:
            continue
        # Skip extreme aspect ratios (>5:1)
        aspect = max(box_w, box_h) / (min(box_w, box_h) + 1e-6)
        if aspect > 5:
            continue

        filtered.append(d)
    return filtered


def _apply_nms(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Non-max suppression to remove overlapping boxes."""
    if not detections:
        return detections

    boxes = torch.tensor([d["bbox"] for d in detections])
    scores = torch.tensor([d["confidence"] for d in detections])
    keep = nms(boxes, scores, iou_threshold)
    return [detections[i] for i in keep.tolist()]


def detect_products(
    image_path: str,
    text_prompt: str = "biscuit packet . product package . snack pack .",
    box_threshold: float = 0.15,
    text_threshold: float = 0.15,
    nms_threshold: float = 0.5,
    device: str = None,
) -> list[dict]:
    """
    Detect products in a shelf image.

    Returns list of dicts with keys: bbox, confidence, label
    bbox is [x1, y1, x2, y2] in pixel coordinates.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, processor = load_model(device)
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs["input_ids"],
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(h, w)],
    )[0]

    detections = []
    for box, score, label in zip(
        results["boxes"].cpu().tolist(),
        results["scores"].cpu().tolist(),
        results["labels"],
    ):
        detections.append({
            "bbox": [round(c, 1) for c in box],
            "confidence": round(score, 4),
            "label": label,
        })

    detections = _filter_boxes(detections, w, h)
    detections = _apply_nms(detections, nms_threshold)
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    print(f"[detector] Found {len(detections)} products in {image_path}")
    return detections


if __name__ == "__main__":
    import sys
    import json

    path = sys.argv[1] if len(sys.argv) > 1 else "shelf_images/shelf_01.jpg"
    dets = detect_products(path)
    print(json.dumps(dets, indent=2))
    print(f"\nTotal: {len(dets)} detections")
