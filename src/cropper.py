"""Crop detected product regions from shelf images."""

import cv2
import numpy as np
from pathlib import Path


def crop_detections(
    image_path: str,
    detections: list[dict],
    output_dir: str = None,
    padding: int = 5,
) -> list[dict]:
    """
    Crop each detected bounding box from the shelf image.

    Args:
        image_path: path to the shelf image
        detections: list of dicts with 'bbox' key ([x1, y1, x2, y2])
        output_dir: if set, save crops as JPEGs
        padding: pixels to add around each box (clamped to image bounds)

    Returns:
        Same detections list, each dict augmented with:
          - 'crop': np.ndarray of the cropped region (BGR)
          - 'crop_path': str path if output_dir was set
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    image_name = Path(image_path).stem

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        crop = img[y1:y2, x1:x2]
        det["crop"] = crop

        if output_dir:
            crop_path = str(Path(output_dir) / f"{image_name}_crop_{i:03d}.jpg")
            cv2.imwrite(crop_path, crop)
            det["crop_path"] = crop_path

    print(f"[cropper] Cropped {len(detections)} regions from {image_path}")
    return detections


if __name__ == "__main__":
    import sys
    from detector import detect_products

    path = sys.argv[1] if len(sys.argv) > 1 else "shelf_images/shelf_01.jpg"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs/crops"

    dets = detect_products(path)
    dets = crop_detections(path, dets, output_dir=out_dir)
    print(f"Saved {len(dets)} crops to {out_dir}")
