"""Visualize detection results on shelf images."""

import cv2
import numpy as np
from pathlib import Path


def draw_detections(
    image_path: str,
    detections: list[dict],
    output_path: str = None,
) -> np.ndarray:
    """Draw bounding boxes and labels on image."""
    img = cv2.imread(image_path)

    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
        conf = det["confidence"]
        label = det.get("label", "product")

        matched = det.get("matched", None)
        if matched is True:
            color = (0, 200, 0)  # green
        elif matched is False:
            color = (0, 0, 200)  # red
        else:
            color = (255, 165, 0)  # orange (unknown match status)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = f"{conf:.2f}"
        if "product_name" in det:
            name = det["product_name"]
            if len(name) > 20:
                name = name[:18] + ".."
            text = f"{name} {conf:.2f}"

        font_scale = 0.45
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"[visualizer] Saved annotated image to {output_path}")

    return img


if __name__ == "__main__":
    import sys
    import json

    from detector import detect_products

    path = sys.argv[1] if len(sys.argv) > 1 else "shelf_images/shelf_01.jpg"
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs/detection_result.jpg"

    dets = detect_products(path)
    draw_detections(path, dets, out)
    print(f"Detections: {len(dets)}")
