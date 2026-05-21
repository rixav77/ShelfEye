"""Visualize detection results on shelf images."""

import cv2
import numpy as np
from pathlib import Path


def draw_detections(
    image_path: str,
    detections: list[dict],
    output_path: str = None,
    show_position: bool = True,
) -> np.ndarray:
    """Draw bounding boxes, labels, and row/column info on image."""
    img = cv2.imread(image_path)

    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]

        matched = det.get("matched", None)
        if matched is True:
            color = (0, 200, 0)
        elif matched is False:
            color = (0, 0, 200)
        else:
            color = (255, 165, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Build label text
        parts = []
        if "product_name" in det:
            name = det["product_name"]
            if len(name) > 22:
                name = name[:20] + ".."
            parts.append(name)

        conf = det.get("similarity", det.get("confidence", 0))
        parts.append(f"{conf:.2f}")

        if det.get("match_method") == "vlm":
            parts.append("VLM")

        text = " | ".join(parts)

        font_scale = 0.4
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness)

        # Row/column tag in bottom-right corner
        if show_position and "row" in det and "column" in det:
            pos_text = f"R{det['row']}C{det['column']}"
            (pw, ph), _ = cv2.getTextSize(pos_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(img, (x2 - pw - 4, y2 - ph - 4), (x2, y2), (50, 50, 50), -1)
            cv2.putText(img, pos_text, (x2 - pw - 2, y2 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (255, 255, 255), 1)

    # Draw summary legend
    h, w = img.shape[:2]
    matched_count = sum(1 for d in detections if d.get("matched"))
    unknown_count = len(detections) - matched_count
    legend = f"Total: {len(detections)} | Matched: {matched_count} | Unknown: {unknown_count}"

    font_scale_lg = 0.55
    (lw, lh), _ = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, font_scale_lg, 1)
    cv2.rectangle(img, (5, h - lh - 12), (lw + 15, h - 2), (0, 0, 0), -1)
    cv2.putText(img, legend, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_lg, (255, 255, 255), 1)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"[visualizer] Saved annotated image to {output_path}")

    return img


if __name__ == "__main__":
    import sys
    from detector import detect_products
    from cropper import crop_detections
    from matcher import match_detections
    from position import assign_positions

    path = sys.argv[1] if len(sys.argv) > 1 else "shelf_images/shelf_01.jpg"
    out = sys.argv[2] if len(sys.argv) > 2 else None

    dets = detect_products(path)
    dets = crop_detections(path, dets)
    dets = match_detections(dets)
    dets = assign_positions(dets)

    if out is None:
        out = f"outputs/annotated_{Path(path).stem}.jpg"

    draw_detections(path, dets, out)
    print(f"Detections: {len(dets)}")
