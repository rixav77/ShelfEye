"""Full ShelfEye pipeline: detect → crop → embed → match → position → JSON output."""

import json
from datetime import datetime
from pathlib import Path

from detector import detect_products
from cropper import crop_detections
from matcher import match_detections
from position import assign_positions
from visualizer import draw_detections


def run_pipeline(
    image_path: str,
    output_dir: str = "outputs",
    use_vlm: bool = True,
    device: str = None,
) -> dict:
    """Run the full pipeline on a single shelf image.

    Returns structured JSON result.
    """
    dets = detect_products(image_path, device=device)
    dets = crop_detections(image_path, dets)
    dets = match_detections(dets, use_vlm=use_vlm, device=device)
    dets = assign_positions(dets)

    products = []
    for d in dets:
        products.append({
            "sku_id": d.get("sku_id"),
            "product_name": d.get("product_name", "unknown"),
            "confidence": d.get("similarity", d.get("confidence", 0)),
            "bbox": [int(c) for c in d["bbox"]],
            "row": d["row"],
            "column": d["column"],
            "matched": d.get("matched", False),
            "match_method": d.get("match_method", "none"),
        })

    matched_count = sum(1 for p in products if p["matched"])

    result = {
        "image": Path(image_path).name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_products": len(products),
        "matched": matched_count,
        "unmatched": len(products) - matched_count,
        "products": products,
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"result_{Path(image_path).stem}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    img_out = str(Path(output_dir) / f"annotated_{Path(image_path).stem}.jpg")
    draw_detections(image_path, dets, img_out)

    print(f"[pipeline] Saved result to {out_path}")
    return result


if __name__ == "__main__":
    import sys

    images = sys.argv[1:] or ["shelf_images/shelf_01.jpg"]

    for img in images:
        result = run_pipeline(img)
        print(f"\n{result['image']}: {result['matched']}/{result['total_products']} matched, "
              f"{result['unmatched']} unknown")
