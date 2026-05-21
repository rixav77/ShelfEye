"""ShelfEye — single-command entry point for the full pipeline."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from detector import detect_products
from cropper import crop_detections
from matcher import match_detections
from position import assign_positions
from compliance import check_compliance
from visualizer import draw_detections


def run(
    image_path: str,
    output_dir: str = "outputs",
    planogram_path: str = "catalogue/planogram.json",
    clip_threshold: float = 0.65,
    use_vlm: bool = True,
    device: str = None,
):
    stem = Path(image_path).stem

    # Full pipeline
    dets = detect_products(image_path, device=device)
    dets = crop_detections(image_path, dets, output_dir=f"{output_dir}/crops")
    dets = match_detections(dets, clip_threshold=clip_threshold, use_vlm=use_vlm, device=device)
    dets = assign_positions(dets)

    # Build result
    products = []
    for d in dets:
        products.append({
            "sku_id": d.get("sku_id"),
            "product_name": d.get("product_name", "unknown"),
            "confidence": round(d.get("similarity", d.get("confidence", 0)), 4),
            "bbox": [int(c) for c in d["bbox"]],
            "row": d["row"],
            "column": d["column"],
            "matched": d.get("matched", False),
            "match_method": d.get("match_method", "none"),
        })

    matched_count = sum(1 for p in products if p["matched"])
    result = {
        "image": Path(image_path).name,
        "total_products": len(products),
        "matched": matched_count,
        "unmatched": len(products) - matched_count,
        "products": products,
    }

    # Save outputs
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    result_path = f"{output_dir}/result_{stem}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    annotated_path = f"{output_dir}/annotated_{stem}.jpg"
    draw_detections(image_path, dets, annotated_path)

    # Compliance check
    if Path(planogram_path).exists():
        report = check_compliance(result, planogram_path)
        compliance_path = f"{output_dir}/compliance_{stem}.json"
        with open(compliance_path, "w") as f:
            json.dump(report, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Image:      {Path(image_path).name}")
    print(f"  Detected:   {len(products)} products")
    print(f"  Matched:    {matched_count}")
    print(f"  Unmatched:  {len(products) - matched_count}")
    if Path(planogram_path).exists():
        print(f"  Compliance: {report['compliance_score']}%")
    print(f"  Outputs:    {output_dir}/")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="ShelfEye — AI-Powered Retail Shelf Intelligence")
    parser.add_argument("--image", "-i", required=True, help="Path to shelf image (or directory)")
    parser.add_argument("--output", "-o", default="outputs", help="Output directory")
    parser.add_argument("--planogram", "-p", default="catalogue/planogram.json", help="Planogram JSON path")
    parser.add_argument("--threshold", "-t", type=float, default=0.65, help="CLIP match threshold")
    parser.add_argument("--no-vlm", action="store_true", help="Disable VLM fallback")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    image_path = Path(args.image)

    if image_path.is_dir():
        images = sorted(image_path.glob("*.jpg")) + sorted(image_path.glob("*.png"))
        print(f"Processing {len(images)} images from {image_path}/")
        for img in images:
            run(str(img), args.output, args.planogram, args.threshold, not args.no_vlm, args.device)
    else:
        run(str(image_path), args.output, args.planogram, args.threshold, not args.no_vlm, args.device)


if __name__ == "__main__":
    main()
