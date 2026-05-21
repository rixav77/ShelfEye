"""Analyze CLIP similarity distribution between shelf crops and catalogue."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detector import detect_products
from cropper import crop_detections
from embedder import embed_catalogue, embed_crops


def main():
    cat = embed_catalogue()
    cat_embs = cat["embeddings"]
    products = cat["products"]

    all_max_sims = []
    all_results = []

    for img_path in sorted(Path("shelf_images").glob("shelf_*.jpg")):
        dets = detect_products(str(img_path))
        dets = crop_detections(str(img_path), dets)

        if not dets:
            continue

        crop_embs = embed_crops(dets)
        sims = crop_embs @ cat_embs.T

        for i, det in enumerate(dets):
            max_idx = np.argmax(sims[i])
            max_sim = sims[i][max_idx]
            all_max_sims.append(max_sim)
            all_results.append({
                "image": img_path.name,
                "crop_idx": i,
                "best_match": products[max_idx]["product_name"],
                "best_sku": products[max_idx]["sku_id"],
                "similarity": float(max_sim),
            })

    all_max_sims = np.array(all_max_sims)

    print("=" * 60)
    print(f"SIMILARITY DISTRIBUTION ({len(all_max_sims)} crops)")
    print("=" * 60)
    print(f"  Min:    {all_max_sims.min():.4f}")
    print(f"  Max:    {all_max_sims.max():.4f}")
    print(f"  Mean:   {all_max_sims.mean():.4f}")
    print(f"  Median: {np.median(all_max_sims):.4f}")
    print(f"  Std:    {all_max_sims.std():.4f}")
    print()

    for pct in [10, 25, 50, 75, 90, 95]:
        val = np.percentile(all_max_sims, pct)
        print(f"  P{pct:2d}:    {val:.4f}")

    print()
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        above = (all_max_sims >= thresh).sum()
        pct = 100 * above / len(all_max_sims)
        print(f"  >= {thresh:.2f}: {above:3d}/{len(all_max_sims)} ({pct:5.1f}%)")

    print()
    print("TOP 15 MATCHES (highest similarity):")
    sorted_results = sorted(all_results, key=lambda x: x["similarity"], reverse=True)
    for r in sorted_results[:15]:
        print(f"  {r['similarity']:.4f}  {r['image']} crop_{r['crop_idx']:03d} -> {r['best_match']}")

    print()
    print("BOTTOM 10 MATCHES (lowest similarity):")
    for r in sorted_results[-10:]:
        print(f"  {r['similarity']:.4f}  {r['image']} crop_{r['crop_idx']:03d} -> {r['best_match']}")


if __name__ == "__main__":
    main()
