"""Match detected crops to catalogue products using CLIP + Qwen2.5-VL fallback."""

import numpy as np

from embedder import embed_catalogue, embed_crops
from vlm_fallback import identify as vlm_identify

CLIP_THRESHOLD = 0.65


def match_detections(
    detections: list[dict],
    catalogue_path: str = "catalogue/catalogue.json",
    clip_threshold: float = CLIP_THRESHOLD,
    use_vlm: bool = True,
    device: str = None,
) -> list[dict]:
    """
    Match each detection to a catalogue product.

    Augments each detection dict with:
      - matched: True/False
      - product_name: matched product name (or 'unknown')
      - sku_id: matched SKU (or None)
      - similarity: cosine similarity score
      - match_method: 'clip' or 'vlm'
    """
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cat = embed_catalogue(catalogue_path)
    cat_embs = cat["embeddings"]
    products = cat["products"]
    product_names = [p["product_name"] for p in products]

    crop_embs = embed_crops(detections, device=device)
    sims = crop_embs @ cat_embs.T

    vlm_queue = []

    for i, det in enumerate(detections):
        max_idx = int(np.argmax(sims[i]))
        max_sim = float(sims[i][max_idx])

        det["similarity"] = round(max_sim, 4)

        if max_sim >= clip_threshold:
            det["matched"] = True
            det["product_name"] = products[max_idx]["product_name"]
            det["sku_id"] = products[max_idx]["sku_id"]
            det["match_method"] = "clip"
        else:
            vlm_queue.append(i)

    if vlm_queue and use_vlm:
        print(f"[matcher] Sending {len(vlm_queue)} low-confidence crops to Qwen2.5-VL...")
        for i in vlm_queue:
            det = detections[i]
            result = vlm_identify(det["crop"], product_names, device=device)

            if result in product_names:
                idx = product_names.index(result)
                det["matched"] = True
                det["product_name"] = result
                det["sku_id"] = products[idx]["sku_id"]
                det["match_method"] = "vlm"
            else:
                det["matched"] = False
                det["product_name"] = "unknown"
                det["sku_id"] = None
                det["match_method"] = "vlm"
    elif vlm_queue:
        for i in vlm_queue:
            det = detections[i]
            det["matched"] = False
            det["product_name"] = "unknown"
            det["sku_id"] = None
            det["match_method"] = "clip"

    matched = sum(1 for d in detections if d.get("matched"))
    print(f"[matcher] {matched}/{len(detections)} products matched")
    return detections


if __name__ == "__main__":
    import sys
    from detector import detect_products
    from cropper import crop_detections

    path = sys.argv[1] if len(sys.argv) > 1 else "shelf_images/shelf_01.jpg"

    dets = detect_products(path)
    dets = crop_detections(path, dets)
    dets = match_detections(dets)

    print(f"\n{'='*60}")
    print(f"Results for {path}")
    print(f"{'='*60}")
    for d in dets:
        status = "MATCH" if d["matched"] else "UNKNOWN"
        print(f"  [{status}] {d['product_name']:40s}  sim={d['similarity']:.4f}  via={d['match_method']}")
