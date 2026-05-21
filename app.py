"""ShelfEye Gradio Demo — AI-Powered Retail Shelf Intelligence."""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
import numpy as np

from detector import detect_products
from cropper import crop_detections
from embedder import embed_catalogue, embed_crops
from matcher import match_detections
from position import assign_positions
from compliance import check_compliance
from visualizer import draw_detections


def analyze_shelf(image_path: str, use_vlm: bool, clip_threshold: float):
    if image_path is None:
        return None, "Upload a shelf image to begin."

    dets = detect_products(image_path)
    dets = crop_detections(image_path, dets)
    dets = match_detections(dets, clip_threshold=clip_threshold, use_vlm=use_vlm)
    dets = assign_positions(dets)

    annotated = draw_detections(image_path, dets)
    annotated_rgb = annotated[:, :, ::-1]

    # Build pipeline result for compliance
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

    # Compliance check
    planogram_path = Path("catalogue/planogram.json")
    if planogram_path.exists():
        compliance = check_compliance(result, str(planogram_path))
    else:
        compliance = {"note": "No planogram found. Skipping compliance check."}

    result_json = json.dumps(result, indent=2)
    compliance_json = json.dumps(compliance, indent=2)

    summary = (
        f"**Detected:** {len(products)} products\n\n"
        f"**Matched:** {matched_count} ({matched_count/len(products)*100:.0f}%)\n\n"
        f"**Unmatched:** {len(products) - matched_count}\n\n"
    )

    if "compliance_score" in compliance:
        summary += (
            f"**Compliance Score:** {compliance['compliance_score']}%\n\n"
            f"**Out of Stock:** {compliance['summary']['out_of_stock']}\n\n"
            f"**Wrong Position:** {compliance['summary']['wrong_position']}\n\n"
            f"**Unexpected:** {compliance['summary']['unexpected']}"
        )

    return annotated_rgb, summary, result_json, compliance_json


with open("catalogue/catalogue.json") as f:
    catalogue = json.load(f)
catalogue_info = "\n".join(f"- {p['product_name']} ({p['brand']})" for p in catalogue)

with gr.Blocks(title="ShelfEye — AI Shelf Intelligence", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# ShelfEye — AI-Powered Retail Shelf Intelligence\n"
        "Upload a retail shelf image to detect products, match them against a catalogue, "
        "and check planogram compliance.\n\n"
        "**Pipeline:** Grounding DINO (detection) → CLIP (matching) → Qwen2.5-VL (fallback) → Compliance Engine"
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="filepath", label="Upload Shelf Image")
            with gr.Accordion("Settings", open=False):
                use_vlm = gr.Checkbox(value=True, label="Enable VLM Fallback (Qwen2.5-VL)")
                clip_threshold = gr.Slider(0.5, 0.9, value=0.65, step=0.05, label="CLIP Match Threshold")
            run_btn = gr.Button("Analyze Shelf", variant="primary")

            gr.Markdown("### Catalogue (20 products)")
            gr.Markdown(catalogue_info)

        with gr.Column(scale=2):
            output_image = gr.Image(label="Annotated Result")
            summary_md = gr.Markdown(label="Summary")

    with gr.Row():
        result_json = gr.Code(label="Detection Result (JSON)", language="json")
        compliance_json = gr.Code(label="Compliance Report (JSON)", language="json")

    run_btn.click(
        fn=analyze_shelf,
        inputs=[input_image, use_vlm, clip_threshold],
        outputs=[output_image, summary_md, result_json, compliance_json],
    )

    gr.Markdown(
        "---\n"
        "Built with Grounding DINO, CLIP, Qwen2.5-VL | "
        "[GitHub](https://github.com/rixav77/ShelfEye)"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
