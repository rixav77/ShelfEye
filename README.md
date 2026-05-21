# ShelfEye

Detect and identify products on retail shelf images, match them against a product catalogue, and check planogram compliance.

## How it works

```
Shelf Image → Grounding DINO (detect boxes) → CLIP (match to catalogue) → Qwen2.5-VL (fallback for low-confidence) → Compliance Check
```

- **Detection**: Grounding DINO (zero-shot, text-prompted) finds product bounding boxes
- **Matching**: CLIP embeddings + cosine similarity against 20-product catalogue (Indian biscuits)
- **Fallback**: Crops below 0.65 similarity go to Qwen2.5-VL-3B for identification
- **Compliance**: Compares detected shelf layout against expected planogram, flags out-of-stock / misplaced / unexpected products

## Setup

```bash
conda create -n shelfeye python=3.11 -y
conda activate shelfeye
pip install -r requirements.txt
```

## Usage

Single image:
```bash
python run_pipeline.py --image shelf_images/shelf_01.jpg
```

All images in a directory:
```bash
python run_pipeline.py --image shelf_images/
```

Options:
```
--output, -o       Output directory (default: outputs/)
--planogram, -p    Planogram JSON path (default: catalogue/planogram.json)
--threshold, -t    CLIP match threshold (default: 0.65)
--no-vlm           Disable VLM fallback, use CLIP-only
--device           Force cuda or cpu
```

Gradio demo:
```bash
python app.py
```

## Output

For each image, the pipeline generates:
- `result_<name>.json` — structured detection + matching results
- `annotated_<name>.jpg` — image with bounding boxes, product names, row/column tags
- `compliance_<name>.json` — planogram compliance report with violation details

## Project structure

```
ShelfEye/
├── shelf_images/           # Input shelf photos
├── catalogue/
│   ├── catalogue.json      # 20-product metadata
│   ├── images/             # Reference product images
│   ├── planogram.json      # Expected shelf layout
│   └── embeddings.pkl      # Cached CLIP embeddings
├── src/
│   ├── detector.py         # Grounding DINO detection
│   ├── cropper.py          # Crop detected regions
│   ├── embedder.py         # CLIP embedding generation
│   ├── matcher.py          # Cosine similarity matching
│   ├── vlm_fallback.py     # Qwen2.5-VL fallback
│   ├── position.py         # Row/column assignment
│   ├── compliance.py       # Planogram compliance check
│   └── visualizer.py       # Annotated image output
├── run_pipeline.py         # CLI entry point
├── app.py                  # Gradio demo
└── requirements.txt
```

## Future work

- Use OCR on product packaging text as an additional matching signal alongside CLIP
- Move to FAISS for catalogue lookup when scaling beyond 100+ products
- Fine-tune a product classifier on real retail data for higher matching accuracy
- Add video support for continuous shelf monitoring
- Build a dashboard to track compliance trends over time
