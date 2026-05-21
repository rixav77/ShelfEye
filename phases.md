# ShelfEye — Build Phases

## Phase 1: Data Collection & Catalogue Building
**Time: Day 1 morning (~2-3 hrs)**

### 1A. Select Category
- Pick ONE category: biscuits, chips, chocolates, or soft drinks
- Indian retail focus (Parle, Britannia, ITC, Nestle, etc.)

### 1B. Collect Shelf Images
- 3-5 realistic Indian retail shelf images (cluttered, varied lighting)
- Sources: Google Images, open retail datasets, personal store photos

### 1C. Build Product Catalogue (15-20 products)
- Scrape product images + metadata from Blinkit / JioMart / BigBasket
- Store clean reference images per product
- Each entry:
  ```json
  {
    "sku_id": "parle_g_50g",
    "product_name": "Parle-G Gold 50g",
    "brand": "Parle",
    "category": "biscuits",
    "image_path": "catalogue/parle_g_gold.jpg"
  }
  ```

### Deliverables
- `catalogue/` folder with product images
- `catalogue.json` with structured metadata
- `shelf_images/` folder with test images

---

## Phase 2: Object Detection Pipeline
**Time: Day 1 afternoon (~2-3 hrs)**

### Approach: Grounding DINO (zero-shot detection)
- Prompt: "product on shelf", "packaged food item"
- No training needed, works out of the box on retail shelves
- Fallback: YOLOv8 with SKU-110K pretrained weights if Grounding DINO has issues

### Pipeline
```
Shelf Image → Grounding DINO → List of bounding boxes + confidence scores
```

### Output per detection
```python
{"bbox": [x1, y1, x2, y2], "confidence": 0.93}
```

### Deliverables
- `src/detector.py` — detection module
- Visualized output: annotated shelf image with bounding boxes

---

## Phase 3: Product Cropping + Embedding Generation
**Time: Day 1 evening (~2 hrs)**

### 3A. Crop detected regions
- Extract each bounding box as individual product image
- Save to `outputs/crops/`

### 3B. Generate CLIP embeddings
- Embed all catalogue images (done once, cached)
- Embed each detected crop

### Deliverables
- `src/embedder.py` — CLIP embedding module
- Cached catalogue embeddings (numpy/pickle)

---

## Phase 4: Similarity Matching + Recognition
**Time: Day 2 morning (~2 hrs)**

### Pipeline
```
Crop embedding → Cosine similarity against catalogue embeddings → Best match
```

### Confidence thresholding
- High confidence (>0.75): accept match
- Low confidence (<0.75): flag as "unknown"

### VLM Fallback (Qwen2.5-VL local)
- For unknown/low-confidence crops, send to Qwen2.5-VL-3B-Instruct (local, no API key needed)
- Ask: "Identify this product from the catalogue list"
- Shows hybrid approach: cheap CLIP first, VLM only when needed

### Deliverables
- `src/matcher.py` — matching module (CLIP similarity + VLM dispatch)
- `src/vlm_fallback.py` — Qwen2.5-VL fallback module

---

## Phase 5: Position Understanding + Structured Output
**Time: Day 2 morning-afternoon (~2 hrs)**

### Position logic
- Sort bboxes by y-coordinate → cluster into shelf rows (k-means or threshold-based)
- Sort within each row by x-coordinate → column position

### Final JSON output
```json
{
  "image": "shelf_01.jpg",
  "timestamp": "2026-05-21T10:30:00",
  "total_products": 12,
  "matched": 10,
  "unmatched": 2,
  "products": [
    {
      "sku_id": "parle_g_50g",
      "product_name": "Parle-G Gold",
      "confidence": 0.94,
      "bbox": [120, 50, 240, 310],
      "row": 1,
      "column": 2,
      "matched": true
    }
  ]
}
```

### Deliverables
- `src/position.py` — row/column assignment
- `outputs/results.json` — structured pipeline output

---

## Phase 6: Visualization
**Time: Day 2 afternoon (~1-2 hrs)**

### Annotated output image
- Bounding boxes color-coded: green (matched), red (unknown)
- Labels: product name + confidence
- Row/column grid overlay

### Deliverables
- `src/visualizer.py`
- `outputs/annotated_shelf.jpg`

---

## Phase 7: Compliance Engine (Simple)
**Time: Day 2 evening (~1-2 hrs)**

### Logic
- Define expected planogram (simple JSON)
- Compare detected shelf state against planogram
- Flag violations: out-of-stock, wrong position, unknown product

### Output
```json
{
  "compliance_score": 0.83,
  "violations": [
    {"type": "out_of_stock", "expected_sku": "oreo_vanilla", "position": "row_1_col_3"},
    {"type": "wrong_position", "sku": "kitkat", "expected": "row_2_col_1", "actual": "row_1_col_4"}
  ]
}
```

### Deliverables
- `src/compliance.py`
- Sample planogram JSON

---

## Phase 8: Entry Point + README + Packaging
**Time: Day 3 (~3-4 hrs)**

### Single-command entry point
```bash
python run_pipeline.py --image shelf_images/shelf_01.jpg --output outputs/
```
Runs full pipeline: detect → crop → embed → match → position → output JSON + annotated image

### README sections
1. Problem statement
2. Architecture diagram (ASCII or image)
3. Tech stack + why each choice
4. Setup & usage instructions
5. Sample results (images + JSON)
6. Tradeoffs & limitations
7. Future improvements

### Deliverables
- `run_pipeline.py`
- `README.md`
- `requirements.txt`

---

## Final Project Structure
```
ShelfEye/
├── shelf_images/           # Input shelf photos
├── catalogue/              # Reference product images
│   └── catalogue.json      # Product metadata
├── outputs/                # Pipeline outputs
│   ├── crops/              # Detected product crops
│   ├── results.json        # Structured output
│   └── annotated_shelf.jpg # Visualized detections
├── src/
│   ├── detector.py         # Grounding DINO detection
│   ├── embedder.py         # CLIP embedding generation
│   ├── matcher.py          # Cosine similarity matching
│   ├── vlm_fallback.py     # GPT-4o Vision fallback
│   ├── position.py         # Row/column assignment
│   ├── compliance.py       # Planogram compliance check
│   ├── visualizer.py       # Annotated image output
│   └── utils.py            # Shared utilities
├── planogram.json          # Expected shelf layout
├── run_pipeline.py         # Single entry point
├── requirements.txt
└── README.md
```

---

## Tech Stack Summary

| Component | Tool | Why |
|-----------|------|-----|
| Detection | Grounding DINO | Zero-shot, no training, works on any shelf |
| Embeddings | CLIP (ViT-B/32) | Strong visual similarity, no fine-tuning needed |
| Matching | Cosine similarity | Simple, effective at catalogue scale (20 items) |
| Fallback recognition | GPT-4o Vision | Handles unknowns, shows hybrid cost thinking |
| Image processing | OpenCV + PIL | Industry standard |
| Data | Pandas + JSON | Clean structured outputs |
| Visualization | OpenCV + Matplotlib | Annotated detection images |

---

## Cut List (Not building, but can discuss)
- FAISS (unnecessary at 20-product scale, mention as scaling solution)
- Analytics dashboard (mention as future work)
- Multi-image robustness testing (mention limitations instead)
- Video pipeline (mention as Module 1 extension)
- Fine-tuned SKU classifier (mention as accuracy improvement path)
