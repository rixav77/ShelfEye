"""VLM fallback using Qwen2.5-VL for low-confidence product identification."""

import numpy as np
from PIL import Image

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

_model = None
_processor = None


def load_vlm(device: str = "cuda"):
    global _model, _processor
    if _model is not None:
        return _model, _processor

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import torch

    print(f"[vlm] Loading Qwen2.5-VL ({QWEN_MODEL_ID}) on {device}...")
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device,
    )
    _processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    _model.eval()
    print("[vlm] Qwen2.5-VL loaded.")
    return _model, _processor


def identify(crop_bgr: np.ndarray, product_names: list[str], device: str = "cuda") -> str:
    """Use Qwen2.5-VL to identify a product crop from a list of known products.

    Returns the matched product name or 'unknown'.
    """
    import torch
    from qwen_vl_utils import process_vision_info

    model, processor = load_vlm(device)

    rgb = crop_bgr[:, :, ::-1]
    pil_img = Image.fromarray(rgb)

    product_list = "\n".join(f"- {name}" for name in product_names)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": (
                    "This is a product on a retail shelf. Identify which product it is "
                    "from this list:\n"
                    f"{product_list}\n\n"
                    "If it matches one, reply with ONLY the exact product name from the list. "
                    "If it does not match any, reply with ONLY the word 'unknown'."
                )},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=50)

    input_len = inputs.input_ids.shape[1]
    response = processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0].strip()

    for name in product_names:
        if name.lower() in response.lower():
            return name

    return response


if __name__ == "__main__":
    import cv2
    import sys

    img_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/crops/shelf_03_crop_000.jpg"
    crop = cv2.imread(img_path)

    product_names = [
        "Parle-G Gold Biscuits", "Britannia Good Day Butter Cookies",
        "Oreo Chocolate Creme Biscuits", "Sunfeast Dark Fantasy Choco Fills",
    ]

    result = identify(crop, product_names)
    print(f"VLM result: {result}")
