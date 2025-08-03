import modal
import io
import os
import torch
import numpy as np
from PIL import Image
import uuid
import base64
import urllib.request

# --- Modal App and Image Setup ---
app = modal.App("sam-image-segmentation")

sam_image = modal.Image.debian_slim().pip_install(
    "torch",
    "torchvision",
    "segment-anything",
    "Pillow",
    "numpy"
)


# --- Model Loading and Generation Function ---
@app.function(
    image=sam_image,
    gpu="T4",
    scaledown_window=300,
    volumes={"/model_cache": modal.Volume.from_name("sam-model-cache", create_if_missing=True)}
)
def segment_image(image_bytes: bytes, image_id: str):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    # --- 1. Download the Model Checkpoint ---
    MODEL_TYPE = "vit_b"
    CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    CHECKPOINT_FILENAME = os.path.basename(CHECKPOINT_URL)
    MODEL_CACHE_PATH = f"/model_cache/{CHECKPOINT_FILENAME}"

    if not os.path.exists(MODEL_CACHE_PATH):
        print(f"Downloading {CHECKPOINT_FILENAME} to persistent volume...")
        urllib.request.urlretrieve(CHECKPOINT_URL, MODEL_CACHE_PATH)
        print("Download complete.")
    else:
        print("Model checkpoint already exists in cache.")

    # --- 2. Load the Model ---
    print(f"Loading SAM model ({MODEL_TYPE})...")
    device = "cuda"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_CACHE_PATH)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)  # This creates the generator
    print("Model loaded successfully.")

    # --- 3. Load the Image ---
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(img)

    # --- 4. Generate Masks ---
    print("Generating masks...")
    raw_masks = mask_generator.generate(image_np)
    print(f"Generated {len(raw_masks)} masks.")

    # --- 5. Format Masks to Match Zod Schema ---
    output_masks = []
    for mask_data in raw_masks:
        binary_mask = mask_data['segmentation']
        mask_image = Image.fromarray(binary_mask)
        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        base64_encoded_mask = base64.b64encode(buffer.getvalue()).decode('utf-8')
        bbox = mask_data['bbox']

        formatted_mask = {
            "id": str(uuid.uuid4()),
            "imageId": image_id,
            "maskData": base64_encoded_mask,
            "boundingBox": {
                "x": int(bbox[0]),
                "y": int(bbox[1]),
                "width": int(bbox[2]),
                "height": int(bbox[3]),
            },
            "confidence": float(mask_data.get('predicted_iou', 0.0))
        }
        output_masks.append(formatted_mask)

    return output_masks


# --- Local Testing Block ---
@app.local_entrypoint()
def main(image_path: str):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    print(f"Sending '{image_path}' to Modal for processing...")
    # Pass a string version of the UUID for testing
    masks = segment_image.remote(image_bytes, str(uuid.uuid4()))

    print("\n✅ Received response from Modal:")
    if masks:
        print(f"Found {len(masks)} masks. Here are the first 5:")
        for i, mask in enumerate(masks[:5]):
            # Updated to print the new data format
            print(f"  Mask {i + 1}: BBox={mask['boundingBox']}, Confidence={mask['confidence']:.4f}")
    else:
        print("No masks were returned.")