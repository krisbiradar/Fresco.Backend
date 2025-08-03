import modal
import io
import os
import torch
import numpy as np
from PIL import Image

# --- Modal App and Image Setup ---
# This part defines the software environment.
app = modal.App("sam-image-segmentation")

sam_image = modal.Image.debian_slim().pip_install(
    "torch",
    "torchvision",
    "segment-anything", # The official packagepi
    "Pillow",           # For image processing
    "numpy"             # For array manipulation
)

# --- Model Loading and Generation Function ---
# This code runs on Modal's GPU servers.
@app.function(
    image=sam_image,
    gpu="T4",
    scaledown_window=300,
    # Mount a persistent volume to store the model checkpoint
    # so we don't have to download it every single time.
    volumes={"/model_cache": modal.Volume.from_name("sam-model-cache", create_if_missing=True)}
)
def segment_image(image_bytes: bytes):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    import urllib.request

    # --- 1. Download the Model Checkpoint ---
    # This is the crucial step you were missing.
    MODEL_TYPE = "vit_b"
    CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    CHECKPOINT_FILENAME = os.path.basename(CHECKPOINT_URL)
    MODEL_CACHE_PATH = f"/model_cache/{CHECKPOINT_FILENAME}"

    # Download the model weights only if they don't already exist in our volume.
    if not os.path.exists(MODEL_CACHE_PATH):
        print(f"Downloading {CHECKPOINT_FILENAME} to persistent volume...")
        urllib.request.urlretrieve(CHECKPOINT_URL, MODEL_CACHE_PATH)
        print("Download complete.")
    else:
        print("Model checkpoint already exists in cache.")

    # --- 2. Load the Model ---
    print(f"Loading SAM model ({MODEL_TYPE}) from {MODEL_CACHE_PATH}...")
    # Use "cuda" to run on the GPU
    device = "cuda"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_CACHE_PATH)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    print("Model loaded successfully.")

    # --- 3. Load the Image ---
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(img)

    # --- 4. Generate Masks ---
    print("Generating masks...")
    masks = mask_generator.generate(image_np)
    print(f"Generated {len(masks)} masks.")

    # The mask data is very large. We need to simplify it before sending back.
    # We will keep the bounding box and the area.
    output_masks = [
        {
            # Get the bbox from the 'mask' dictionary
            "bbox": [int(mask['bbox'][0]), int(mask['bbox'][1]), int(mask['bbox'][2]), int(mask['bbox'][3])],
            "area": int(mask["area"]),
        }
        for mask in masks
    ]

    return output_masks

# --- Local Testing Block ---
# This part runs on your computer.
@app.local_entrypoint()
def main(image_path: str):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    print(f"Sending '{image_path}' to Modal for processing...")
    masks = segment_image.remote(image_bytes)
    print("\n✅ Received response from Modal:")
    print(f"Found {len(masks)} masks. Here are the first 5:")
    for i, mask in enumerate(masks[:5]):
        print(f"  Mask {i+1}: Bounding Box={mask['bbox']}, Area={mask['area']}")