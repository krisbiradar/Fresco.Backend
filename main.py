import modal
import uuid
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# This is the main web application
app = FastAPI()

# --- CORS Middleware ---
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- API Endpoint ---
@app.post("/api/v1/generate-masks")
async def generate_masks(file: UploadFile = File(...)):
    """
    Receives an image file, calls the Modal GPU worker to generate masks,
    and returns the result in the specified schema format.
    """
    # Get a handle to our deployed Modal function
    try:
        segment_image_function = modal.Function.lookup("sam-image-segmentation", "segment_image")
    except modal.exception.NotFoundError:
        raise HTTPException(status_code=500, detail="Modal function not found. Please deploy it first.")

    # Read the image file bytes
    image_bytes = await file.read()

    # Call the Modal function and get the result
    try:
        # --- Start timing the process ---
        start_time = time.time()

        print("Calling Modal function to generate masks...")
        result_masks = segment_image_function.remote(image_bytes)
        print("Received result from Modal.")

        # --- End timing ---
        end_time = time.time()
        processing_time = end_time - start_time

        # --- Generate a unique ID ---
        image_id = str(uuid.uuid4())

        # --- Structure the response to match the Zod schema ---
        return {
            "imageId": image_id,
            "masks": result_masks,
            "processingTime": processing_time
        }

    except Exception as e:
        # Handle potential errors from the Modal call
        print(f"Error calling Modal function: {e}")
        raise HTTPException(status_code=503, detail="The mask generation service is currently unavailable.")


@app.get("/")
def read_root():
    return {"message": "SAM FastAPI Backend is running."}