import modal
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# This is the main web application
app = FastAPI()

# --- CORS Middleware ---
# This is CRITICAL for your Vercel frontend to be able to call your backend.
# It allows requests from any origin. For production, you might restrict this
# to your Vercel app's domain.
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
    and returns the result.
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
        print("Calling Modal function to generate masks...")
        result = segment_image_function.remote(image_bytes)
        print("Received result from Modal.")
        return {"masks": result}
    except Exception as e:
        # Handle potential errors from the Modal call
        print(f"Error calling Modal function: {e}")
        raise HTTPException(status_code=503, detail="The mask generation service is currently unavailable.")

@app.get("/")
def read_root():
    return {"message": "SAM FastAPI Backend is running."}