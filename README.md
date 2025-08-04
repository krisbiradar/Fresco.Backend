# 🎨 Fresco - AI Image Painter

Fresco is a full-stack web application that allows users to upload an image of a building, generate segmentation masks using the Segment Anything Model (SAM), and interactively paint and download the final creation.

---
## Live Links

* **Frontend (Vercel):** `https://fresco-frontend-ashy.vercel.app/`
* **Backend API (Render):** `https://fresco-backend.onrender.com/`

---
## Tech Stack & Architecture

This project utilizes a modern, decoupled architecture to separate the user interface, web logic, and heavy computation.

---
## Running Locally

To run this project on your local machine, you will need to start both the backend and frontend services.

### Backend Setup

1.  **Navigate to the backend directory:**
    ```bash
    cd Fresco.Backend
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Modal:**
    Ensure you have authenticated with Modal.
    ```bash
    modal token new
    ```

5.  **Run the FastAPI server:**
    ```bash
    uvicorn main:app --reload --port 8000
    ```
    The backend will be available at `http://localhost:8000`.

### Frontend Setup

1.  **Navigate to the frontend directory:**
    ```bash
    cd Fresco.Frontend
    ```

2.  **Install Node modules:**
    ```bash
    npm install
    ```

3.  **Set Environment Variable:**
    Your frontend code needs to know where the backend is running. Create a `.env.local` file in the `Fresco.Frontend` directory and add the following line:
    ```
    VITE_API_URL=http://localhost:8000
    ```

4.  **Run the Vite development server:**
    ```bash
    npm run dev
    ```
    The frontend will be available at `http://localhost:5173` (or another port if specified).

---
## Deployment

This application is deployed across three cloud services:

1.  **GPU Worker (`sam_service.py`):** Deployed to **Modal** using the `modal deploy sam_service.py` command.
2.  **Backend API (`main.py`):** Deployed to **Render** as a Python Web Service. It requires `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` as environment variables to connect to the GPU worker.
3.  **Frontend (Vite App):** Deployed to **Vercel** as a static site. The `VITE_API_URL` environment variable on Vercel is set to the public URL of the Render backend.
