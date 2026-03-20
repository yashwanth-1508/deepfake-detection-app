# Deepfake Detection Project

A simple beginner-friendly project to detect deepfakes in images and videos using PyTorch and FastAPI.

## Project Structure
- `model/`: Contains the PyTorch CNN model (Xception) and training scripts.
- `api/`: Contains the FastAPI server and OpenCV video processing utilities.
- `frontend/`: Contains a simple HTML/JS interface to upload files and see predictions.

## Prerequisites
- Python 3.8+
- pip (Python package manager)

## Setup Instructions

1. **Clone or navigate to the project repository.**
   ```bash
   cd deepfake-project
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Start the FastAPI Backend Server:**
   Run the following command from the root of the project:
   ```bash
   uvicorn api.main:app --reload
   ```
   The API will be available at `http://localhost:8000`.
   You can explore the automatic interactive API documentation at `http://localhost:8000/docs`.

2. **Start the Frontend:**
   Since it's a simple HTML file, you can just open `frontend/index.html` in your web browser. Alternatively, serve it using a simple HTTP server:
   ```bash
   cd frontend
   python -m http.server 8080
   ```
   Then navigate to `http://localhost:8080` in your web browser.

## Training the Model

For detailed instructions on how to train the model with your own dataset, please refer to the [Training Guide](training_guide.md).

You can also run a quick verification of the training infrastructure with synthetic data:
```bash
python model/train.py
```

## Features implemented
- Accepts both images and videos.
- Extracts frames from videos using OpenCV.
- Utilizes a pre-trained Xception model modified for binary classification (Real vs Deepfake).
- Features an LSTM head for sequence-based video processing.
- Provides a simple web interface for uploading files and displaying the prediction and confidence score.
