from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import mimetypes

# Adjust import paths depending on how the app is run
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import DeepfakeDetector
from api.utils import extract_frames_from_video

app = FastAPI(title="Deepfake Detection API")

# Add CORS middleware to allow requests from our simple frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
detector = DeepfakeDetector()

@app.post("/detect")
async def detect_deepfake(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
        
    content = await file.read()
    content_type = file.content_type
    
    if content_type is None:
        content_type, _ = mimetypes.guess_type(file.filename)
        
    if content_type and content_type.startswith('video'):
        # Video processing - Extract a sequence of frames for temporal modeling
        frames = extract_frames_from_video(content, num_frames=10)
        if not frames:
            raise HTTPException(status_code=400, detail="Could not extract frames from video")
            
        # The detector now handles multiple faces and returns a list of individual results
        result = detector.predict_with_explainability(frames)
        return result
        
    elif content_type and content_type.startswith('image'):
        # Image processing
        try:
            image = Image.open(io.BytesIO(content)).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
            
        result = detector.predict_with_explainability(image)
        return result
        
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an image or video.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
