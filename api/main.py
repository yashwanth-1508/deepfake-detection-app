from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import mimetypes

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import DeepfakeDetector
from api.utils import (
    extract_frames_from_video,
    detect_watermark_artifacts,
    analyze_lighting_consistency,
    analyze_lip_motion,
    extract_audio_energy_from_video,
)
import cv2
import numpy as np

app = FastAPI(title="TruthLense AI — Deepfake Detection API v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model singleton ──────────────────────────────────────────────────────────
detector = DeepfakeDetector()


# ─────────────────────────────────────────────────────────────────────────────
# /detect  — Main detection (image + video with multi-modal consistency)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/detect")
async def detect_deepfake(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    content = await file.read()
    content_type = file.content_type

    if content_type is None:
        content_type, _ = mimetypes.guess_type(file.filename)

    # ── VIDEO ────────────────────────────────────────────────────────────────
    if content_type and content_type.startswith("video"):
        frames = extract_frames_from_video(content, num_frames=10)
        if not frames:
            raise HTTPException(status_code=400, detail="Could not extract frames from video")

        result = detector.predict_robust(frames)

        # ── Multi-Modal Consistency Check ────────────────────────────────────
        try:
            consistency = _run_multimodal_consistency(content, frames, result)
            result["consistency"] = consistency
        except Exception as e:
            print(f"Consistency check error: {e}")
            result["consistency"] = None

        return result

    # ── IMAGE ────────────────────────────────────────────────────────────────
    elif content_type and content_type.startswith("image"):
        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        result = detector.predict_robust(image)
        result["consistency"] = None  # Single image — no temporal analysis
        return result

    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload an image or video.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# /live-detect  — Lightweight endpoint for real-time webcam frames
# (skips ITA augmentation for speed)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/live-detect")
async def live_detect(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    # Use fast predict_with_explainability (no ITA flip for speed)
    result = detector.predict_with_explainability(image)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# /watermark  — AI watermark / spectral artifact detection
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/watermark")
async def watermark_detect(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    content = await file.read()
    content_type = file.content_type
    if content_type is None:
        content_type, _ = mimetypes.guess_type(file.filename)

    if not (content_type and content_type.startswith("image")):
        raise HTTPException(status_code=400, detail="Watermark detection requires an image file.")

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    result = detect_watermark_artifacts(image)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper — Multi-Modal Consistency Analysis for videos
# ─────────────────────────────────────────────────────────────────────────────
def _run_multimodal_consistency(video_bytes: bytes, frames, detection_result: dict) -> dict:
    """
    Runs lip-motion and lighting consistency checks on a video.
    Returns a dict with lip_sync and lighting sub-results.
    """
    consistency = {}

    # ── Prepare BGR frames + face boxes from result ──────────────────────────
    frames_bgr = []
    for f in frames:
        arr = np.array(f)
        frames_bgr.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    # Collect face boxes from the first detected face track
    faces = detection_result.get("faces", [])
    primary_box = faces[0].get("box") if faces else None

    # ── Lip Motion Analysis ──────────────────────────────────────────────────
    if primary_box and len(frames_bgr) >= 2:
        # Use the same box for all frames (simplified — stable face assumed)
        boxes = [primary_box] * len(frames_bgr)
        lip_result = analyze_lip_motion(frames_bgr, boxes)
        consistency["lip_sync"] = lip_result
    else:
        consistency["lip_sync"] = {
            "lip_motion_score": None,
            "details": "No face detected for lip analysis",
        }

    # ── Lighting Consistency ─────────────────────────────────────────────────
    if primary_box and frames_bgr:
        mid_frame = frames_bgr[len(frames_bgr) // 2]
        x1, y1, x2, y2 = primary_box
        face_crop = mid_frame[max(0, y1):y2, max(0, x1):x2]
        if face_crop.size > 0:
            lighting_result = analyze_lighting_consistency(face_crop, mid_frame, primary_box)
            consistency["lighting"] = lighting_result
        else:
            consistency["lighting"] = {"lighting_score": None, "details": "Could not crop face region"}
    else:
        consistency["lighting"] = {"lighting_score": None, "details": "No face detected"}

    # ── Audio Energy ─────────────────────────────────────────────────────────
    audio_result = extract_audio_energy_from_video(video_bytes)
    consistency["audio"] = audio_result

    # ── Overall Consistency Score ────────────────────────────────────────────
    scores = []
    lip_score = consistency["lip_sync"].get("lip_motion_score")
    light_score = consistency["lighting"].get("lighting_score")
    if lip_score is not None:
        scores.append(lip_score)
    if light_score is not None:
        scores.append(light_score)

    consistency["overall_score"] = round(float(np.mean(scores)), 3) if scores else None
    consistency["overall_label"] = (
        "Consistent" if (consistency["overall_score"] or 0) >= 0.65
        else "Suspicious" if (consistency["overall_score"] or 0) >= 0.35
        else "Highly Inconsistent"
    ) if consistency["overall_score"] is not None else "Insufficient data"

    return consistency


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
