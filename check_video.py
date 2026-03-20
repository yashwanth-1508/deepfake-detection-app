import torch
import cv2
import os
import sys
from model.model import DeepfakeDetector
from api.utils import extract_frames_from_video

def check_video(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    print(f"Analyzing video: {video_path}...")
    
    # 1. Initialize detector (will automatically load fine-tuned weights if they exist)
    detector = DeepfakeDetector()
    
    # 2. Extract frames
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
        
    frames = extract_frames_from_video(video_bytes, num_frames=10)
    
    if not frames:
        print("Error: Could not extract frames from video.")
        return
        
    # 3. Predict
    print("Running prediction...")
    results = detector.predict(frames)
    
    # 4. Show results
    print("\n--- Results ---")
    print(f"Prediction: {results['prediction']}")
    print(f"Confidence: {results['confidence']:.2%}")
    print("----------------")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Check if we have a sample video in kaggle_data/deepfake/
        sample_path = "kaggle_data/deepfake/1.mp4"
        if os.path.exists(sample_path):
             check_video(sample_path)
        else:
            print("Usage: python3 -m check_video <path_to_video>")
    else:
        check_video(sys.argv[1])
