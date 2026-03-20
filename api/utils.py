import cv2
import tempfile
import os
import numpy as np
from PIL import Image

def compute_optical_flow(prev_frame, curr_frame):
    """
    Computes optical flow between two frames.
    Returns the average magnitude of the flow.
    """
    prev_gray = cv2.cvtColor(np.array(prev_frame), cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(np.array(curr_frame), cv2.COLOR_RGB2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    return np.mean(mag)

def extract_frames_from_video(video_bytes, num_frames=10):
    """
    Extracts a specified number of frames from a video.
    Returns a list of PIL Images.
    """
    frames = []
    
    # Save video bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_bytes)
        temp_video_path = temp_video.name
        
    try:
        cap = cv2.VideoCapture(temp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return frames
            
        # Extract more frames to allow for temporal modeling
        # We try to get num_frames evenly spaced
        step = max(1, total_frames // num_frames)
        
        for i in range(num_frames):
            frame_id = i * step
            if frame_id >= total_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                
        cap.release()
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            
    return frames
