import cv2
import numpy as np
import os

def create_synthetic_multiface_video():
    real_v = "kaggle_data/video/1.mp4"
    fake_v = "kaggle_data/deepfake/1.mp4"
    out_v = "multiface_test.mp4"
    
    if not os.path.exists(real_v) or not os.path.exists(fake_v):
        print("Error: Source videos not found.")
        return

    cap_real = cv2.VideoCapture(real_v)
    cap_fake = cv2.VideoCapture(fake_v)
    
    # Get properties from real video
    width = int(cap_real.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_real.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_real.get(cv2.CAP_PROP_FPS)
    
    # Output will be double width
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_v, fourcc, fps, (width * 2, height))
    
    print(f"Creating synthetic video: {out_v}...")
    
    frame_count = 0
    while frame_count < 60: # Just 2 seconds for testing
        ret_r, frame_r = cap_real.read()
        ret_f, frame_f = cap_fake.read()
        
        if not ret_r or not ret_f:
            break
            
        # Resize fake to match real height if needed
        if frame_f.shape[0] != height:
            frame_f = cv2.resize(frame_f, (int(frame_f.shape[1] * height / frame_f.shape[0]), height))
        
        # Ensure widths are consistent for side-by-side
        frame_f_resized = cv2.resize(frame_f, (width, height))
        
        # Combine side by side
        combined = np.hstack((frame_r, frame_f_resized))
        out.write(combined)
        frame_count += 1
        
    cap_real.release()
    cap_fake.release()
    out.release()
    print(f"SUCCESS: Created {out_v} (approx {frame_count} frames)")

if __name__ == "__main__":
    create_synthetic_multiface_video()
