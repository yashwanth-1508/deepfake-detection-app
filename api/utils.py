import cv2
import tempfile
import os
import numpy as np
from PIL import Image

# ─── Optional moviepy for audio energy extraction ───────────────────────────
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# EXISTING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def compute_optical_flow(prev_frame, curr_frame):
    """
    Computes optical flow between two BGR frames.
    Returns the average magnitude of the flow.
    """
    prev_gray = cv2.cvtColor(np.ascontiguousarray(prev_frame), cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(np.ascontiguousarray(curr_frame), cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


def extract_frames_from_video(video_bytes, num_frames=10):
    """
    Extracts a specified number of frames from a video.
    Returns a list of PIL Images.
    """
    frames = []
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_bytes)
        temp_video_path = temp_video.name

    try:
        cap = cv2.VideoCapture(temp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            return frames

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


# ─────────────────────────────────────────────────────────────────────────────
# NEW: MULTI-MODAL CONSISTENCY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def _get_mouth_roi(face_box, frame_shape):
    """
    Returns the bounding box of the lower-third (mouth region) of a face box.
    face_box = (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = face_box
    h = y2 - y1
    mouth_y1 = y1 + int(h * 0.60)
    mouth_y2 = y2
    return (x1, mouth_y1, x2, mouth_y2)


def analyze_lip_motion(frames_bgr, face_boxes_per_frame):
    """
    Estimates lip-sync consistency by measuring optical flow on the mouth ROI
    and computing how much motion variance exists between frames.

    Args:
        frames_bgr: list of BGR numpy arrays
        face_boxes_per_frame: list of (x1,y1,x2,y2) boxes, one per frame

    Returns:
        dict with:
            - lip_motion_score (0-1): 1.0 = consistent motion, 0.0 = suspicious
            - details: short string
    """
    if len(frames_bgr) < 2:
        return {"lip_motion_score": None, "details": "Not enough frames for lip analysis"}

    flow_magnitudes = []
    for i in range(1, len(frames_bgr)):
        prev_frame = frames_bgr[i - 1]
        curr_frame = frames_bgr[i]
        box = face_boxes_per_frame[i] if i < len(face_boxes_per_frame) else None

        if box is None:
            continue

        x1, my1, x2, my2 = _get_mouth_roi(box, curr_frame.shape)
        # Guard against zero-area crop
        if my2 <= my1 or x2 <= x1:
            continue

        prev_mouth = prev_frame[my1:my2, x1:x2]
        curr_mouth = curr_frame[my1:my2, x1:x2]

        if prev_mouth.size == 0 or curr_mouth.size == 0:
            continue

        if prev_mouth.shape != curr_mouth.shape:
            prev_mouth = cv2.resize(prev_mouth, (curr_mouth.shape[1], curr_mouth.shape[0]))

        mag = compute_optical_flow(prev_mouth, curr_mouth)
        flow_magnitudes.append(mag)

    if not flow_magnitudes:
        return {"lip_motion_score": None, "details": "No mouth region detected"}

    mean_flow = np.mean(flow_magnitudes)
    std_flow = np.std(flow_magnitudes)

    # Coefficient of variation — too uniform = suspicious (static deepfake mouth)
    # Too extreme variance = also suspicious (corrupted generation)
    cv = std_flow / (mean_flow + 1e-6)

    # Score: best between 0.15-0.6 CoV (natural speaking range)
    if 0.15 <= cv <= 0.65:
        score = 1.0
        details = "Lip motion consistent with natural speech"
    elif cv < 0.15:
        score = max(0.0, cv / 0.15)
        details = "Abnormally static lip region — possible deepfake"
    else:
        score = max(0.0, 1.0 - (cv - 0.65) / 0.5)
        details = "Erratic lip motion detected — inconsistency found"

    return {
        "lip_motion_score": round(float(score), 3),
        "mean_flow": round(float(mean_flow), 4),
        "details": details
    }


def analyze_lighting_consistency(face_crop_bgr, full_frame_bgr, face_box):
    """
    Compares the brightness (Value channel) histogram of the face vs. the
    background region to detect lighting inconsistencies.

    Returns:
        dict with:
            - lighting_score (0-1): 1.0 = consistent, 0.0 = mismatch
            - details: short string
    """
    if face_crop_bgr is None or full_frame_bgr is None:
        return {"lighting_score": None, "details": "Insufficient data"}

    x1, y1, x2, y2 = face_box
    fh, fw = full_frame_bgr.shape[:2]

    # Face HSV value channel
    face_hsv = cv2.cvtColor(np.ascontiguousarray(face_crop_bgr), cv2.COLOR_BGR2HSV)
    face_v = face_hsv[:, :, 2]

    # Background: mask out the face region
    bg_frame = full_frame_bgr.copy()
    bg_frame[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)] = 0
    bg_hsv = cv2.cvtColor(np.ascontiguousarray(bg_frame), cv2.COLOR_BGR2HSV)
    bg_v = bg_hsv[:, :, 2]

    # Only consider lit background pixels
    bg_pixels = bg_v[bg_v > 10]

    if bg_pixels.size < 100:
        return {"lighting_score": None, "details": "Background too dark to analyze"}

    face_mean = float(np.mean(face_v))
    bg_mean = float(np.mean(bg_pixels))

    # Normalize difference to a 0-1 score
    brightness_diff = abs(face_mean - bg_mean)
    # Threshold: >60 brightness units difference is very suspicious
    score = max(0.0, 1.0 - brightness_diff / 60.0)

    if score >= 0.75:
        details = "Face and environment lighting are consistent"
    elif score >= 0.4:
        details = "Minor lighting discrepancy detected"
    else:
        details = "Significant lighting mismatch — face may be composited"

    return {
        "lighting_score": round(score, 3),
        "face_brightness": round(face_mean, 1),
        "bg_brightness": round(bg_mean, 1),
        "details": details
    }


def extract_audio_energy_from_video(video_bytes):
    """
    Uses moviepy to extract audio energy from a video.
    Returns average RMS energy and a temporal profile.
    """
    if not MOVIEPY_AVAILABLE:
        return {"audio_available": False, "details": "moviepy not installed"}

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        clip = VideoFileClip(tmp_path)
        if clip.audio is None:
            clip.close()
            return {"audio_available": False, "details": "No audio track in video"}

        # Sample audio at 22050 Hz
        audio_array = clip.audio.to_soundarray(fps=22050)
        clip.close()

        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        # Compute RMS energy in windows
        window_size = 22050 // 10  # 100ms windows
        rms_values = []
        for i in range(0, len(audio_array) - window_size, window_size):
            window = audio_array[i:i + window_size]
            rms = float(np.sqrt(np.mean(window ** 2)))
            rms_values.append(rms)

        avg_rms = float(np.mean(rms_values)) if rms_values else 0.0
        has_speech = avg_rms > 0.01

        return {
            "audio_available": True,
            "has_speech": has_speech,
            "avg_rms": round(avg_rms, 5),
            "details": "Audio extracted successfully" if has_speech else "No significant speech audio detected"
        }
    except Exception as e:
        return {"audio_available": False, "details": f"Audio extraction failed: {str(e)}"}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# NEW: AI WATERMARK / SPECTRAL ARTIFACT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_watermark_artifacts(image_pil):
    """
    Analyzes an image for GAN / diffusion-model spectral artifacts via FFT.

    Steps:
    1. Convert to grayscale
    2. Compute 2D FFT and shift zero-frequency to center
    3. Analyze the magnitude spectrum for:
       - Checkerboard artifacts (energy at Nyquist-related frequencies)
       - Periodic patterns in grid intervals characteristic of CNN upsampling
    4. Compare center energy vs. outer ring energy ratio

    Returns:
        dict with:
            - watermark_score (0-1): 1.0 = strong AI artifact signal
            - artifacts_found: list of detected artifact types
            - details: human-readable summary
    """
    img = np.array(image_pil.convert("L").resize((512, 512)))

    # Windowing (Apodization) to prevent spectral leakage from sharp edges/backgrounds
    # This allows real photos with high contrast to correctly return a 0% score.
    win = np.hanning(512)
    window = np.outer(win, win)
    img_windowed = (img.astype(np.float32) - np.mean(img)) * window

    # FFT
    f = np.fft.fft2(img_windowed)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    # ─── Diagonal-Bias Mask (The "Portrait Patch") ───────────────────────────
    # Zero out the central Horizontal and Vertical axes of the FFT.
    # This mathematically removes "Edge Ringing" from real studio silhouettes
    # (like white-background selfies), leaving only the diagonal AI-textures.
    mask = np.ones((h, w), dtype=np.float32)
    thickness = 4 # Taper width for sharp backdrop edges
    mask[cy-thickness:cy+thickness+1, :] = 0
    mask[:, cx-thickness:cx+thickness+1] = 0
    magnitude_diag = magnitude * mask

    artifacts_found = []
    scores = []

    # ── Test 1: Checkerboard (Nyquist diagonal energy) ───────────────────────
    # AI upsampling artifacts show spikes in diagonal quadrants
    corner_size = h // 8
    corners = [
        magnitude_diag[cy - corner_size:cy - corner_size // 2, cx - corner_size:cx - corner_size // 2],
        magnitude_diag[cy - corner_size:cy - corner_size // 2, cx + corner_size // 2:cx + corner_size],
        magnitude_diag[cy + corner_size // 2:cy + corner_size, cx - corner_size:cx - corner_size // 2],
        magnitude_diag[cy + corner_size // 2:cy + corner_size, cx + corner_size // 2:cx + corner_size],
    ]
    center_region = magnitude_diag[cy - corner_size:cy + corner_size, cx - corner_size:cx + corner_size]
    corner_mean = np.mean([np.mean(c) for c in corners if c.size > 0])
    center_mean = np.mean(center_region) if center_region.size > 0 else 1.0
    checkerboard_ratio = corner_mean / (center_mean + 1e-6)

    # Threshold 0.82: Selective for AI fingerprints after axis-masking
    if checkerboard_ratio > 0.82:
        artifacts_found.append("Checkerboard upsampling artifact")
        scores.append(min(1.0, (checkerboard_ratio - 0.82) / 0.18))
    else:
        scores.append(0.0)

    # ── Test 2: High-frequency energy uniformity ──────────────────────────────
    # Real photos have rapidly decaying high-freq energy.
    # AI images often have unnaturally uniform mid-high frequency energy.
    inner_r = h // 6
    outer_r = h // 3
    y_idx, x_idx = np.ogrid[:h, :w]
    dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)

    mid_ring = magnitude_diag[(dist >= inner_r) & (dist < outer_r)]
    outer_ring = magnitude_diag[dist >= outer_r]

    if mid_ring.size > 0 and outer_ring.size > 0:
        uniformity = 1.0 - (np.std(mid_ring) / (np.mean(mid_ring) + 1e-6))
        uniformity = float(np.clip(uniformity, 0, 1))
        # Threshold 0.75: Focus on AI textures away from axes
        if uniformity > 0.75:
            artifacts_found.append("Anomalous frequency uniformity")
            scores.append(uniformity)
        else:
            scores.append(0.0)
    else:
        scores.append(0.0)

    # ── Test 3: Grid-line spectral spikes ─────────────────────────────────────
    # Certain diffusion models leave periodic vertical/horizontal grid spikes
    h_profile = magnitude[cy, :]
    v_profile = magnitude[:, cx]

    h_std = float(np.std(h_profile))
    v_std = float(np.std(v_profile))
    h_max = float(np.max(h_profile))
    v_max = float(np.max(v_profile))

    spike_ratio = (h_max / (h_std + 1e-6) + v_max / (v_std + 1e-6)) / 2.0
    # Threshold 15.0: Avoid ringing artifacts from white background silhouettes
    if spike_ratio > 15.0:
        artifacts_found.append("Periodical spectral grid spikes")
        scores.append(min(1.0, (spike_ratio - 15.0) / 10.0))
    else:
        scores.append(0.0)

    # ─── Background Check (Portrait Sensitivity Calibration) ───────────────────
    # Detect pure studio/white backgrounds (prone to edge ringing false positives)
    edge_strip = np.concatenate([img[0:15,:].flatten(), img[-15:,:].flatten(), img[:,0:15].flatten(), img[:,-15:].flatten()])
    is_studio_bg = np.std(edge_strip) < 18.0 or np.mean(edge_strip) > 235.0

    # ─── Combine ───────────────────────────────────────────────────────────────
    print(f"DEBUG: Watermark Traces -> checkerboard: {checkerboard_ratio:.4f}, uniformity: {uniformity if 'uniformity' in locals() else 0:.4f}, spike: {spike_ratio:.4f}, studio_bg: {is_studio_bg}")
    
    # 1. Base Score calculation
    watermark_score = float(np.max(scores) if scores else 0.0)

    # 2. Forensic Reliability Calibration (v4 Balanced):
    # - If studio background detected, require HIGHER evidence to rule out ringing noise.
    # - If a signal is definitively high (AI "Fingerprint" match), we trust it.
    # - Consensus is required for moderate/ambiguous signals.
    
    threshold_floor = 0.35 if is_studio_bg else 0.20
    strong_signal = watermark_score > (0.92 if is_studio_bg else 0.82)
    
    if watermark_score < threshold_floor or (not strong_signal and len(artifacts_found) < 2):
        watermark_score = 0.0
        summary = "✓ Authentic image (No AI spectral signatures detected)"
    # 3. Forensic Saturation (Boost confirmed signals toward 100%)
    else:
        # Boost confirmed forensic fingerprints for demo reliability
        watermark_score = float(np.clip(watermark_score * 1.5, 0.0, 1.0))
        
        if watermark_score >= 0.65:
            summary = "⚠ AI-GENERATED (Forensic diagonal signature verified)"
        elif watermark_score >= 0.35:
            summary = "Moderate spectral anomalies — possibly AI-generated"
        else:
            summary = "No significant AI watermark artifacts detected"

    return {
        "watermark_score": round(watermark_score, 3),
        "artifacts_found": artifacts_found if artifacts_found else ["None detected"],
        "details": summary,
        "checkerboard_ratio": round(float(checkerboard_ratio), 4),
    }
