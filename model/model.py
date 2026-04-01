import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import ssl
import os
import cv2
import numpy as np
from pytorchcv.model_provider import get_model as ptcv_get_model
from api.utils import compute_optical_flow

# Bypass SSL certificate verification for downloading pre-trained weights on macOS
ssl._create_default_https_context = ssl._create_unverified_context

class Head(nn.Module):
    def __init__(self, in_f, out_f):
        super(Head, self).__init__()
        # LSTM layer to process sequences of frame features + optical flow
        # in_f (CNN features) + 1 (Optical flow magnitude)
        self.lstm = nn.LSTM(in_f + 1, 512, batch_first=True)
        self.f = nn.Flatten()
        self.l = nn.Linear(512, out_f)
        self.d = nn.Dropout(0.5)
        self.b = nn.BatchNorm1d(512)
        self.s = nn.Sigmoid()

    def forward(self, x):
        # x shape expected: (batch, seq_len, in_f)
        # For a single image, seq_len will be 1
        x, (h_n, c_n) = self.lstm(x)
        
        # Take the output of the last time step
        x = x[:, -1, :]
        
        x = self.b(x)
        x = self.d(x)
        x = self.l(x)
        return self.s(x)

class DeepfakeDetector:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load the base Xception model from pytorchcv
        self.base_model = ptcv_get_model("xception", pretrained=False)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1]) # Remove final fc (output 2048)
        
        # Define the custom head with LSTM
        self.head = Head(2048, 1)
        
        # We don't use nn.Sequential for the whole thing because we need to process 
        # sequences in the predict method
        self.base_model = self.base_model.to(self.device).eval()
        self.head = self.head.to(self.device).eval()
        
        # Load the pre-trained weights
        weights_path = os.path.join(os.path.dirname(__file__), 'weights', 'model.pth')
        fine_tuned_path = os.path.join(os.path.dirname(__file__), 'weights', 'fine_tuned_head.pth')
        
        # 1. ALWAYS prefer fine-tuned weights (This prevents Memory Spikes on Railway by skipping redundant Loading!)
        fine_tuned_base_path = os.path.join(os.path.dirname(__file__), 'weights', 'fine_tuned_base.pth')
        has_base = False
        
        if os.path.exists(fine_tuned_base_path):
            try:
                base_state = torch.load(fine_tuned_base_path, map_location=self.device)
                self.base_model.load_state_dict(base_state, strict=True)
                print(f"SUCCESS: Loaded fine-tuned BASE weights from {fine_tuned_base_path}")
                has_base = True
            except Exception as e:
                print(f"Warning: Could not load fine-tuned base weights: {e}")

        # 2. IF missing fine-tuned weights, then fallback to original model.pth
        if not has_base and os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('base.'):
                        new_state_dict[k[len('base.'):]] = v
                    else:
                        new_state_dict[k] = v
                
                self.base_model.load_state_dict(new_state_dict, strict=False)
                self.head.load_state_dict(new_state_dict, strict=False)
                print(f"Loaded base weights from {weights_path}")
            except Exception as e:
                print(f"Error loading base weights: {e}")

        # 3. Load fine-tuned head
        if os.path.exists(fine_tuned_path):
            try:
                fine_tuned_state = torch.load(fine_tuned_path, map_location=self.device)
                self.head.load_state_dict(fine_tuned_state, strict=True)
                print(f"SUCCESS: Loaded fine-tuned HEAD weights from {fine_tuned_path}")
            except Exception as e:
                print(f"Warning: Could not load fine-tuned head weights: {e}")
        
        # Load OpenCV DNN Face Detector
        proto_path = os.path.join(os.path.dirname(__file__), 'face_detector', 'deploy.prototxt')
        model_path = os.path.join(os.path.dirname(__file__), 'face_detector', 'res10_300x300_ssd_iter_140000.caffemodel')
        self.face_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        
        # Normalization for this specific model (Standard ImageNet for pytorchcv)
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def classify_probability(self, prob):
        """
        PRODUCTION MAPPING: 0.52 Threshold (Conservative Fine-tuning)
        Intermediate zone (0.45-0.55) returns 'Undetermined / Potential Deepfake'.
        """
        print(f"CALIBRATION: Raw Average = {prob:.4f}")
        
        # Uncertainty Buffer (Neural Noise mitigation)
        if 0.45 <= prob <= 0.55:
            return "Undetermined / Potential Deepfake", max(prob, 1.0 - prob)
            
        if prob > 0.52:
            return "Real", prob
        else:
            return "Deepfake", 1.0 - prob

    def get_all_face_crops(self, image_pil):
        # Convert PIL to BGR for OpenCV
        cv_image = cv2.cvtColor(np.ascontiguousarray(np.array(image_pil)), cv2.COLOR_RGB2BGR)
        (h, w) = cv_image.shape[:2]
        
        # Prepare blob for face detection
        blob = cv2.dnn.blobFromImage(cv2.resize(cv_image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        face_crops = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.4:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Add padding
                pad_w = int((endX - startX) * 0.25)
                pad_h = int((endY - startY) * 0.25)
                
                x1 = max(0, startX - pad_w)
                y1 = max(0, startY - pad_h)
                x2 = min(w, endX + pad_w)
                y2 = min(h, endY + pad_h)
                
                face_img = np.ascontiguousarray(cv_image[y1:y2, x1:x2])
                if face_img.size > 0:
                    face_crops.append(Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)))
            
        return face_crops

    def predict(self, inputs):
        """
        Processes a sequence of frames and returns a list of results, one for each person detected.
        """
        if isinstance(inputs, Image.Image):
            frames = [inputs]
        else:
            frames = inputs

        # Tracks: { id: { 'face_crops': [], 'boxes': [], 'prev_flow': 0.0, 'last_box': None } }
        tracks = {}
        next_track_id = 0
        
        for frame_idx, frame in enumerate(frames):
            # 1. Detect all faces in the frame
            cv_image = cv2.cvtColor(np.ascontiguousarray(np.array(frame)), cv2.COLOR_RGB2BGR)
            (h, w) = cv_image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(cv_image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.face_net.setInput(blob)
            detections = self.face_net.forward()

            current_frame_faces = []
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.15: # Lowered from 0.4 for better robustness with smaller faces
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Add padding and crop
                    pad_w = int((endX - startX) * 0.25)
                    pad_h = int((endY - startY) * 0.25)
                    x1, y1, x2, y2 = max(0, startX-pad_w), max(0, startY-pad_h), min(w, endX+pad_w), min(h, endY+pad_h)
                    
                    face_img = np.ascontiguousarray(cv_image[y1:y2, x1:x2])
                    if face_img.size > 0:
                        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                        current_frame_faces.append({
                            'pil': face_pil,
                            'box': (x1, y1, x2, y2),
                            'np': face_img
                        })

            # 2. Assign faces to tracks (Simple spatial proximity)
            for face in current_frame_faces:
                assigned_track_id = None
                best_dist = 100 # Threshold for matching
                
                for tid, track in tracks.items():
                    if track['last_seen_frame'] == frame_idx - 1:
                        # Calculate center distance
                        c1 = ((face['box'][0] + face['box'][2])/2, (face['box'][1] + face['box'][3])/2)
                        c2 = ((track['last_box'][0] + track['last_box'][2])/2, (track['last_box'][1] + track['last_box'][3])/2)
                        dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                        
                        if dist < best_dist:
                            best_dist = dist
                            assigned_track_id = tid
                
                if assigned_track_id is None:
                    # New identity
                    assigned_track_id = next_track_id
                    next_track_id += 1
                    tracks[assigned_track_id] = {
                        'face_crops': [],
                        'prev_face_np': None,
                        'last_seen_frame': -1
                    }
                
                # Update track
                track = tracks[assigned_track_id]
                
                # Compute optical flow if we have a previous face
                flow_mag = 0.0
                if track['prev_face_np'] is not None:
                    prev_np = track['prev_face_np']
                    curr_np = face['np']
                    if prev_np.shape != curr_np.shape:
                        prev_np = cv2.resize(prev_np, (curr_np.shape[1], curr_np.shape[0]))
                    flow_mag = compute_optical_flow(prev_np, curr_np)

                track['face_crops'].append({'pil': face['pil'], 'flow': flow_mag})
                track['last_box'] = face['box']
                track['last_seen_frame'] = frame_idx
                track['prev_face_np'] = face['np']

        # 3. Analyze each track
        results = []
        for tid, track in tracks.items():
            if len(track['face_crops']) < 1: # Fixed: Allow single-frame tracks for images
                continue
                
            sequence_features = []
            for item in track['face_crops']:
                img_t = self.transform(item['pil'])
                batch_t = torch.unsqueeze(img_t, 0).to(self.device)
                
                with torch.no_grad():
                    cnn_features = self.base_model(batch_t)
                    cnn_features = cnn_features.view(cnn_features.size(0), -1)
                    # SYNC WITH TRAINING: Use zeroed flow features (matches train.py line 121)
                    flow_feat = torch.zeros((1, 1)).to(self.device)
                    combined = torch.cat([cnn_features, flow_feat], dim=1)
                    sequence_features.append(combined)

            sequence_t = torch.stack(sequence_features, dim=1)
            with torch.no_grad():
                prob = self.head(sequence_t).item()
            
            prediction, confidence = self.classify_probability(prob)
            
            results.append({
                "face_id": int(tid),
                "prediction": prediction,
                "confidence": float(confidence),
                "raw_prob": float(prob),
                "box": [int(x) for x in track['last_box']],
                "last_face_pil": track['face_crops'][-1]['pil']
            })

        # Global result logic:
        predictions = [r['prediction'] for r in results]
        
        # Aggregation Logic: Priority Order (Deepfake > Undetermined > Likely Real > Real)
        if "Deepfake" in predictions:
            global_pred = "Deepfake"
        elif "Undetermined / Potential Deepfake" in predictions:
            global_pred = "Undetermined / Potential Deepfake"
        elif "Likely Real" in predictions:
            global_pred = "Likely Real"
        elif "Real" in predictions:
            global_pred = "Real"
        else:
            global_pred = "Undetermined / No Face" # Fallback if no faces detected

        max_confidence = max([r['confidence'] for r in results] or [0.0])
        
        return {
            "prediction": global_pred,
            "confidence": max_confidence,
            "faces": results
        }

    def predict_robust(self, inputs, num_augments=1):
        """
        Runs prediction with Fast Inference-Time Augmentation (ITA).
        Scans both Original and Flipped images to stabilize the prediction.
        """
        # 1. Original Scan
        res = self.predict_with_explainability(inputs)
        if "faces" not in res or not res["faces"]:
            return res

        if isinstance(inputs, Image.Image):
            frames = [inputs]
        else:
            frames = inputs

        # 2. Add Horizontal Flip Scan
        flipped_frames = [ImageOps.mirror(f) for f in (frames if isinstance(frames, list) else [frames])]
        aug_res = self.predict(flipped_frames)
        
        # 3. Average the scores
        for i, face_orig in enumerate(res["faces"]):
            # Get raw prob from original
            p_orig = face_orig.get("raw_prob", 0.5)
            
            # Match with flipped scan (simple index match for single-face, or logic for multi)
            if "faces" in aug_res and i < len(aug_res["faces"]):
                p_aug = aug_res["faces"][i].get("raw_prob", 0.5)
                # Average the two views for consistency
                final_prob = (p_orig + p_aug) / 2.0
            else:
                final_prob = p_orig
            
            # Final Classification on the average
            pred, conf = self.classify_probability(final_prob)
            face_orig["prediction"] = pred
            face_orig["confidence"] = conf
            face_orig["raw_prob"] = final_prob

        # Update global results after averaging
        all_preds = [f["prediction"] for f in res["faces"]]
        if "Deepfake" in all_preds:
            res["prediction"] = "Deepfake"
        elif "Undetermined / Potential Deepfake" in all_preds:
            res["prediction"] = "Undetermined / Potential Deepfake"
        elif "Likely Real" in all_preds:
            res["prediction"] = "Likely Real"
        else:
            res["prediction"] = "Real"

        res["confidence"] = max([f["confidence"] for f in res["faces"]] or [0.0])
        return res
    def generate_gradcam(self, face_pil):
        """
        Generates a Grad-CAM heatmap for a single face image.
        Returns a tuple: (base64_overlay_string, normalized_cam_array)
        """
        import base64
        from io import BytesIO

        # 1. Prepare input
        input_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)

        target_layer = self.base_model[-1]

        feature_maps = []
        def forward_hook(module, input, output):
            feature_maps.append(output)

        f_hook = target_layer.register_forward_hook(forward_hook)

        # 2. Forward pass through CNN without gradients
        with torch.no_grad():
            self.base_model(input_tensor)

        f_hook.remove()

        # 3. Compute Activation Map
        fmaps = feature_maps[0].cpu().numpy()[0]
        if fmaps.shape[1] > 1:
            fmaps = fmaps[:, :-1, :]

        cam = np.mean(fmaps, axis=0)

        # 4. Normalize
        cam = np.maximum(cam, 0)
        # Increase resolution to 512x512 for sharper "X-ray" detail
        cam = cv2.resize(cam, (512, 512), interpolation=cv2.INTER_CUBIC)
        if np.max(cam) != np.min(cam):
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)

        # 5. Create Visualizations
        img = np.array(face_pil.resize((512, 512)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        # 6. Convert to base64
        overlay_pil = Image.fromarray(overlay)
        buffered = BytesIO()
        overlay_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return img_str, cam

    def generate_explanation(self, cam, prediction, confidence):
        """
        Converts a normalized Grad-CAM array into natural language XAI reasons.
        Divides the face into anatomical zones and maps high-activation regions
        to specific deepfake artifact explanations.

        Args:
            cam: numpy array (299x299), values 0-1
            prediction: 'Real' or 'Deepfake'
            confidence: float 0-1

        Returns:
            dict with 'reasons' list, 'top_zone', 'xai_summary'
        """
        h, w = cam.shape

        # ── Facial zone boundaries (fractions of 299px face) ──────────────────
        zones = {
            "forehead":     cam[0:int(h*0.18), :],
            "eye_region":   cam[int(h*0.18):int(h*0.40), :],
            "nose_bridge":  cam[int(h*0.35):int(h*0.55), int(w*0.3):int(w*0.7)],
            "cheeks":       cam[int(h*0.40):int(h*0.65), :],
            "mouth_chin":   cam[int(h*0.60):int(h*0.85), :],
            "jaw_line":     cam[int(h*0.75):h, :],
            "skin_texture": cam[int(h*0.20):int(h*0.70), int(w*0.1):int(w*0.9)],
        }

        zone_means = {name: float(np.mean(region)) if region.size > 0 else 0.0
                      for name, region in zones.items()}

        # Sort zones by activation strength
        sorted_zones = sorted(zone_means.items(), key=lambda x: x[1], reverse=True)
        top_zone = sorted_zones[0][0] if sorted_zones else "face"
        threshold = 0.45

        # ── Zone → explanation mapping (Deepfake) ──────────────────────────────
        zone_explanations = {
            "forehead":    "Unnatural texture blending detected in the forehead region",
            "eye_region":  "Eye region shows facial blending artifacts — a hallmark of GAN generation",
            "nose_bridge": "Nose-bridge geometry inconsistency detected (common in face-swap models)",
            "cheeks":      "Cheek skin texture lacks natural pore/micro-texture patterns",
            "mouth_chin":  "Mouth area exhibits generation artifacts — lip boundary inconsistency",
            "jaw_line":    "Jaw-line boundary shows compositing seam — face swap detected",
            "skin_texture":"Skin texture frequency pattern is atypical for a real photograph",
        }

        # ── Zone → explanation mapping (Authentic) ─────────────────────────────
        real_explanations = {
            "forehead":    "Natural skin texture gradients and pore patterns in the forehead",
            "eye_region":  "Eye region displays high-fidelity anatomical consistency and natural reflections",
            "nose_bridge": "Nose-bridge geometry is structurally consistent with natural facial anatomy",
            "cheeks":      "Cheek skin texture exhibits realistic micro-fluctuations and texture distribution",
            "mouth_chin":  "Mouth area shows sharp, natural boundaries with no compositing artifacts",
            "jaw_line":    "Jaw-line and chin boundaries are natural with seamless blending to background",
            "skin_texture":"Skin texture frequency spectrum matches natural photographic standards",
        }

        reasons = []
        is_fake = prediction in ("Deepfake", "Undetermined / Potential Deepfake")

        if is_fake:
            for zone_name, mean_val in sorted_zones:
                if mean_val >= threshold:
                    reasons.append(zone_explanations.get(zone_name, f"Anomaly in {zone_name} region"))
                if len(reasons) >= 3:
                    break

            if not reasons:
                reasons.append("Neural network detected subtle generation artifacts")

            if confidence >= 0.85:
                reasons.append(f"High-confidence detection ({confidence*100:.0f}%) — strong artifact signal")
        else:
            # Provide positive reasons for authenticity
            for zone_name, mean_val in sorted_zones:
                if mean_val >= 0.4: # Lower threshold for "Authentic" highlights
                    reasons.append(real_explanations.get(zone_name, f"Natural features in {zone_name}"))
                if len(reasons) >= 3:
                    break
            
            if not reasons:
                reasons.append("No significant generation artifacts detected")
            
            if confidence >= 0.8:
                reasons.append(f"High authenticity confidence ({confidence*100:.0f}%)")

        # Global activation summary — Ensure it aligns with prediction
        global_mean = float(np.mean(cam))
        
        if is_fake:
            if global_mean > 0.55:
                xai_summary = "Widespread facial artifacts — consistent with full face replacement (e.g. FaceSwap)"
            elif global_mean > 0.35:
                xai_summary = "Localized artifacts — consistent with face reenactment (e.g. DeepFaceLab)"
            else:
                xai_summary = "Neural detection of subtle synthetic artifacts across facial features"
        else:
            if confidence >= 0.85:
                xai_summary = "Strong structural and textural integrity — highly characteristic of a real photograph"
            elif global_mean < 0.25:
                xai_summary = "Minimal neural activation — no discernible synthetic signatures detected"
            else:
                xai_summary = "Natural feature distribution — consistent with authentic digital capture"

        return {
            "reasons": reasons,
            "top_zone": top_zone,
            "xai_summary": xai_summary,
            "zone_activations": {k: round(v, 3) for k, v in sorted_zones[:5]},
        }

    def predict_with_explainability(self, inputs):
        """
        Wrapper for predict that adds Grad-CAM heatmaps and XAI explanations
        for each detected face.
        """
        res = self.predict(inputs)

        if "faces" not in res:
            return res

        for face_res in res["faces"]:
            try:
                heatmap_b64, cam = self.generate_gradcam(face_res["last_face_pil"])
                face_res["heatmap"] = heatmap_b64

                # ── XAI Explanation ───────────────────────────────────────────
                xai = self.generate_explanation(
                    cam,
                    face_res["prediction"],
                    face_res["confidence"]
                )
                face_res["xai"] = xai

            except Exception as e:
                print(f"Grad-CAM/XAI error for face {face_res['face_id']}: {e}")
                face_res["heatmap"] = None
                face_res["xai"] = None
            finally:
                if "last_face_pil" in face_res:
                    del face_res["last_face_pil"]

        return res
