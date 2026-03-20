import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
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
        
        # 1. Load base model weights if available
        if os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                # Check for 'base.' prefix and strip it
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('base.'):
                        new_state_dict[k[len('base.'):]] = v
                    else:
                        new_state_dict[k] = v
                
                # Load bits that fit into base_model and head
                # We use strict=False because some keys might be for head, some for base
                self.base_model.load_state_dict(new_state_dict, strict=False)
                self.head.load_state_dict(new_state_dict, strict=False)
                print(f"Loaded base weights from {weights_path}")
            except Exception as e:
                print(f"Error loading base weights: {e}")

        # 2. OVERRIDE with fine-tuned head weights if available
        if os.path.exists(fine_tuned_path):
            try:
                fine_tuned_state = torch.load(fine_tuned_path, map_location=self.device)
                self.head.load_state_dict(fine_tuned_state, strict=True)
                print(f"SUCCESS: Loaded fine-tuned weights from {fine_tuned_path}")
            except Exception as e:
                print(f"Warning: Could not load fine-tuned weights: {e}")
        
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

    def get_all_face_crops(self, image_pil):
        # Convert PIL to BGR for OpenCV
        cv_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
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
                
                face_img = cv_image[y1:y2, x1:x2]
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
            cv_image = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
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
                    
                    face_img = cv_image[y1:y2, x1:x2]
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
                    flow_t = torch.tensor([[item['flow']]], device=self.device, dtype=torch.float32)
                    combined = torch.cat([cnn_features, flow_t], dim=1)
                    sequence_features.append(combined)

            sequence_t = torch.stack(sequence_features, dim=1)
            with torch.no_grad():
                prob = self.head(sequence_t).item()
            
            prediction = "Deepfake" if prob < 0.5 else "Real"
            confidence = 1 - prob if prob < 0.5 else prob
            
            results.append({
                "face_id": int(tid),
                "prediction": prediction,
                "confidence": float(confidence),
                "box": [int(x) for x in track['last_box']],
                "last_face_pil": track['face_crops'][-1]['pil']
            })

        # Final Aggregation
        if not results:
            return { "prediction": "No Faces Detected", "confidence": 0.0, "faces": [] }
            
        # If ANY face is deepfake, the video is flagged
        is_deepfake = any(r['prediction'] == "Deepfake" for r in results)
        max_confidence = max(r['confidence'] for r in results)
        
        return {
            "prediction": "Deepfake" if is_deepfake else "Real",
            "confidence": max_confidence,
            "faces": results
        }

    def predict_robust(self, inputs, num_augments=3):
        """
        Runs prediction with Inference-Time Augmentation (ITA) for improved robustness.
        Averages predictions from original and slightly augmented versions of the input.
        """
        # Original prediction
        orig_res = self.predict(inputs)
        all_probs = [orig_res['confidence'] if orig_res['prediction'] == "Real" else 1 - orig_res['confidence']]
        
        # Augmented predictions
        if isinstance(inputs, Image.Image):
            frames = [inputs]
        else:
            frames = inputs
            
        for _ in range(num_augments):
            # Apply light random augmentations (blur, brightness) to the whole sequence
            aug_frames = []
            for frame in frames:
                # Random brightness
                enhancer = ImageEnhance.Brightness(frame)
                aug_frame = enhancer.enhance(np.random.uniform(0.8, 1.2))
                # Random blur
                if np.random.rand() > 0.5:
                    aug_frame = aug_frame.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0, 1)))
                aug_frames.append(aug_frame)
            
            res = self.predict(aug_frames)
            prob = res['confidence'] if res['prediction'] == "Real" else 1 - res['confidence']
            all_probs.append(prob)
            
        avg_prob = np.mean(all_probs)
        
        if avg_prob < 0.5:
            prediction = "Deepfake"
            confidence = 1 - avg_prob
        else:
            prediction = "Real"
            confidence = avg_prob
            
        return {
            "prediction": prediction,
            "confidence": confidence,
            "robust_score": True
        }
    def generate_gradcam(self, face_pil):
        """
        Generates a Grad-CAM heatmap for a single face image.
        Returns a base64 encoded string of the heatmap overlay.
        """
        import torch.nn.functional as F
        import base64
        from io import BytesIO

        # 1. Prepare input
        input_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True

        # 2. Hooks to capture activations and gradients
        # In Xception from pytorchcv, the last convolutional layer is usually in the last block of 'features'
        # Let's target the last block in the base_model (which is features)
        target_layer = self.base_model[-1]
        
        feature_maps = []
        gradients = []

        def forward_hook(module, input, output):
            feature_maps.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        f_hook = target_layer.register_forward_hook(forward_hook)
        b_hook = target_layer.register_full_backward_hook(backward_hook)

        # 3. Forward pass through CNN
        self.base_model.zero_grad()
        features = self.base_model(input_tensor)
        features_flat = features.view(features.size(0), -1)
        
        # We need a scalar to backprop from. We'll use the mean of the features 
        # as a proxy for the 'class score' before the LSTM/Linear head.
        score = features_flat.mean()
        
        # 4. Backward pass
        score.backward()

        # 5. Compute Grad-CAM
        grads = gradients[0].cpu().data.numpy()
        fmaps = feature_maps[0].cpu().data.numpy()
        
        weights = np.mean(grads, axis=(2, 3))[0]
        cam = np.zeros(fmaps.shape[2:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * fmaps[0, i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (299, 299))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        # 6. Create Visualizations
        img = np.array(face_pil.resize((299, 299)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        
        # 7. Convert to base64
        overlay_pil = Image.fromarray(overlay)
        buffered = BytesIO()
        overlay_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Clean up hooks
        f_hook.remove()
        b_hook.remove()

        return img_str

    def predict_with_explainability(self, inputs):
        """
        Wrapper for predict that also adds Grad-CAM heatmaps for each detected face.
        """
        res = self.predict(inputs)
        
        if "faces" not in res:
            return res
            
        for face_res in res["faces"]:
            try:
                # Generate heatmap for the last crop of this face track
                heatmap_b64 = self.generate_gradcam(face_res["last_face_pil"])
                face_res["heatmap"] = heatmap_b64
                # Clean up for JSON
                del face_res["last_face_pil"]
            except Exception as e:
                print(f"Grad-CAM error for face {face_res['face_id']}: {e}")
                face_res["heatmap"] = None
            
        return res
