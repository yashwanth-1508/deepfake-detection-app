import os
import torch
from PIL import Image
from model.model import DeepfakeDetector

def audit_image(detector, img_path, expected_label):
    if not os.path.exists(img_path):
        print(f"  FILE NOT FOUND: {img_path}")
        return
        
    img = Image.open(img_path).convert('RGB')
    print(f"\nScanning: {img_path} (Expected: {expected_label})")
    res = detector.predict_robust(img) # Use our calibrated robust logic
    
    print(f"  GLOBAL PREDICTION: {res['prediction']} (Confidence: {res['confidence']*100:.1f}%)")
    audit_image(detector, 'global_audit/real/obama.jpg', 'Real')
    audit_image(detector, 'global_audit/fake/gan_screenshot.png', 'Deepfake')
