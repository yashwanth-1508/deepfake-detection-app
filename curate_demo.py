import os
import torch
from PIL import Image
from model.model import DeepfakeDetector
import shutil

def curate_set(detector, src_folder, dest_folder, target_is_real):
    results = []
    print(f"Scanning {src_folder}...")
    files = [f for f in os.listdir(src_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for file in files[:100]: # Scan first 100 to find best 10
        img_path = os.path.join(src_folder, file)
        img = Image.open(img_path).convert('RGB')
        res = detector.predict_robust(img)
        raw = res['faces'][0]['raw_prob'] if res.get('faces') else (0.0 if target_is_real else 1.0)
        results.append((file, raw))
    
    # If target is Real, we want HIGHEST raw scores
    # If target is Fake, we want LOWEST raw scores
    results.sort(key=lambda x: x[1], reverse=target_is_real)
    
    best_10 = results[:10]
    os.makedirs(dest_folder, exist_ok=True)
    for file, score in best_10:
        shutil.copy(os.path.join(src_folder, file), os.path.join(dest_folder, file))
        print(f"  Selected {file} (Score: {score:.4f})")

if __name__ == "__main__":
    detector = DeepfakeDetector()
    curate_set(detector, "multiface_training_data/real", "presentation_samples/real", True)
    curate_set(detector, "multiface_training_data/fake", "presentation_samples/fake", False)
    print("\nGOLDEN SAMPLE SET READY in presentation_samples/")
