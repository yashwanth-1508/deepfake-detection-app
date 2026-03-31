import os
import torch
from PIL import Image
from model.model import DeepfakeDetector

def test_folder(detector, folder_path, expected_label):
    correct = 0
    total = 0
    print(f"\nTesting {folder_path} (Expected: {expected_label})...")
    for file in os.listdir(folder_path):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, file)
            img = Image.open(img_path).convert('RGB')
            res = detector.predict_robust(img)
            prediction = res.get('prediction', 'Unknown')
            is_correct = (prediction == expected_label)
            if is_correct: correct += 1
            total += 1
            raw = res['faces'][0]['raw_prob'] if res.get('faces') else 0
            print(f"  {file}: {prediction} (Raw: {raw:.4f}) {'[OK]' if is_correct else '[FAIL]'}")
    return correct, total

if __name__ == "__main__":
    detector = DeepfakeDetector()
    r_c, r_t = test_folder(detector, "presentation_samples/real", "Real")
    f_c, f_t = test_folder(detector, "presentation_samples/fake", "Deepfake")
    
    print(f"\nFINAL RESULTS:")
    print(f"Real Accuracy: {r_c}/{r_t} ({r_c/r_t*100:.1f}%)")
    print(f"Fake Accuracy: {f_c}/{f_t} ({f_c/f_t*100:.1f}%)")
    print(f"Overall Accuracy: {(r_c+f_c)/(r_t+f_t)*100:.1f}%")
