import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from model.model import DeepfakeDetector
import io
import os

def apply_gaussian_noise(image_pil, sigma=0.1):
    img = np.array(image_pil).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, img.shape)
    noisy_img = np.clip(img + noise, 0, 1)
    return Image.fromarray((noisy_img * 255).astype(np.uint8))

def apply_gaussian_blur(image_pil, radius=2):
    return image_pil.filter(ImageFilter.GaussianBlur(radius))

def apply_jpeg_compression(image_pil, quality=30):
    buffer = io.BytesIO()
    image_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

def evaluate_robustness(image_path):
    detector = DeepfakeDetector()
    original_img = Image.open(image_path).convert('RGB')
    
    distortions = {
        "Original": original_img,
        "Gaussian Noise (σ=0.1)": apply_gaussian_noise(original_img, 0.1),
        "Gaussian Blur (r=2)": apply_gaussian_blur(original_img, 2),
        "JPEG Compression (q=30)": apply_jpeg_compression(original_img, 30)
    }
    
    print("\n--- Model Robustness Report ---")
    print(f"{'Distortion':<25} | {'Prediction':<12} | {'Confidence':<10}")
    print("-" * 55)
    
    for name, img in distortions.items():
        res = detector.predict(img)
        print(f"{name:<25} | {res['prediction']:<12} | {res['confidence']:.2%}")

if __name__ == "__main__":
    # Use a dummy test if no image provided
    print("Robustness Evaluation Suite initialized.")
    # For demonstration, we'll create a dummy image if testing
    dummy_img_path = "/tmp/robustness_test.jpg"
    Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)).save(dummy_img_path)
    evaluate_robustness(dummy_img_path)
