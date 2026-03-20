import urllib.request
import os

print("Downloading pre-trained deepfake detection model...")
url = "https://github.com/polimi-ispl/icpr2020dfdc/releases/download/v1.0/efficientnetb4_b0.pth"
os.makedirs("model/weights", exist_ok=True)
filepath = "model/weights/efficientnetb4_b0.pth"

try:
    urllib.request.urlretrieve(url, filepath)
    print(f"Successfully downloaded to {filepath}")
except Exception as e:
    print(f"Error downloading: {e}")
