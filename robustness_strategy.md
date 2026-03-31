# Robustness Strategy: Handling Real-World Images

When AI models are tested with "outside" images (not from the training set), they often fail due to:
1.  **Compression Artifacts**: Images from social media (WhatsApp, Instagram) are heavily compressed, which can look like "fake" textures to a model.
2.  **Lighting/Noise**: Training data is often clear, but real-world photos have sensor noise or poor lighting.
3.  **Adversarial Examples**: If a fake is very well made, it might barely show any "signature" of manipulation.

## Our Approach to Fixing This

### 1. Inference-Time Augmentation (ITA)
Instead of asking the model once, we "perturb" the input image (slight blur, slight brightness change) and ask it 4-5 times. If the model is consistently saying "Deepfake" across all versions, we can be much more confident. This filters out errors caused by random noise in a single image.

### 2. Multi-Face Tracking
If an image has multiple people, the model should analyze each face individually. We have implemented a tracking system that isolates each face and gives a separate prediction, ensuring that a "Real" person next to a "Deepfake" person doesn't confuse the system.

### 3. Spatial-Temporal Consistency
For videos, we don't just look at frames; we look at the **flow** of pixels between frames (Optical Flow). Deepfakes often have "flickering" near the chin or eyes when moving. Our LSTM model is specifically trained to catch these temporal inconsistencies that are invisible to the naked eye.

### 4. Re-Training with "Hard" Examples
The next step is to collect the images that our model missed and include them in the training set as "Hard Examples." This "Active Learning" cycle is the gold standard for improving model accuracy.
