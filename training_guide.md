# Training Guide: Deepfake Detection

This guide explains how to train and fine-tune the deepfake detection model using your own dataset.

## 1. Dataset Preparation

To train the model, you need a collection of videos and a metadata file.

### Metadata CSV Format
Create a CSV file (e.g., `metadata.csv`) with the following columns:
- `filename`: The name of the video file (e.g., `video1.mp4`).
- `label`: `1` for Real, `0` for Deepfake.

Example (`metadata.csv`):
```csv
filename,label
real_video_1.mp4,1
fake_video_1.mp4,0
```

### Video Directory
Store all your video files in a single directory (e.g., `/path/to/my/videos/`).

---

## 2. Training the Model

The training process uses a pre-trained **Xception** model for feature extraction and a fine-tunable **LSTM** head to process sequences of frames.

### Step-by-Step Instructions

1.  **Activate your virtual environment:**
    ```bash
    source venv/bin/activate
    ```

2.  **Run the training script:**
    Execute the `model/train.py` script, providing the path to your video directory and your metadata CSV.
    ```bash
    python model/train.py /path/to/my/videos/ /path/to/metadata.csv
    ```

### Training Configuration
- **Epochs**: Default is 5.
- **Learning Rate**: Default is 0.001.
- **Batch Size**: Default is 4.

You can modify these parameters directly in `model/train.py` (lines 56 and 75).

---

## 3. How Training Works (Fine-Tuning)

- **Base Model (CNN)**: The Xception base model is **frozen**. This means its weights are not updated during training, saving significant computational resources.
- **Head (LSTM)**: Only the custom LSTM and Linear layers (the "Head") are trained. This allows the model to learn the temporal differences between real and fake videos.
- **Weights**: The fine-tuned weights are saved to `model/weights/fine_tuned_head.pth`.

---

## 4. Verifying the Training Loop (Dry Run)

If you don't have a dataset yet but want to verify that the training infrastructure is working, run the script without any arguments:

```bash
python model/train.py
```

This will run a **Synthetic Training (Dry Run)** using randomly generated data to ensure all dependencies and the training loop are correctly set up.

---

## 5. Using the Fine-Tuned Weights

Once training is complete, ensure the backend (`api/main.py`) is configured to load your new weights. You may need to update the weight loading logic in `model/model.py` to point to `fine_tuned_head.pth` if you want to use it exclusively.

---

## 6. Training with Kaggle Datasets

You can use the provided helper script to automatically download and prepare a sample dataset from Kaggle.

1.  **Ensure Kaggle API is configured**:
    - Make sure your `kaggle.json` is in `~/.kaggle/kaggle.json`.
    
2.  **Run the preparation script**:
    ```bash
    python3 -m model.prepare_kaggle_data
    ```
    This will:
    - Download the `simongraves/deepfake-dataset` (approx. 67MB).
    - Unzip it into a `kaggle_data/` folder.
    - Generate a `kaggle_data/metadata.csv` with the correct labels.

3.  **Start training**:
    ```bash
    python3 -m model.train kaggle_data/ kaggle_data/metadata.csv
    ```
