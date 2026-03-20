import os
import subprocess
import pandas as pd
import shutil

def prepare_kaggle_data():
    dataset_name = "simongraves/deepfake-dataset"
    data_dir = "kaggle_data"
    
    # 1. Download dataset
    print(f"Downloading dataset {dataset_name}...")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "-p", data_dir, "--unzip"], check=True)
        print("Download complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        return

    # 2. Generate Metadata
    print("Generating metadata.csv...")
    metadata = []
    
    # Structure of simongraves/deepfake-dataset:
    # kaggle_data/deepfake/ -> label 0
    # kaggle_data/video/ -> label 1
    
    deepfake_dir = os.path.join(data_dir, "deepfake")
    video_dir = os.path.join(data_dir, "video")
    
    if os.path.exists(deepfake_dir):
        for f in os.listdir(deepfake_dir):
            if f.endswith(('.mp4', '.mov', '.MOV')):
                metadata.append({"filename": os.path.join("deepfake", f), "label": 0})
                
    if os.path.exists(video_dir):
        for f in os.listdir(video_dir):
            if f.endswith(('.mp4', '.mov', '.MOV')):
                metadata.append({"filename": os.path.join("video", f), "label": 1})
                
    if not metadata:
        print("No video files found in the dataset.")
        return
        
    df = pd.DataFrame(metadata)
    metadata_path = os.path.join(data_dir, "metadata.csv")
    df.to_csv(metadata_path, index=False)
    print(f"Metadata saved to {metadata_path}")
    
    print("\n--- Summary ---")
    print(f"Total videos: {len(metadata)}")
    print(f"Deepfake (0): {len(df[df['label'] == 0])}")
    print(f"Real (1): {len(df[df['label'] == 1])}")
    print("\nYou can now run training with:")
    print(f"python3 -m model.train {data_dir} {metadata_path}")

if __name__ == "__main__":
    prepare_kaggle_data()
