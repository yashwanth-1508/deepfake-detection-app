import os
import pandas as pd
import subprocess

def prepare_multiface_training_data():
    dataset_name = "vijaydevane/deepfake-detection-challenge-dataset-face-images"
    data_dir = "multiface_training_data"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    print(f"Downloading {dataset_name} (126MB)...")
    try:
        # Download dataset from Kaggle
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "--unzip", "-p", data_dir], check=True)
        print("Download complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        return

    # This dataset has 'real' and 'fake' folders with images
    csv_data = []
    for label_name, label_val in [('real', 1), ('fake', 0)]:
        label_dir = os.path.join(data_dir, label_name)
        if os.path.exists(label_dir):
            for file in os.listdir(label_dir):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    csv_data.append({
                        'image_path': os.path.join(label_name, file),
                        'label': label_val
                    })
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(data_dir, 'metadata.csv')
        df.to_csv(csv_path, index=False)
        print(f"Created metadata.csv at {csv_path} with {len(df)} entries.")
    else:
        print("Error: No images found in the downloaded dataset.")

if __name__ == "__main__":
    prepare_multiface_training_data()
