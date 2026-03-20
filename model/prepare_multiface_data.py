import os
import json
import pandas as pd
import subprocess

def prepare_multiface_data():
    dataset_name = "krishna191919/dfdc-train-sample-dataset"
    data_dir = "multiface_data"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    print(f"Downloading {dataset_name}...")
    try:
        # Download dataset from Kaggle
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "--unzip", "-p", data_dir], check=True)
        print("Download complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        return

    # DFDC Sample usually has a metadata.json
    metadata_json_path = os.path.join(data_dir, 'metadata.json')
    if not os.path.exists(metadata_json_path):
        # Check subdirectories
        for root, dirs, files in os.walk(data_dir):
            if 'metadata.json' in files:
                metadata_json_path = os.path.join(root, 'metadata.json')
                break
                
    if os.path.exists(metadata_json_path):
        print(f"Found metadata at {metadata_json_path}. Converting to CSV...")
        with open(metadata_json_path, 'r') as f:
            meta_data = json.load(f)
            
        csv_data = []
        for video_file, info in meta_data.items():
            label = 1 if info['label'] == 'REAL' else 0
            # Resolve actual path relative to multiface_data
            actual_path = None
            for root, dirs, files in os.walk(data_dir):
                if video_file in files:
                    # Store path relative to data_dir
                    actual_path = os.path.relpath(os.path.join(root, video_file), data_dir)
                    break
            
            if actual_path:
                csv_data.append({'video_path': actual_path, 'label': label})
        
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(data_dir, 'metadata.csv')
        df.to_csv(csv_path, index=False)
        print(f"Created metadata.csv at {csv_path} with {len(df)} entries.")
    else:
        print("Warning: metadata.json not found. You may need to create metadata.csv manually.")

if __name__ == "__main__":
    prepare_multiface_data()
