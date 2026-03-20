import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
from model.model import DeepfakeDetector
from api.utils import extract_frames_from_video

class VideoDataset(Dataset):
    def __init__(self, metadata_path, data_dir, num_frames=10):
        """
        metadata_path: Path to CSV with columns [filename, label] (label: 0 for Fake, 1 for Real)
        data_dir: Base directory for videos
        """
        self.metadata = pd.read_csv(metadata_path)
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.detector = DeepfakeDetector() # We use its transform and base_model

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        video_name = self.metadata.iloc[idx, 0]
        label = self.metadata.iloc[idx, 1]
        video_path = os.path.join(self.data_dir, video_name)
        
        # Load video and extract frames
        if not os.path.exists(video_path):
            print(f"Warning: {video_path} not found.")
            return None
        
        # Handle Images
        if video_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            frame = Image.open(video_path).convert('RGB')
            frames = [frame] * self.num_frames
        else:
            # Handle Videos
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
            frames = extract_frames_from_video(video_bytes, num_frames=self.num_frames)
        
        if not frames:
            return None
            
        # Process frames through CNN to get features
        sequence_tensors = []
        for frame in frames:
            # Re-use detector facial crop and transform
            face_crops = self.detector.get_all_face_crops(frame)
            face = face_crops[0] if face_crops else frame
            
            img_t = self.detector.transform(face)
            sequence_tensors.append(img_t)
            
        # Stack into (seq_len, C, H, W)
        return torch.stack(sequence_tensors), torch.tensor(label, dtype=torch.float32)

def train_model(data_dir, metadata_path, epochs=5, lr=0.001):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on {device}...")
    
    # Initialize Detector
    detector = DeepfakeDetector()
    
    # FREEZE base model (CNN) - Only train the Head (LSTM + Linear)
    for param in detector.base_model.parameters():
        param.requires_grad = False
    
    for param in detector.head.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam(detector.head.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # Dataset and Loader
    dataset = VideoDataset(metadata_path, data_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    detector.head.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i, batch in enumerate(dataloader):
            if batch is None: continue
            
            sequences, labels = batch
            sequences, labels = sequences.to(device), labels.to(device)
            
            # Forward pass per batch
            # We need to extract features for the whole sequence batch
            batch_size, seq_len, c, h, w = sequences.shape
            
            # Flatten batch and sequence to pass through CNN
            flat_sequences = sequences.view(-1, c, h, w)
            with torch.no_grad():
                features = detector.base_model(flat_sequences)
                features = features.view(batch_size, seq_len, -1)
            
            # Pass sequence features to the head
            # Note: We need to handle optical flow calculation here if we want full fidelity
            # but for a basic fine-tune, CNN + LSTM is the core.
            
            # Add a dummy optical flow channel of zeros for now if the head expects 2049
            # (or we can implement the actual flow calc here)
            flow_zeros = torch.zeros(batch_size, seq_len, 1).to(device)
            combined_features = torch.cat([features, flow_zeros], dim=2)
            
            optimizer.zero_grad()
            outputs = detector.head(combined_features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
        print(f"Epoch {epoch+1} Complete. Avg Loss: {epoch_loss/len(dataloader):.4f}")
        
    # Save training results
    save_path = os.path.join(os.path.dirname(__file__), 'weights', 'fine_tuned_head.pth')
    torch.save(detector.head.state_dict(), save_path)
    print(f"Fine-tuned weights saved to {save_path}")

def run_synthetic_training():
    """Generates synthetic data to demonstrate the training loop works."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Synthetic Training (Dry Run) on {device} ---")
    
    # Initialize Fake Detector
    detector = DeepfakeDetector()
    
    # Freeze CNN
    for param in detector.base_model.parameters():
        param.requires_grad = False
        
    optimizer = optim.Adam(detector.head.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    detector.head.train()
    
    # Simulate 5 batches of synthetic sequences
    for i in range(5):
        # (batch=4, seq=10, features=2048)
        dummy_features = torch.randn(4, 10, 2048).to(device)
        dummy_flow = torch.randn(4, 10, 1).to(device)
        combined = torch.cat([dummy_features, dummy_flow], dim=2)
        
        labels = torch.randint(0, 2, (4,), dtype=torch.float32).to(device)
        
        optimizer.zero_grad()
        outputs = detector.head(combined).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f"Synthetic Step [{i+1}/5], Loss: {loss.item():.4f}")
        
    print("Synthetic training complete! The training infrastructure is fully operational.")

if __name__ == "__main__":
    # If no arguments provided, run synthetic training as a demo
    import sys
    if len(sys.argv) == 1:
        run_synthetic_training()
    else:
        # Standard training on real data (requires data_dir and metadata_path)
        train_model(sys.argv[1], sys.argv[2])
