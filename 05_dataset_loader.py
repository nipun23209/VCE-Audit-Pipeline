"""
Phase 5: Medical Dataset Integration
Team: Nipun Aditya, Shiv Raj Singh, Subrat Kumar Pattanayak
Description: Custom PyTorch dataset class to ingest the Kaggle Chest X-Ray dataset. Includes a custom collate function to handle metadata (file paths) alongside the tensors.
Dataset Source: Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification", Mendeley Data. Hosted on Kaggle.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, split="test"):
        self.root_dir = os.path.join(root_dir, split)
        self.image_paths = []
        self.labels = []
        self.classes = {"NORMAL": 0, "PNEUMONIA": 1}
        
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Could not find the dataset directory at: {self.root_dir}")
            
        for class_name, class_label in self.classes.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found.")
                continue
                
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.png', '.jpeg', '.jpg')):
                    self.image_paths.append(os.path.join(class_dir, file_name))
                    self.labels.append(class_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Fallback for corrupted images to prevent pipeline crash
            image = Image.new('RGB', (224, 224), color='black') 
            
        label = self.labels[idx]
        return image, label, img_path

# --- THE FIX ---
# This overrides PyTorch's default math-only batching and safely returns lists so we can keep track of filenames for our CSV.
def custom_collate(batch):
    # 'batch' is a list of tuples: [(image1, label1, path1), ...]
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    paths = [item[2] for item in batch]
    return images, labels, paths
# ---------------

def test_dataloader():
    print("Initializing PyTorch DataLoader...")
    dataset_path = "chest_xray" 
    
    try:
        test_dataset = ChestXRayDataset(root_dir=dataset_path, split="test")
        print(f"\nSuccessfully indexed {len(test_dataset)} Chest X-Rays from the SSD.")
        
        # We inject our custom collate function here
        test_loader = DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=True, 
            collate_fn=custom_collate
        )
        
        print("\nTesting retrieval of the first batch...")
        for images, labels, paths in test_loader:
            print(f"Image Object: {type(images[0])}")
            print(f"Ground Truth Label: {labels[0]}")
            print(f"File Path: {paths[0]}")
            break 
            
        print("\n-------------------------------------------")
        print("DataLoader Test Passed. Ready for Phase 4.")
        print("-------------------------------------------")
        
    except FileNotFoundError as e:
        print(f"\n[!] DATASET ERROR: {e}")

if __name__ == "__main__":
    test_dataloader()