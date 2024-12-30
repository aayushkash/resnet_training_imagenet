import kagglehub
from pathlib import Path
import shutil
import os
from config import Config

def setup_dataset():
    """
    Downloads ImageNet dataset from kagglehub and organizes it in the specified directory
    """
    print("Downloading ImageNet dataset...")
    
    # Create data directory if it doesn't exist
    Config.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    downloaded_path = kagglehub.dataset_download("mayurmadnani/imagenet-dataset")
    
    if isinstance(downloaded_path, list):
        downloaded_path = downloaded_path[0]
    
    downloaded_path = Path(downloaded_path)
    
    # Check if the dataset is already in the correct location
    if (Config.DATA_ROOT / "train").exists() and (Config.DATA_ROOT / "val").exists():
        print(f"Dataset already exists in {Config.DATA_ROOT}")
        return
    
    # Move or copy files to the permanent location
    print(f"Moving dataset to {Config.DATA_ROOT}...")
    
    for split in ['train', 'val', 'test']:
        src_dir = downloaded_path / split
        dst_dir = Config.DATA_ROOT / split
        
        if src_dir.exists():
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
    
    print("Dataset setup complete!")
    
    # Print some statistics
    if Config.TRAIN_DIR.exists():
        num_train_classes = len(os.listdir(Config.TRAIN_DIR))
        print(f"Number of training classes: {num_train_classes}")
        
        # Count total training images
        train_images = sum(len(files) for _, _, files in os.walk(Config.TRAIN_DIR))
        print(f"Total training images: {train_images}")
    
    if Config.VAL_DIR.exists():
        num_val_classes = len(os.listdir(Config.VAL_DIR))
        print(f"Number of validation classes: {num_val_classes}")
        
        # Count total validation images
        val_images = sum(len(files) for _, _, files in os.walk(Config.VAL_DIR))
        print(f"Total validation images: {val_images}")

if __name__ == "__main__":
    setup_dataset() 