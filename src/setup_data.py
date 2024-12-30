import kagglehub
from pathlib import Path
import shutil
import os
from config import Config
import sys
import time
import requests
from zipfile import BadZipFile

def download_with_retry(max_retries=3, delay=5):
    """
    Attempt to download the dataset with retries
    """
    force = False
    for attempt in range(max_retries):
        try:
            print(f"Download attempt {attempt + 1}/{max_retries}...")
            downloaded_path = kagglehub.dataset_download(
                "mayurmadnani/imagenet-dataset",
                force=force  # Force fresh download on retry
            )
            
            if isinstance(downloaded_path, list):
                downloaded_path = downloaded_path[0]
            
            # Verify the download
            downloaded_path = Path(downloaded_path)
            if not downloaded_path.exists():
                raise FileNotFoundError("Download path does not exist")
                
            return downloaded_path
            
        except (BadZipFile, FileNotFoundError, requests.exceptions.RequestException) as e:
            print(f"Download attempt {attempt + 1} failed: {str(e)}")
            force = True
            if attempt < max_retries - 1:
                print(f"Waiting {delay} seconds before retrying...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print("All download attempts failed.")
                raise

def verify_dataset_structure(data_dir):
    """
    Verify that the dataset has the expected structure
    """
    required_splits = ['train', 'val']
    for split in required_splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            return False
        
        # Check if there are class directories
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        if not class_dirs:
            return False
    
    return True

def setup_dataset():
    """
    Downloads ImageNet dataset from kagglehub and organizes it in the specified directory
    """
    print("Setting up ImageNet dataset...")
    
    try:
        # Create data directory if it doesn't exist
        Config.DATA_ROOT.mkdir(parents=True, exist_ok=True)
        
        # Check if dataset already exists and is valid
        if verify_dataset_structure(Config.DATA_ROOT):
            print(f"Valid dataset already exists in {Config.DATA_ROOT}")
            print_dataset_stats()
            return
        
        # Download dataset with retry mechanism
        downloaded_path = download_with_retry()
        
        print(f"Moving dataset to {Config.DATA_ROOT}...")
        
        # Clear target directory if it exists
        if Config.DATA_ROOT.exists():
            print("Cleaning existing data directory...")
            shutil.rmtree(Config.DATA_ROOT)
        Config.DATA_ROOT.mkdir(parents=True)
        
        # Move files to permanent location
        for split in ['train', 'val', 'test']:
            src_dir = downloaded_path / split
            dst_dir = Config.DATA_ROOT / split
            
            if src_dir.exists():
                print(f"Setting up {split} split...")
                shutil.copytree(src_dir, dst_dir)
        
        # Verify the final dataset
        if not verify_dataset_structure(Config.DATA_ROOT):
            raise RuntimeError("Dataset setup failed - invalid structure")
        
        print("Dataset setup complete!")
        print_dataset_stats()
        
    except Exception as e:
        print(f"Error setting up dataset: {str(e)}")
        print("Please try running the setup script again.")
        sys.exit(1)

def print_dataset_stats():
    """
    Print statistics about the dataset
    """
    if Config.TRAIN_DIR.exists():
        num_train_classes = len([d for d in Config.TRAIN_DIR.iterdir() if d.is_dir()])
        print(f"Number of training classes: {num_train_classes}")
        
        train_images = sum(
            len([f for f in files if f.endswith(('.JPEG', '.jpg', '.png'))])
            for _, _, files in os.walk(Config.TRAIN_DIR)
        )
        print(f"Total training images: {train_images}")
    
    if Config.VAL_DIR.exists():
        num_val_classes = len([d for d in Config.VAL_DIR.iterdir() if d.is_dir()])
        print(f"Number of validation classes: {num_val_classes}")
        
        val_images = sum(
            len([f for f in files if f.endswith(('.JPEG', '.jpg', '.png'))])
            for _, _, files in os.walk(Config.VAL_DIR)
        )
        print(f"Total validation images: {val_images}")

if __name__ == "__main__":
    setup_dataset() 