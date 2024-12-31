from pathlib import Path

class Config:
    # Data paths
    DATA_ROOT = Path("/home/ubuntu/.cache/kagglehub/datasets/mayurmadnani/imagenet-dataset/versions/1")  # Default EC2 user home directory
    TRAIN_DIR = DATA_ROOT / "train"
    VAL_DIR = DATA_ROOT / "val"
    TEST_DIR = DATA_ROOT / "test"
    
    # Training parameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 90
    NUM_WORKERS = 4
    IMAGE_SIZE = 224
    NUM_CLASSES = 1000
    
    # Optimizer parameters
    BASE_LR = 1e-3
    WEIGHT_DECAY = 1e-4
    
    # One Cycle LR parameters
    MAX_LR = 3e-3
    PCT_START = 0.3
    
    # Model saving
    MODEL_DIR = Path("models")
    LOGS_DIR = Path("logs")
    
    # Debug mode (using 5% of data)
    DEBUG = True
    DEBUG_FRACTION = 0.05
    
    # Hugging Face
    HF_REPO_ID = "your-username/resnet50-imagenet"
    
    @classmethod
    def setup_directories(cls):
        cls.MODEL_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True) 