import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
import time
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import HfApi

from config import Config
from dataset import get_dataloaders
from model import create_model
from utils import TrainingLogger, accuracy
from setup_data import setup_dataset

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_acc += accuracy(outputs, labels)
        
        pbar.set_postfix({'loss': loss.item()})
    
    return running_loss / len(train_loader), running_acc / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            running_acc += accuracy(outputs, labels)
    
    return running_loss / len(val_loader), running_acc / len(val_loader)

def main():
    # Setup dataset first
    # setup_dataset()
    
    Config.setup_directories()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_model(Config.NUM_CLASSES)
    model = model.to(device)
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(Config)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=Config.BASE_LR, weight_decay=Config.WEIGHT_DECAY)
    
    # Setup OneCycleLR scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=Config.NUM_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START
    )
    
    # Setup logger
    logger = TrainingLogger(Config.LOGS_DIR)
    
    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step the scheduler
        scheduler.step()
        
        # Log metrics and save if best model
        is_best = logger.log_metrics(epoch, train_loss, train_acc, val_loss, val_acc)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        }
        
        torch.save(checkpoint, Config.MODEL_DIR / f'checkpoint_epoch_{epoch}.pt')
        if is_best:
            torch.save(checkpoint, Config.MODEL_DIR / 'best_model.pt')
        
        # Plot current metrics
        logger.plot_metrics()

    # Upload to Hugging Face
    # api = HfApi()
    # api.upload_folder(
    #     folder_path=str(Config.MODEL_DIR),
    #     repo_id=Config.HF_REPO_ID,
    #     repo_type="model"
    # )

if __name__ == "__main__":
    main() 