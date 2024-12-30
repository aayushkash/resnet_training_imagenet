import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import logging

class TrainingLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0
        
        # Setup logging
        logging.basicConfig(
            filename=log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
    def log_metrics(self, epoch: int, train_loss: float, train_acc: float, 
                   val_loss: float, val_acc: float):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        
        logging.info(f"Epoch {epoch}: {metrics}")
        
        # Save metrics to JSON
        with open(self.log_dir / 'metrics.json', 'w') as f:
            json.dump({
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accs': self.train_accs,
                'val_accs': self.val_accs
            }, f)
        
        # Update best validation accuracy
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            return True
        return False
        
    def plot_metrics(self):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc')
        plt.plot(self.val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_curves.png')
        plt.close()

def accuracy(output, target):
    _, predicted = output.max(1)
    total = target.size(0)
    correct = predicted.eq(target).sum().item()
    return correct / total 