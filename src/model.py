import torch
import torch.nn as nn
import torchvision.models as models

def create_model(num_classes: int):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model 