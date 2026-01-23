import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from config import (
    MODEL_NAME, 
    NUM_CLASSES, 
    LEARNING_RATE, 
    EPOCHS, 
    BATCH_SIZE, 
    DROP_RATE, 
    DEVICE, 
    CLASS_WEIGHTS,
    PATIENCE,
    WEIGHT_DECAY,
    FOCAL_GAMMA,
    LABEL_SMOOTHING,
    MODEL_SAVE_PATH
)
from model import create_model
from dataset import create_dataloaders

class FocalLoss(nn.Module):
    def __init__(self, gamma=FOCAL_GAMMA, weight=None, label_smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction="none",
            weight=self.weight,
            label_smoothing=self.label_smoothing)
        
        pt = torch.exp(-ce_loss)
        focal_loss = ((1-pt)**self.gamma * ce_loss)
        return focal_loss.mean()

def validate_model(model, val_loader, train_loader, criterion):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
    avg_loss = running_loss / len(val_loader)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    train_loss = running_loss / len(train_loader)
        
    print(f"Validation loss: {avg_loss:.4f} | train_loss: {train_loss:.4f} | accuracy: {accuracy:.4f} precision: {precision:.4f}| recall: {recall:.4f} | f1: {f1:.4f}")
        
    return avg_loss
    
def train_model(model_name = None, save_path=None, epochs=None):
    train_loader, val_loader, test_loader = create_dataloaders()
    epochs = epochs or EPOCHS
    save_path = save_path or MODEL_SAVE_PATH

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)

    model = create_model(model_name=model_name)
    model.to(DEVICE)

    criterion = FocalLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\ntraining: {model_name or 'convnext_tiny'}")
    print(f"device: {DEVICE}, epochs: {epochs}\n") 

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_loss = validate_model(model, val_loader, train_loader, criterion)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"\nbest model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\npatience counter reached {PATIENCE}. early stopping")
                break
            

if __name__ == "__main__":
    train_model()