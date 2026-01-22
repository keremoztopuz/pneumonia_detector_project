import os
import torch
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, recall_score, 
    precision_score, f1_score, 
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns 

from src.config import DEVICE, MODEL_SAVE_PATH, CLASS_NAMES
from src.model import create_model
from src.dataset import create_dataloaders

def evaluate_model(model_name=None, save_path=None):
    model_path = save_path or MODEL_SAVE_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = create_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    all_preds = []
    all_labels = []

    train_loader, val_loader, test_loader = create_dataloaders()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="evaluating"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }
    
    return metrics, all_preds, all_labels

def print_results(metrics, all_preds, all_labels, save_plots=True):
    print(f"\n{'='*50}")
    print(f"accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall:    {metrics['recall']:.4f}")
    print(f"f1 score:  {metrics['f1']:.4f}")
    print(f"{'='*50}")
    
    print("\nclassification report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    
    if save_plots:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.xlabel("predicted")
        plt.ylabel("true")
        plt.title(f"confusion matrix - accuracy: {metrics['accuracy']*100:.2f}%")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        print("\nconfusion matrix saved to: confusion_matrix.png")

if __name__ == "__main__":
    metrics, all_preds, all_labels = evaluate_model()
    print_results(metrics, all_preds, all_labels)

    