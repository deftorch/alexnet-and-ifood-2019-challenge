import os
import argparse
import torch
import torch.nn as nn
from src.data_loader import get_dataloaders
from src.models.alexnet import get_model
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

def main():
    parser = argparse.ArgumentParser(description='Evaluate AlexNet')
    parser.add_argument('--data_dir', type=str, default='data_mock', help='Path to data directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--model_name', type=str, default='alexnet_baseline', help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get class names
    class_list_file = os.path.join(args.data_dir, 'class_list.txt')
    with open(class_list_file, 'r') as f:
        classes = [line.strip().split(' ')[1] for line in f.readlines()]
    num_classes = len(classes)

    # Load model
    model = get_model(args.model_name, num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    dataloaders = get_dataloaders(args.data_dir, batch_size=args.batch_size, num_workers=2)

    if 'val' in dataloaders:
        print("Evaluating on Validation Set...")
        labels, preds = evaluate_model(model, dataloaders['val'], device)
        acc = accuracy_score(labels, preds)
        print(f"Validation Accuracy: {acc:.4f}")

        # Save confusion matrix or other metrics if needed
        cm = confusion_matrix(labels, preds)
        print("Confusion Matrix shape:", cm.shape)

    else:
        print("No validation set found.")

if __name__ == "__main__":
    main()
