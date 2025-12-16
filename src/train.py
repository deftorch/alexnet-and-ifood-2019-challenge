import os
import argparse
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from src.data_loader import get_dataloaders
from src.models.alexnet import get_model
from tqdm import tqdm
import numpy as np

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device='cpu', use_wandb=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} epoch {epoch}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if use_wandb:
                wandb.log({f"{phase}_loss": epoch_loss, f"{phase}_acc": epoch_acc, "epoch": epoch})

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def main():
    parser = argparse.ArgumentParser(description='Train AlexNet on iFood 2019')
    parser.add_argument('--data_dir', type=str, default='data_mock', help='Path to data directory')
    parser.add_argument('--model_name', type=str, default='alexnet_baseline',
                        choices=['alexnet_baseline', 'alexnet_mod1', 'alexnet_mod2', 'alexnet_combined'],
                        help='Model architecture to use')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')

    args = parser.parse_args()

    if args.use_wandb:
        wandb.init(project="ifood-alexnet", config=args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders = get_dataloaders(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    # Get class names to determine num_classes
    class_list_file = os.path.join(args.data_dir, 'class_list.txt')
    with open(class_list_file, 'r') as f:
        num_classes = len(f.readlines())

    print(f"Number of classes: {num_classes}")

    model = get_model(args.model_name, num_classes)
    model = model.to(device)

    # Handle class imbalance (Optional: Weighted Cross Entropy)
    if 'train' in dataloaders:
        print("Calculating class weights for imbalance handling...")
        train_dataset = dataloaders['train'].dataset
        # Extract labels from the dataset.
        # Note: This assumes the dataset has a data_info attribute with labels in the second column (index 1)
        try:
            # Efficiently get all labels
            all_labels = train_dataset.data_info.iloc[:, 1].tolist()
            class_counts = np.bincount(all_labels, minlength=num_classes)

            # Calculate weights: inverse frequency
            # Add small epsilon to avoid division by zero if a class has 0 samples
            weights = 1.0 / (class_counts + 1e-6)

            # Normalize weights so they sum to num_classes (optional, but keeps scale consistent)
            weights = weights / weights.sum() * num_classes

            # Convert to tensor
            class_weights = torch.FloatTensor(weights).to(device)
            print("Class weights calculated.")

            criterion = nn.CrossEntropyLoss(weight=class_weights)
        except Exception as e:
            print(f"Warning: Could not calculate class weights: {e}. Using standard CrossEntropyLoss.")
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    model, val_hist = train_model(model, dataloaders, criterion, optimizer,
                                  num_epochs=args.num_epochs, device=device, use_wandb=args.use_wandb)

    # Save model
    save_path = f"model_{args.model_name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
