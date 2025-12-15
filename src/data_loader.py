import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np

class IFoodDataset(Dataset):
    def __init__(self, root_dir, csv_file, class_list_file, transform=None, mode='train'):
        """
        Args:
            root_dir (string): Directory with all the images (extracted from tar).
            csv_file (string): Path to the csv file with annotations.
            class_list_file (string): Path to the class list file.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (string): 'train', 'val', or 'test'.
        """
        self.root_dir = root_dir
        self.data_info = pd.read_csv(csv_file)
        self.transform = transform
        self.mode = mode

        # Load class names
        with open(class_list_file, 'r') as f:
            self.classes = [line.strip().split(' ')[1] for line in f.readlines()]

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_info.iloc[idx, 0])

        try:
            image = Image.open(img_name).convert('RGB')
        except (IOError, FileNotFoundError):
            # In case of missing image, return a black image or raise error
            # For robustness, we might want to skip, but Dataset doesn't allow skipping easily.
            # We will generate a random image or black image.
            print(f"Warning: Image {img_name} not found. Using black image.")
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        if self.mode == 'test':
             return image, self.data_info.iloc[idx, 0] # Return image and name for submission

        label = int(self.data_info.iloc[idx, 1])
        return image, label

def get_dataloaders(data_dir, batch_size=32, num_workers=4, transforms=None):
    """
    Helper function to create dataloaders.
    Assumes data_dir contains:
        train_images/
        val_images/
        test_images/
        train_info.csv
        val_info.csv
        test_info.csv
        class_list.txt
    """
    if transforms is None:
        # Default transforms if none provided
        from torchvision import transforms as T
        transforms = {
            'train': T.Compose([
                T.Resize((256, 256)),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': T.Compose([
                T.Resize((256, 256)),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
             'test': T.Compose([
                T.Resize((256, 256)),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    dataloaders = {}

    # Train
    if os.path.exists(os.path.join(data_dir, 'train_info.csv')):
        train_dataset = IFoodDataset(
            root_dir=os.path.join(data_dir, 'train_images'),
            csv_file=os.path.join(data_dir, 'train_info.csv'),
            class_list_file=os.path.join(data_dir, 'class_list.txt'),
            transform=transforms.get('train'),
            mode='train'
        )
        dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Val
    if os.path.exists(os.path.join(data_dir, 'val_info.csv')):
        val_dataset = IFoodDataset(
            root_dir=os.path.join(data_dir, 'val_images'),
            csv_file=os.path.join(data_dir, 'val_info.csv'),
            class_list_file=os.path.join(data_dir, 'class_list.txt'),
            transform=transforms.get('val'),
            mode='val'
        )
        dataloaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Test
    if os.path.exists(os.path.join(data_dir, 'test_info.csv')):
        test_dataset = IFoodDataset(
            root_dir=os.path.join(data_dir, 'test_images'),
            csv_file=os.path.join(data_dir, 'test_info.csv'),
            class_list_file=os.path.join(data_dir, 'class_list.txt'),
            transform=transforms.get('test'),
            mode='test'
        )
        dataloaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloaders
