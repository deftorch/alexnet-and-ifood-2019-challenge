import torch
import torch.nn as nn
from torchvision.models import AlexNet

class AlexNetCustom(nn.Module):
    def __init__(self, num_classes=251, modification=None):
        super(AlexNetCustom, self).__init__()
        self.modification = modification

        # Define features
        layers = []

        # 1st Conv Layer
        layers.append(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2))
        if self.modification == 'batch_norm' or self.modification == 'combined':
            layers.append(nn.BatchNorm2d(64))

        if self.modification == 'activation' or self.modification == 'combined':
            layers.append(nn.LeakyReLU(inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        # 2nd Conv Layer
        layers.append(nn.Conv2d(64, 192, kernel_size=5, padding=2))
        if self.modification == 'batch_norm' or self.modification == 'combined':
            layers.append(nn.BatchNorm2d(192))

        if self.modification == 'activation' or self.modification == 'combined':
            layers.append(nn.LeakyReLU(inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        # 3rd Conv Layer
        layers.append(nn.Conv2d(192, 384, kernel_size=3, padding=1))
        if self.modification == 'batch_norm' or self.modification == 'combined':
            layers.append(nn.BatchNorm2d(384))

        if self.modification == 'activation' or self.modification == 'combined':
            layers.append(nn.LeakyReLU(inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))

        # 4th Conv Layer
        layers.append(nn.Conv2d(384, 256, kernel_size=3, padding=1))
        if self.modification == 'batch_norm' or self.modification == 'combined':
            layers.append(nn.BatchNorm2d(256))

        if self.modification == 'activation' or self.modification == 'combined':
            layers.append(nn.LeakyReLU(inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))

        # 5th Conv Layer
        layers.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        if self.modification == 'batch_norm' or self.modification == 'combined':
            layers.append(nn.BatchNorm2d(256))

        if self.modification == 'activation' or self.modification == 'combined':
            layers.append(nn.LeakyReLU(inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # Classifier
        classifier_layers = []
        classifier_layers.append(nn.Dropout())
        classifier_layers.append(nn.Linear(256 * 6 * 6, 4096))

        if self.modification == 'activation' or self.modification == 'combined':
            classifier_layers.append(nn.LeakyReLU(inplace=True))
        else:
            classifier_layers.append(nn.ReLU(inplace=True))

        classifier_layers.append(nn.Dropout())
        classifier_layers.append(nn.Linear(4096, 4096))

        if self.modification == 'activation' or self.modification == 'combined':
            classifier_layers.append(nn.LeakyReLU(inplace=True))
        else:
            classifier_layers.append(nn.ReLU(inplace=True))

        classifier_layers.append(nn.Linear(4096, num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_model(model_name, num_classes):
    if model_name == 'alexnet_baseline':
        return AlexNetCustom(num_classes=num_classes, modification=None)
    elif model_name == 'alexnet_mod1': # BatchNorm
        return AlexNetCustom(num_classes=num_classes, modification='batch_norm')
    elif model_name == 'alexnet_mod2': # LeakyReLU
        return AlexNetCustom(num_classes=num_classes, modification='activation')
    elif model_name == 'alexnet_combined': # Both
        return AlexNetCustom(num_classes=num_classes, modification='combined')
    else:
        raise ValueError(f"Unknown model name: {model_name}")
