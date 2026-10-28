# Defines the 1D CNN and the spectrogram CNN
import torch.nn as nn

# Building the neural network
class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Look for patterns in the signal
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=64, stride=8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(8, 16, kernel_size=16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=8),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        # Turn those patterns into one score per class
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

class SpecCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

