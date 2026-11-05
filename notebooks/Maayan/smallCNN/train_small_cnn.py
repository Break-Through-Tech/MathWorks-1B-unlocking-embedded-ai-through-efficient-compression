import torch.nn as nn

from notebooks.Maayan.common import PrepData, BaseFaultModel, training_loop, test_model

CHECKPOINT_PATH = "notebooks/Maayan/smallCNN/small_model.pt"
CONFUSION_MATRIX_PATH = "notebooks/Maayan/smallCNN/confusion_matrix.png"


class FaultCNN(BaseFaultModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=7)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=7)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, self.num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    # run from the repo root: python -m notebooks.Maayan.smallCNN.train_small_cnn
    train_loader, val_loader, test_loader = PrepData().create_loaders()

    model = FaultCNN()
    criterion, optimizer = model.loss_optimizer()

    training_loop(model, train_loader, val_loader, criterion, optimizer, CHECKPOINT_PATH)
    test_model(FaultCNN(), test_loader, CHECKPOINT_PATH, CONFUSION_MATRIX_PATH)
