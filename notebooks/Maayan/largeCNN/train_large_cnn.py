import torch.nn as nn
from notebooks.Maayan.common import PrepData, BaseFaultModel, training_loop, test_model

CHECKPOINT_PATH = "notebooks/Maayan/largeCNN/large_model.pt"
CONFUSION_MATRIX_PATH = "notebooks/Maayan/largeCNN/confusion_matrix.png"


class FaultCNNLarge(BaseFaultModel):
    def __init__(self):
        super().__init__()

        def block(cin, cout, k):
                return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.ReLU(), nn.MaxPool1d(4))

        self.features = nn.Sequential(
            block(1, 16, 7),
            block(16, 32, 7),
            block(32, 64, 5),
            block(64, 128, 3)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, self.num_classes)
        )
        

    
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool((x))
        x = x.squeeze(-1)
        x = self.classifier(x)

        return x





if __name__ == "__main__":
    # run from the repo root: python -m notebooks.Maayan.largeCNN.train_large_cnn
    train_loader, val_loader, test_loader = PrepData().create_loaders()

    model = FaultCNNLarge()
    print(f"params: {model.count_params()}")
    criterion, optimizer = model.loss_optimizer()

    training_loop(model, train_loader, val_loader, criterion, optimizer, CHECKPOINT_PATH)
    test_model(FaultCNNLarge(), test_loader, CHECKPOINT_PATH, CONFUSION_MATRIX_PATH)
