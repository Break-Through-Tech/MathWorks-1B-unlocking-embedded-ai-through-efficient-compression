from notebooks.Maayan.common import PrepData, BaseFaultModel, training_loop, test_model

CHECKPOINT_PATH = "notebooks/Maayan/largeCNN/large_model.pt"
CONFUSION_MATRIX_PATH = "notebooks/Maayan/largeCNN/confusion_matrix.png"


class FaultCNNLarge(BaseFaultModel):
    # TODO: override learning_rate / weight_decay here if the large model needs different values

    def __init__(self):
        # TODO: define conv blocks (Conv1d + BatchNorm1d + ReLU + MaxPool1d),
        # global pooling, and the classifier head
        raise NotImplementedError

    def forward(self, x):
        # TODO: x (batch, 1, 5000) -> logits (batch, num_classes)
        raise NotImplementedError


if __name__ == "__main__":
    # run from the repo root: python -m notebooks.Maayan.largeCNN.train_large_cnn
    # TODO: same flow as train_small_cnn.py, using FaultCNNLarge and the paths above
    raise NotImplementedError
