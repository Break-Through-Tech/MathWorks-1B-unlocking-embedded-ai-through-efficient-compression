import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from notebooks.Maayan.common import PrepData, benchmark_model, save_confusion_matrix, training_loop
from notebooks.Maayan.largeCNN.train_large_cnn import FaultCNNLarge, CHECKPOINT_PATH

SPARSITY = 0.8  # try 0.5, 0.7, 0.8, 0.9
PRUNED_CHECKPOINT_PATH = "notebooks/Maayan/largeCNN/models/large_model_pruned.pt"

def prunable(model):
    return [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]

def apply_pruning(model, sparsity):
    prune.global_unstructured(prunable(model), pruning_method=prune.L1Unstructured, amount=sparsity)

def make_permanent(model):
    for m, name in prunable(model):
        prune.remove(m, name)  # bakes the zeros into the weights, drops the mask


SWEEP = [0.5, 0.7, 0.8, 0.9]
FINETUNE_EPOCHS = 10
PRUNED_CONFUSION_MATRIX_PATH = "notebooks/Maayan/largeCNN/results/confusion_matrix_pruned.png"


def prune_and_finetune(sparsity, train_loader, val_loader, checkpoint_path):
    model = FaultCNNLarge()
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    apply_pruning(model, sparsity)

    criterion, optimizer = model.loss_optimizer()
    # mask stays active while fine-tuning, so pruned weights stay zero
    training_loop(model, train_loader, val_loader, criterion, optimizer,
                  checkpoint_path, num_epochs=FINETUNE_EPOCHS)

    # the best-val checkpoint holds weight_orig/weight_mask, so rebuild that
    # structure before loading it, then bake the zeros in
    best = FaultCNNLarge()
    for m, name in prunable(best):
        prune.identity(m, name)
    best.load_state_dict(torch.load(checkpoint_path))
    make_permanent(best)
    torch.save(best.state_dict(), checkpoint_path)
    return best


if __name__ == "__main__":
    # run from the repo root: python -m notebooks.Maayan.prune_large_cnn
    train_loader, val_loader, test_loader = PrepData().create_loaders()

    for sparsity in SWEEP:
        path = PRUNED_CHECKPOINT_PATH.replace(".pt", f"_{int(sparsity * 100)}.pt")
        print(f"\n=== sparsity {sparsity} ===")
        model = prune_and_finetune(sparsity, train_loader, val_loader, path)
        results = benchmark_model(model, test_loader, path)
        print(f"sparsity={sparsity} acc={results['accuracy']:.4f} f1={results['macro_f1']:.4f} "
              f"nonzero={results['nonzero_params']}/{results['params']} "
              f"size_kb={results['size_kb']:.1f} latency_ms={results['latency_ms']:.2f}")
        if sparsity == SPARSITY:
            torch.save(model.state_dict(), PRUNED_CHECKPOINT_PATH)
            save_confusion_matrix(model, test_loader, PRUNED_CONFUSION_MATRIX_PATH)
