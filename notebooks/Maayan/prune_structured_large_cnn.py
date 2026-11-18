import torch

from notebooks.Maayan.common import PrepData, benchmark_model, save_confusion_matrix, training_loop
from notebooks.Maayan.largeCNN.train_large_cnn import FaultCNNLarge, CHECKPOINT_PATH

KEEP = 0.5  # fraction of channels kept in every conv layer
SWEEP = [0.75, 0.5, 0.35, 0.25]
FINETUNE_EPOCHS = 20
PRUNED_CHECKPOINT_PATH = "notebooks/Maayan/largeCNN/models/large_model_structured.pt"
PRUNED_CONFUSION_MATRIX_PATH = "notebooks/Maayan/largeCNN/results/confusion_matrix_structured.png"


def structured_prune(model, keep):
    """Build a narrower FaultCNNLarge keeping the highest-L1 channels of each conv.

    Removing output channel c of a conv also removes its BatchNorm channel and
    the matching input channel of the next conv (or the first classifier Linear).
    """
    convs = [block[0] for block in model.features]
    kept = []
    for conv in convs:
        n = max(1, round(conv.out_channels * keep))
        scores = conv.weight.detach().abs().sum(dim=(1, 2))
        kept.append(scores.topk(n).indices.sort().values)

    small = FaultCNNLarge(widths=tuple(len(k) for k in kept))
    prev = torch.arange(convs[0].in_channels)
    for i, (block, small_block) in enumerate(zip(model.features, small.features)):
        conv, bn = block[0], block[1]
        out = kept[i]
        small_block[0].weight.data = conv.weight.data[out][:, prev].clone()
        small_block[0].bias.data = conv.bias.data[out].clone()
        for name in ("weight", "bias", "running_mean", "running_var"):
            getattr(small_block[1], name).data = getattr(bn, name).data[out].clone()
        prev = out

    fc, small_fc = model.classifier[0], small.classifier[0]
    small_fc.weight.data = fc.weight.data[:, prev].clone()
    small_fc.bias.data = fc.bias.data.clone()
    small.classifier[3].load_state_dict(model.classifier[3].state_dict())
    return small


if __name__ == "__main__":
    # run from the repo root: python -m notebooks.Maayan.prune_structured_large_cnn
    train_loader, val_loader, test_loader = PrepData().create_loaders()

    for keep in SWEEP:
        path = PRUNED_CHECKPOINT_PATH.replace(".pt", f"_{int(keep * 100)}.pt")
        base = FaultCNNLarge()
        base.load_state_dict(torch.load(CHECKPOINT_PATH))
        model = structured_prune(base.eval(), keep)
        widths = [b[0].out_channels for b in model.features]

        criterion, optimizer = model.loss_optimizer()
        print(f"\n=== keep {keep} widths {widths} ===")
        training_loop(model, train_loader, val_loader, criterion, optimizer,
                      path, num_epochs=FINETUNE_EPOCHS)
        model.load_state_dict(torch.load(path))

        results = benchmark_model(model, test_loader, path)
        print(f"keep={keep} widths={widths} acc={results['accuracy']:.4f} f1={results['macro_f1']:.4f} "
              f"params={results['params']} size_kb={results['size_kb']:.1f} "
              f"latency_ms={results['latency_ms']:.2f}")
        if keep == KEEP:
            torch.save(model.state_dict(), PRUNED_CHECKPOINT_PATH)
            save_confusion_matrix(model, test_loader, PRUNED_CONFUSION_MATRIX_PATH)
