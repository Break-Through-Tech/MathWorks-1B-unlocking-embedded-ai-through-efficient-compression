import torch
import torch.nn as nn

CONV_POSITION = {"conv2": 4, "conv3": 8}


def project_conv(conv, rank):
    # kxk conv becomes a kxk conv down to `rank` channels, then a 1x1 conv back up
    out_c, in_c, k, _ = conv.weight.shape
    U, S, Vh = torch.linalg.svd(conv.weight.detach().reshape(out_c, -1), full_matrices=False)
    first = nn.Conv2d(in_c, rank, k)
    second = nn.Conv2d(rank, out_c, 1)
    first.weight.data = Vh[:rank].reshape(rank, in_c, k, k).clone()
    first.bias.data.zero_()
    second.weight.data = (U[:, :rank] * S[:rank]).reshape(out_c, rank, 1, 1).clone()
    second.bias.data = conv.bias.data.clone()
    return first, second


def project_model(model, ranks):
    layers = []
    for i, layer in enumerate(model.features):
        name = next((n for n, p in CONV_POSITION.items() if p == i), None)
        if name in ranks:
            layers.extend(project_conv(layer, ranks[name]))
        else:
            layers.append(layer)
    model.features = nn.Sequential(*layers)
    return model
