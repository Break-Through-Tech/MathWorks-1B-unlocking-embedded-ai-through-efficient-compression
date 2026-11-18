import copy
import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub, fuse_modules, get_default_qconfig, prepare, convert
from model import SpecCNN

torch.backends.quantized.engine = "qnnpack"


CONV_POSITION = {"conv1": 0, "conv2": 4}


def filter_scores(model, layer):
    pos = CONV_POSITION[layer]
    return model.features[pos].weight.detach().abs().sum(dim=(1, 2, 3))


def keep_largest(model, layer, count):
    scores = filter_scores(model, layer)
    return scores.argsort(descending=True)[:count].tolist()


def shrink(model, layer, keep):
    pos = CONV_POSITION[layer]
    keep = sorted(keep)
    filters1 = model.features[0].out_channels
    filters2 = model.features[4].out_channels
    if layer == "conv1":
        filters1 = len(keep)
    else:
        filters2 = len(keep)
    new = SpecCNN(model.classifier.out_features, filters1, filters2)

    changed = [pos, pos + 1, pos + 4]
    for i in range(len(model.features)):
        if i not in changed:
            new.features[i].load_state_dict(model.features[i].state_dict())
    new.classifier.load_state_dict(model.classifier.state_dict())

    old_conv, new_conv = model.features[pos], new.features[pos]
    new_conv.weight.data = old_conv.weight.data[keep].clone()
    new_conv.bias.data = old_conv.bias.data[keep].clone()

    old_bn, new_bn = model.features[pos + 1], new.features[pos + 1]
    new_bn.weight.data = old_bn.weight.data[keep].clone()
    new_bn.bias.data = old_bn.bias.data[keep].clone()
    new_bn.running_mean = old_bn.running_mean[keep].clone()
    new_bn.running_var = old_bn.running_var[keep].clone()

    old_next, new_next = model.features[pos + 4], new.features[pos + 4]
    new_next.weight.data = old_next.weight.data[:, keep].clone()
    new_next.bias.data = old_next.bias.data.clone()
    return new


PROJECT_POSITION = {"conv2": 4, "conv3": 8}


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
        name = next((n for n, p in PROJECT_POSITION.items() if p == i), None)
        if name in ranks:
            layers.extend(project_conv(layer, ranks[name]))
        else:
            layers.append(layer)
    model.features = nn.Sequential(*layers)
    return model


def fuse_groups(features):
    return [[str(i - 1), str(i), str(i + 1)] for i, m in enumerate(features) if isinstance(m, nn.BatchNorm2d)]


class QuantModel(nn.Module):
    def __init__(self, features, classifier):
        super().__init__()
        self.quant = QuantStub()
        self.features = features
        self.classifier = classifier
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.features(self.quant(x))
        x = self.classifier(x.flatten(1))
        return self.dequant(x)


def quantize(model, calib_loader):
    model = copy.deepcopy(model).eval()
    features = fuse_modules(model.features, fuse_groups(model.features))
    quant_model = QuantModel(features, model.classifier).eval()
    quant_model.qconfig = get_default_qconfig("qnnpack")
    prepare(quant_model, inplace=True)
    with torch.no_grad():
        for X, y in calib_loader:
            quant_model(X)
    convert(quant_model, inplace=True)
    return quant_model
