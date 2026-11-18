import copy
import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub, fuse_modules, get_default_qconfig, prepare, convert

torch.backends.quantized.engine = "qnnpack"


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
