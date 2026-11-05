from model import SpecCNN

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
