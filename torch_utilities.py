import torch.nn as nn


def copy_state_dict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        state_dict_new = dict()
        for key, value in state_dict.items():
            state_dict_new[key.split(".", 1)[1]] = value
        return state_dict_new
    else:
        return state_dict


def initialize_weights(modules):
    for module in modules:
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(0, 0.01)
            module.bias.data.zero_()
