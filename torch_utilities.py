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
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
