import json
import os
import random
import numpy as np
import torch
from torch import nn

from constants import KAIMING_UNIFORM, KAIMING_NORMAL


def save_json(data, filename, save_pretty=False, sort_keys=False, cls=None):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys, cls=cls))
        else:
            json.dump(data, f, cls=cls)


def initialize_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


def initialization(module, initialize_type=KAIMING_UNIFORM):
    if initialize_type == KAIMING_NORMAL:
        nn.init.kaiming_normal_(module, mode='fan_out', nonlinearity='relu')
    elif initialize_type == KAIMING_UNIFORM:
        torch.nn.init.kaiming_uniform_(module, mode='fan_out', nonlinearity='relu')
    else:
        raise NotImplementedError


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        print(name, params)
        total_params += params

    print(f"Total Trainable Params: {total_params}")
    return total_params


def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)
