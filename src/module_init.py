from enum import Enum, auto
from logging import Logger
from typing import Optional

import torch
import torch.nn as nn


class ModuleGain(Enum):
    RELU = nn.init.calculate_gain('relu')
    PRELU = nn.init.calculate_gain('leaky_relu', 0.25)
    LEAKY_RELU = nn.init.calculate_gain('leaky_relu', 0.25)
    LINEAR_IDENTITY = nn.init.calculate_gain('linear')


@torch.no_grad()
def xavier_uniform_init(module: nn.Module, gain: ModuleGain = ModuleGain.LINEAR_IDENTITY, logger: Optional[Logger] = None):
    logger = logger.info if logger else print
    if isinstance(module, nn.Linear):
        logger(f'Xavier uniform initializing {module} with gain {gain}')
        torch.nn.init.xavier_uniform_(module.weight, gain.value)
        torch.nn.init.zeros_(module.bias)
    elif _skip_this_module(module):
        pass
    elif isinstance(module, nn.ModuleList) or isinstance(module, nn.Sequential):
        for child_module in module.children():
            xavier_uniform_init(child_module, gain)
    else:
        raise AssertionError(f'Unknown module {module}, cannot initialize')


def _skip_this_module(module: nn.Module):
    return isinstance(module, nn.PReLU) or \
           isinstance(module, nn.ReLU) or \
           isinstance(module, nn.LeakyReLU) or \
           isinstance(module, nn.Identity) or \
           isinstance(module, nn.Softmax)
