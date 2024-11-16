from logging import Logger

import torch
from torch.types import Device


def choose_device(use_cuda: bool, logger: Logger = None) -> Device:
    logger = logger.warning if logger else print
    if not torch.cuda.is_available() and use_cuda:
        use_cuda = False
        logger('"use_cuda" is set, but no cuda device is available.')
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device
