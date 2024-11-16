import torch
from torch import nn
from torch.nn import functional as F

from src.configuration import config


class MaxNormConstrainedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.max_val = 1.0
        self.eps = torch.finfo(config.torch_numeric_precision_type).eps

    # pass
    def forward(self, input):
        # return super(MaxNormConstrainedLinear, self).forward(input)
        # return F.linear(input, self.weight.clamp(min=-1.0, max=1.0), self.bias)
        # norm = torch.sqrt(torch.square(self.weight).sum(dim=0, keepdim=True))
        norm = self.weight.norm(2, dim=0, keepdim=True)
        desired = norm.clamp(0, self.max_val)
        return F.linear(input, self.weight * (desired / (self.eps + norm)), self.bias)
