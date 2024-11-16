import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from torch import nn
from torch.optim import Optimizer


@dataclass
class BestEpochCheckpoint:
    total_loss: float = math.inf
    epoch: int = 0
    model_state_dict: Optional[Dict] = None
    optimizer_state_dict: Optional[Dict] = None
    cv_acc: float = 0
    cv_model_state_dict: Optional[Dict] = None
    cv_optimizer_state_dict: Optional[Dict] = None
    cv_epoch: int = 0

    def add_checkpoint(self, total_loss: float, i_epoch: int, model: nn.Module, optimizer: Optimizer) -> Tuple[bool, float]:
        old_total_loss = self.total_loss
        if self.total_loss > total_loss:
            self.total_loss = total_loss
            self.epoch = i_epoch
            self.model_state_dict = model.state_dict()
            self.optimizer_state_dict = optimizer.state_dict()
            return True, old_total_loss
        return False, old_total_loss

    def add_cv_checkpoint(self, cv_acc: float, i_epoch: int, model: nn.Module, optimizer: Optimizer) -> Tuple[bool, float]:
        old_cv_acc = self.cv_acc
        if self.cv_acc < cv_acc:
            self.cv_acc = cv_acc
            self.cv_epoch = i_epoch
            self.cv_model_state_dict = model.state_dict()
            self.cv_optimizer_state_dict = optimizer.state_dict()
            return True, old_cv_acc
        return False, old_cv_acc

    def reset_total_loss(self):
        self.total_loss = math.inf
        self.model_state_dict = None
        self.optimizer_state_dict = None
