from dataclasses import dataclass
from functools import cached_property
from typing import Dict

import torch


@dataclass(frozen=True)
class TrainingBatchLoss:
    loss_name_to_value_t: Dict[str, torch.Tensor]
    losses_coef: Dict[str, float]

    @cached_property
    def total_loss_t(self) -> torch.Tensor:
        return torch.stack(list(self.tuned_losses_t.values())).sum()

    @cached_property
    def total_loss(self) -> float:
        return self.total_loss_t.item()

    @cached_property
    def tuned_losses_t(self) -> Dict[str, torch.Tensor]:
        return {loss_name: loss_value_t * self.losses_coef[loss_name] for loss_name, loss_value_t in
                self.loss_name_to_value_t.items()}

    @cached_property
    def tuned_losses(self) -> Dict[str, float]:
        return {loss_name: loss_value_t.item() for loss_name, loss_value_t in self.tuned_losses_t.items()}
