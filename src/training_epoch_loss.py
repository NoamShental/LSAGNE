from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TrainingEpochLoss:
    loss_name_to_value: Dict[str, float]
    total_loss: float
    numeric_round: int = 3

    def __str__(self):
        loss_name_to_value = self.loss_name_to_value.copy()
        loss_name_to_value['TOTAL'] = round(self.total_loss, self.numeric_round)
        for loss_name, value in loss_name_to_value.items():
            loss_name_to_value[loss_name] = round(value, self.numeric_round)
        return f'{loss_name_to_value}'
