from collections import defaultdict
from typing import Dict, List

import numpy as np

from src.training_batch_loss import TrainingBatchLoss
from src.training_epoch_loss import TrainingEpochLoss


class LossesAggregator:
    @property
    def last_epoch_loss(self) -> TrainingEpochLoss:
        return TrainingEpochLoss(self.last_epoch_average_losses, self.epoch_total_loss_history[-1])

    @property
    def all_epochs_history(self):
        epoch_losses_history = self.epoch_losses_history.copy()
        epoch_losses_history['TOTAL'] = self.epoch_total_loss_history
        return epoch_losses_history

    @property
    def all_batches_history(self) -> Dict[str, List[float]]:
        batch_losses_history = self.batch_losses_history.copy()
        batch_losses_history['TOTAL'] = self.batch_total_loss_history
        for epoch_i, batch_length in enumerate(self.epoch_to_batch_length):
            batch_losses_history['epoch_i'].extend([epoch_i] * batch_length)
        return batch_losses_history

    def __init__(self):
        self.batch_losses_history = defaultdict(list)
        self.batch_total_loss_history = []
        self.epoch_losses_history = defaultdict(list)
        self.epoch_total_loss_history = []
        self._batch_length = 0
        self.epoch_to_batch_length = []
        self.last_epoch_average_losses = None

    def add_batch_loss(self, loss: TrainingBatchLoss) -> None:
        for loss_name, loss_value in loss.tuned_losses.items():
            self.batch_losses_history[loss_name].append(loss_value)
        self.batch_total_loss_history.append(loss.total_loss)
        self._batch_length += 1

    def end_epoch(self):
        self.epoch_to_batch_length.append(self._batch_length)
        self.last_epoch_average_losses = {loss_name: np.mean(history[-self._batch_length:]) for loss_name, history in
                                          self.batch_losses_history.items()}
        for loss_name, epoch_loss in self.last_epoch_average_losses.items():
            self.epoch_losses_history[loss_name].append(epoch_loss)
        self.epoch_total_loss_history.append(np.mean(self.batch_total_loss_history[-self._batch_length:]))
        self._batch_length = 0
