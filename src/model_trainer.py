import os
import time
from abc import ABC, abstractmethod
from datetime import timedelta
from logging import Logger
from typing import Generic, TypeVar

import torch
from torch.profiler import tensorboard_trace_handler

from src.configuration import config
from src.technical_learning_parameters import TechnicalLearningParameters
from src.os_utilities import create_dir_if_not_exists
from src.training_summary import TrainingSummary

Params = TypeVar('Params', bound=TechnicalLearningParameters)


class ModelTrainer(ABC, Generic[Params]):
    def __init__(self, logger: Logger, params: Params):
        self.logger = logger
        self.params = params
        self.logger.info("*" * 100 + "\n" + "Description:\n" + config.description + "*" * 100)

    # @abstractmethod
    # def on_batch_started(self, state, i_batch):
    #     pass
    #
    # @abstractmethod
    # def on_batch_finished(self, state):
    #     pass

    def _log_params(self):
        self.logger.info("=" * 100 + "params:")
        for key, val in self.params.__dict__.items():
            self.logger.info(f'param {key} --> {val}')
        self.logger.info("=" * 100 + "params END")

    @abstractmethod
    def on_epoch_started(self, i_epoch):
        pass

    @abstractmethod
    def on_epoch_finished(self, i_epoch):
        pass

    @abstractmethod
    def on_training_started(self):
        pass

    @abstractmethod
    def on_training_finished(self) -> TrainingSummary:
        pass

    @abstractmethod
    def get_data_loader(self):
        pass

    @abstractmethod
    def perform_batch(self, i_epoch, i_batch, batch):
        pass

    def train_model(self) -> TrainingSummary:
        self._log_params()
        start_time = time.time()

        self.on_training_started()
        #
        # profiler_path = os.path.join(self.params.working_directory, 'profiler')
        # create_dir_if_not_exists(profiler_path)
        # with torch.profiler.profile(
        #         # activities=[
        #         #     torch.profiler.ProfilerActivity.CPU,
        #         #     torch.profiler.ProfilerActivity.CUDA],
        #         schedule=torch.profiler.schedule(
        #             wait=2,
        #             warmup=2,
        #             active=5,
        #             repeat=1),
        #         record_shapes=True,
        #         # profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
        #         on_trace_ready=tensorboard_trace_handler(profiler_path),
        #         with_stack=True,
        #
        # ) as profiler:
        for i_epoch in range(1, self.params.n_epochs + 1):
            # self.logger.info(f"Epoch {i_epoch} out of {self.params.n_epochs}.")

            self.on_epoch_started(i_epoch)

            data_loader = self.get_data_loader()

            for i_batch, batch in enumerate(data_loader):
                self.logger.debug(f'batch: {i_batch + 1}')
                self.perform_batch(i_epoch, i_batch + 1, batch)
                    # if i_epoch > 30:
                    #     profiler.step()

            self.on_epoch_finished(i_epoch)

        res = self.on_training_finished()

        end_time = time.time()
        seconds_passed = end_time - start_time
        self.logger.info(f'Total training time => {seconds_passed} sec. Delta = {timedelta(seconds=seconds_passed)}.')

        return res
