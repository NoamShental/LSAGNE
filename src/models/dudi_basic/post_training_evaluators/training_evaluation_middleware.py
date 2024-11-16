# import logging
# from abc import ABC, abstractmethod
# from dataclasses import dataclass
#
# from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
# from src.drawer import Drawer
# from src.models.dudi_basic.evaluators.training_evaluator_params import TrainingEvaluatorParams
# from src.training_summary import TrainingSummary
#
#
# @dataclass
# class TrainingEvaluationMiddleware(ABC):
#     logger: logging.Logger
#
#     @abstractmethod
#     def run(self, training_evaluator_params: TrainingEvaluatorParams, **kwargs):
#         raise NotImplementedError()
