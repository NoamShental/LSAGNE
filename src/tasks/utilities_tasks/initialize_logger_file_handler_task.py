import logging
import os
from typing import List

import prefect
from prefect import Parameter

from src.configuration import config
from src.logger_annotations import LogAnnotations
from src.os_utilities import create_dir_if_not_exists
from src.tasks.abstract_tasks.experiment_task import ExperimentTask


class MichaelCustomFilter(logging.Filter):
    def __init__(self, michael_mode: bool):
        super(MichaelCustomFilter, self).__init__()
        self.michael_mode = michael_mode

    def filter(self, record):
        if not self.michael_mode and \
           LogAnnotations.MICHAEL_ONLY in record.__dict__:
            return False
        if self.michael_mode and \
           LogAnnotations.MICHAEL_NOT_INTERESTED in record.__dict__:
            return False
        return True


class InitializeLoggerFileHandlerTask(ExperimentTask):

    def __init__(self, flow_parameters: List[Parameter], michael_theme:bool):
        super(InitializeLoggerFileHandlerTask, self).__init__(flow_parameters=flow_parameters)
        self.file_handler = None
        self.michael_theme = michael_theme

    def run(self):
        create_dir_if_not_exists(self.working_directory)
        self.file_handler = logging.FileHandler(os.path.join(self.working_directory, 'log.txt'), 'a+')
        self.file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s | %(message)s"))
        self.logger.parent.setLevel(config.logger_level)
        self.logger.parent.addHandler(self.file_handler)
        self.logger.parent.addFilter(MichaelCustomFilter(self.michael_theme))

    def close_logger(self):
        self.logger.parent.removeHandler(self.file_handler)
        self.file_handler = None

    @staticmethod
    def connect_to_flow(flow: prefect.Flow, root_tasks: List[prefect.Task], parameters: List[Parameter], michael_theme: bool):
        logger_task = InitializeLoggerFileHandlerTask(flow_parameters=parameters, michael_theme=michael_theme)
        logger_task.tags.add('logger')
        flow.set_dependencies(
            task=logger_task,
            downstream_tasks=root_tasks
        )
