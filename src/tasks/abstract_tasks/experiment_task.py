from typing import List

from prefect import context, Task, Parameter
import os
from prefect.engine.results import LocalResult
from src.configuration import config


class ExperimentTask(Task):

    def __init__(self, cache_task=False, flow_parameters: List[Parameter] = None, *args, **kwargs):
        if cache_task and (flow_parameters is None or len(flow_parameters) == 0):
            raise Exception('cache_task is set, but no parameters where passed in order to create a target.')

        if flow_parameters is None:
            flow_parameters = []
        self.flow_parameters = flow_parameters
        target_folder = '-'.join(list(map(lambda parameter: f'{{parameters[{parameter.name}]}}', flow_parameters)))

        if cache_task:
            super().__init__(
                checkpoint=True,
                result=LocalResult(dir=config.output_folder_path),
                log_stdout=config.log_stdout,
                target=os.path.join(config.output_folder_path,
                                    # '{flow_name}/{parameters[tumor]}-{parameters[perturbation]}/_cache/{task_name}'),
                                    f'{target_folder}/_cache/{{task_name}}'),
                *args, **kwargs)
        else:
            super().__init__(
                log_stdout=config.log_stdout,
                *args, **kwargs)
        self.output_folder_path = config.output_folder_path

    @property
    def working_directory(self):
        if len(self.flow_parameters) == 0:
            raise Exception('No parameter were passed, so no working_directory could be properly defined...' +
                            ' Please pass a parameter such as run_name in order to avoid this error.')
        target_folder = '-'.join(
            list(map(lambda parameter:
                     f'{context.parameters[parameter.name] if parameter.name in context.parameters else parameter.default}',
                     self.flow_parameters)))
        return os.path.join(self.output_folder_path,
                            # f'{context.parameters["tumor"]}-{context.parameters["perturbation"]}')
                            target_folder)
