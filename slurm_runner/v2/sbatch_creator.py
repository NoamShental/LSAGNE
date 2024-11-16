import dataclasses
import os.path
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jinja2


@dataclass
class SbatchParameters:
    conda_env_path: Path | str
    code_path: Path | str
    sbatch_output_file: Path
    sbatch_error_file: Path
    cpu_cores: int
    ram_mem: int
    job_name: str
    command: str
    command_params: dict[str, Any] | None = None
    gpu_type: str | None = None


def create_sbatch_file(
        sbatch_parameters: SbatchParameters,
        output_path: Path
) -> None:
    template_loader = jinja2.FileSystemLoader(searchpath=os.path.dirname(__file__))
    template_env = jinja2.Environment(loader=template_loader)
    sbatch_template = template_env.get_template('sbatch_template.jinja2')
    sbatch = sbatch_template.render(dataclasses.asdict(sbatch_parameters))
    with open(output_path, 'w') as f:
        f.write(sbatch)
