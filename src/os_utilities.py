import os
import shutil
from os import PathLike


def create_dir_if_not_exists(folder_path: str | PathLike):
    os.makedirs(folder_path, exist_ok=True)


def delete_dir_if_exists(folder_path: str | PathLike):
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
