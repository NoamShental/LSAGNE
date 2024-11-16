import os

from src.os_utilities import create_dir_if_not_exists


def write_str_to_file(folder_path: str, file_name: str, str_to_write: str):
    create_dir_if_not_exists(folder_path)
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w') as file:
        file.write(str_to_write)
