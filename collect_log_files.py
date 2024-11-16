import shutil
import sys
from pathlib import Path


if __name__ == '__main__':
    results_dir = Path('/RG/compbio/emil/experiments_root/experiments')
    # results_dir = Path('/Users/emil/MSc/lsagne-1/output/cloud_replica/experiments')
    experiment_prefix = sys.argv[-2]
    log_files_dest = Path(sys.argv[-1])
    print(f'Creating dir "{log_files_dest}"')
    log_files_dest.mkdir(exist_ok=True)
    for log_file_path in results_dir.glob(f"{experiment_prefix}*/results/*/log.txt"):
        experiment_name = log_file_path.parent.name
        print(f'Copying log of "{experiment_name}"')
        dest = log_files_dest / f'{experiment_name}.txt'
        try:
            shutil.copy2(log_file_path, dest)
            print(f"File copied")
        except FileNotFoundError:
            print(f"Source file not found: {log_file_path}")
        except PermissionError:
            print(f"Permission denied to write to: {dest}")
        except Exception as e:
            print(f"An error occurred: {e}")
