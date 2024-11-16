import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np


def take_log_time(log_line: str) -> datetime | None:
    pattern = r"\[(.*?)\]"
    match = re.search(pattern, log_line)

    if match:
        return datetime.fromisoformat(match.group(1))
    else:
        return None


def humanize_time_delta(delta: timedelta) -> str:
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{days} days, {hours} Hours, {minutes} Minutes and {seconds} seconds"


def find_running_time(log_lines: list[str]):
    times = {}
    first_log = log_lines[0]
    if "Task 'InitializeLoggerFileHandlerTask': Finished task run for task with final state: 'Success'" in first_log:
        first_log_time = take_log_time(first_log)
        times['start'] = first_log_time
    else:
        raise AssertionError(f'First log is not recognized: "{first_log}"')
    print(f'First log time: {first_log_time}')

    last_log = log_lines[-1]
    if "Flow run SUCCESS: all reference tasks succeeded" in last_log:
        last_log_time = take_log_time(last_log)
        times['end'] = last_log_time
        print(f'Run took {humanize_time_delta(last_log_time - first_log_time)}')

    for log_entry in log_lines:
        match = re.match(r'.*Epoch (\d+)$', log_entry)
        if match:  # Match "Epoch" followed by one or more digits
            log_time = take_log_time(log_entry)
            times[int(match.group(1))] = log_time

    return times


if __name__ == '__main__':
    log_files_dir_path = Path(sys.argv[-1])
    aug_to_cpu_to_diff = defaultdict(lambda: defaultdict(list))
    for log_file in log_files_dir_path.glob("*.txt"):
        specs = log_file.name.split('__')
        tissue, perturbation, rep, total_reps = specs[0].split('_')
        cpu_count = int(specs[2].split('_')[5])
        augmentation_enabled = specs[8] == 'True'

        with open(log_file, 'r') as file:
            log_lines = file.readlines()
        times = find_running_time(log_lines)
        # aug_to_cpu_to_diff[augmentation_enabled][cpu_count].append(times[30] - times[20])
        # aug_to_cpu_to_diff[augmentation_enabled][cpu_count].append(times['end'] - times['start'])
        aug_to_cpu_to_diff[augmentation_enabled][cpu_count].append(times[60] - times[50])
        aug_to_cpu_to_diff[augmentation_enabled][cpu_count].append(times[50] - times[40])
        aug_to_cpu_to_diff[augmentation_enabled][cpu_count].append(times[40] - times[30])

    cpu_to_mean = {
        cpu: np.mean(diffs)
        for cpu, diffs in aug_to_cpu_to_diff[True].items()
    }
    for cpu, mean_time in cpu_to_mean.items():
        print(f'{cpu} ==> {humanize_time_delta(mean_time)}')
    pass

