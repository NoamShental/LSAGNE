import re
import sys
from datetime import datetime, timedelta


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
    first_log = log_lines[0]
    if "Task 'InitializeLoggerFileHandlerTask': Finished task run for task with final state: 'Success'" in first_log:
        first_log_time = take_log_time(first_log)
    else:
        raise AssertionError(f'First log is not recognized: "{first_log}"')
    print(f'First log time: {first_log_time}')

    last_log = log_lines[-1]
    if "Flow run SUCCESS: all reference tasks succeeded" in last_log:
        last_log_time = take_log_time(last_log)
        print(f'Log diff = {humanize_time_delta(last_log_time - first_log_time)}')
        return
    else:
        print('Run did not finish! searching for closest epoch finished log...')

    for log_entry in reversed(log_lines):
        if re.match(r'.*Epoch \d+$', log_entry):  # Match "Epoch" followed by one or more digits
            print(f'Found epoch log: "{log_entry}"')
            last_log_time = take_log_time(log_entry)
            x = last_log_time - first_log_time
            print(f'Log diff = {humanize_time_delta(last_log_time - first_log_time)}')
            return

    print('Could not find an "Epoch" log...')


if __name__ == '__main__':
    log_file_path = sys.argv[-1]

    with open(log_file_path, 'r') as file:
        log_lines = file.readlines()

    find_running_time(log_lines)
