import logging
import os
from datetime import datetime


def configure_logging(file_timestamp: bool = True, level: str = 'INFO') -> None:
    logs_directory = 'logs/'
    os.makedirs(logs_directory, exist_ok=True)
    log_filename = f'{datetime.now().isoformat() if file_timestamp else "log"}.log'
    logfile_path = os.path.join(logs_directory, log_filename)
    print(f'Logging to {logfile_path} with level {logging.getLevelName(get_log_level(level))}')

    logging.root.handlers = []

    logging.basicConfig(
        level=get_log_level(level),
        format='[%(levelname)s] %(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(logfile_path),
            logging.StreamHandler()
        ]
    )


def get_log_level(level) -> int:
    """
    Defaults to logging.INFO
    :return:
    """

    value_map = {
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'WARNING': logging.WARNING,
        'WARN': logging.WARNING,
    }

    return value_map.get(level, logging.INFO)
