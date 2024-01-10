import logging
from logging import FileHandler
import os
from datetime import datetime

def root_logging(level: str = 'INFO'):
    log_directory = os.path.join('logs', datetime.now().strftime("%Y%m%dT%H%M%S"))
    os.makedirs(log_directory, exist_ok=True)
    
    logfile_path = os.path.join(log_directory, 'pipeline.log')
    
    logging.basicConfig(
        level=get_log_level(level),
        format='[%(levelname)s] %(asctime)s (%(process)d) - %(message)s',
        handlers=[
            FileHandler(logfile_path),
            logging.StreamHandler(),
        ]
    )
    return log_directory

def mp_logging(pid: str, log_subdir, level: str = 'INFO'):    
    os.makedirs(log_subdir, exist_ok=True)
    
    logfile_path = os.path.join(log_subdir, f'{pid}.log')
    
    logger = logging.getLogger(pid)
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(get_log_level(level))
    
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s (%(process)d) - %(message)s')
    
    file_handler = logging.FileHandler(logfile_path)        
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
    

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