import logging
import os
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    
def mp_logging(log_name: str, level: str = 'INFO', log_dir = 'logs/') -> logging.Logger:       
    logger = logging.getLogger(log_name)
    if logger.handlers:
        return logger
    
    if log_name == 'pipeline':
        log_dir = f'logs/{timestamp}'
    
    os.makedirs(log_dir, exist_ok=True)
    logfile_path = os.path.join(log_dir, f'{log_name}.log')
        
    logger.setLevel(level)
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s (%(name)s) - %(message)s')
    
    file_handler = logging.FileHandler(logfile_path)        
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger