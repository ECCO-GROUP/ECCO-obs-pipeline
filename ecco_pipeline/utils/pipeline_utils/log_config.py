import logging
import os
from datetime import datetime

# Identifies this pipeline run. Fixed at import, in local time, and used both as the
# log directory name (logs/<timestamp>/) and as the run_id stamped on every Solr doc
# the run writes.
timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")


def run_id() -> str:
    """
    Identifier for the current pipeline run — the same string as the `logs/<run_id>/`
    directory name.

    Only meaningful in the parent process. Workers started with the "spawn" start
    method (the default on macOS) re-import this module and generate their own,
    *different* timestamp, so any code running in a worker must receive the parent's
    run id as an explicit argument rather than calling this.
    """
    return timestamp


def mp_logging(log_name: str, level: str = "INFO", log_dir="logs/") -> logging.Logger:
    logger = logging.getLogger(log_name)
    if logger.handlers:
        return logger

    if log_name == "pipeline":
        log_dir = f"logs/{timestamp}"

    os.makedirs(log_dir, exist_ok=True)
    logfile_path = os.path.join(log_dir, f"{log_name}.log")

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s (%(name)s) - %(message)s"
    )

    file_handler = logging.FileHandler(logfile_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
