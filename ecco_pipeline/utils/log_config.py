import logging
from logging import FileHandler
import os
from glob import glob
from datetime import datetime


def configure_logging(file_timestamp: bool = True, level: str = 'INFO', wipe_logs: bool = False) -> None:
    logs_directory = 'logs/'
    os.makedirs(logs_directory, exist_ok=True)
    
    if wipe_logs:
        for log_file in glob(f'{logs_directory}*.log'):
            os.remove(log_file)
    
    log_filename = f'{datetime.now().isoformat(timespec="seconds") if file_timestamp else "log"}.log'
    logfile_path = os.path.join(logs_directory, log_filename)
    print(f'Logging to {logfile_path} with level {logging.getLevelName(get_log_level(level))}')


    logging.basicConfig(
        level=get_log_level(level),
        format='[%(levelname)s] %(asctime)s - %(message)s',
        handlers=[
            MultiProcessingLog(logfile_path),
            logging.StreamHandler(),
        ]
    )
    # logging.root.handlers = []


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


import multiprocessing, threading, logging, sys, traceback

class MultiProcessingLog(logging.Handler):
    def __init__(self, name):
        logging.Handler.__init__(self)

        self._handler = FileHandler(name)
        self.queue = multiprocessing.Queue(-1)

        t = threading.Thread(target=self.receive)
        t.daemon = True
        t.start()

    def setFormatter(self, fmt):
        logging.Handler.setFormatter(self, fmt)
        self._handler.setFormatter(fmt)

    def receive(self):
        while True:
            try:
                record = self.queue.get()
                self._handler.emit(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except EOFError:
                break
            except:
                traceback.print_exc(file=sys.stderr)

    def send(self, s):
        self.queue.put_nowait(s)

    def _format_record(self, record):
        # ensure that exc_info and args
        # have been stringified.  Removes any chance of
        # unpickleable things inside and possibly reduces
        # message size sent over the pipe
        if record.args:
            record.msg = record.msg % record.args
            record.args = None
        if record.exc_info:
            dummy = self.format(record)
            record.exc_info = None

        return record

    def emit(self, record):
        try:
            s = self._format_record(record)
            self.send(s)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

    def close(self):
        self._handler.close()
        logging.Handler.close(self)