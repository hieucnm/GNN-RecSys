import datetime as dt
import os
import sys

from custom.utils_data import mkdir_if_missing


class Logger(object):
    def __init__(self, file_path=None):
        self.console = sys.stdout
        self.file = None
        if file_path is not None:
            mkdir_if_missing(os.path.dirname(file_path))
            self.file = open(file_path, 'a')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        timestamp = dt.datetime.now().replace(microsecond=0)
        msg = f'[{timestamp}] {msg}'
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
