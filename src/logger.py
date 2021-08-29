import datetime as dt
import os
import sys

from src.utils_data import mkdir_if_missing


class Logger(object):
    def __init__(self, file_path=None):
        self.console = sys.stdout
        self.file = None
        if file_path is not None:
            mkdir_if_missing(file_path)
            self.file = open(file_path, 'a')
        self.nl = True

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def _write_with_ts(self, msg):
        ts = dt.datetime.now().replace(microsecond=0)
        msg = f'[{ts}] {msg}'
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def _write_without_ts(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def write(self, x):
        """Write function overloaded."""
        if x == '\n':
            self._write_without_ts(x)
            self.nl = True
        elif self.nl:
            self._write_with_ts(x)
            self.nl = False
        else:
            self._write_without_ts(x)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
