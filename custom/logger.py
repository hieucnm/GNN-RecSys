import os
import sys
import os.path as osp


def mkdir_if_missing(path: str, _type: str = 'path'):
    assert _type in ['path', 'dir'], 'type must be `path` or `dir`'
    if _type == 'path':
        dir_path = osp.dirname(path)
    else:
        dir_path = path
    if not osp.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        return True
    return False


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
