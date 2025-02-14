### Modified https://qiita.com/bee2/items/4e462b545140a81abd44

import sys

class SeterrIO():
    """with構文でI/Oを切り替えるためのクラス"""
    def __init__(self, stdErrFilename: str):
        self.stderrfilename = stdErrFilename

    def __enter__(self):
        sys.stderr = _STDLogger(out_file=self.stderrfilename)

    def __exit__(self, *args):
        sys.stderr = sys.__stderr__

class _STDLogger():
    """カスタムI/O"""
    def __init__(self, out_file='out.log'):
        self.log = open(out_file, "a+")

    def write(self, message):
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        pass