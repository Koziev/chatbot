import re
import io


def is_empty_line(s):
    return re.match(r'([ ]*)\n', s) is not None


def calc_indent(s):
    mx = re.search(r'^([ ]*)', s)
    s = mx.group(1)
    return len(s)


class LineReader:
    def __init__(self, src_path):
        self.rdr = io.open(src_path, 'r', encoding='utf8')
        self.line = None
        self.last_line = None
        self.is_eof = False

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.rdr.close()

    def readline(self):
        if self.line:
            res = self.line
            self.line = None
            self.last_line = res
            return res
        else:
            res = self.rdr.readline()
            self.last_line = res
            if not res:
                self.is_eof = True

            return res

    def eof(self):
        return self.is_eof

    def back(self):
        assert(self.line is None)
        self.line = self.last_line
        self.last_line = None
