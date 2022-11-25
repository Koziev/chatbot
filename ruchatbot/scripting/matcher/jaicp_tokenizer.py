import re
import io


class JAICP_Tokenizer:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.source = None
        self.src_len = 0
        self.cursor = 0
        self.delims1 = ',;[](){}-=+!?<>:"|&^%#@/⟦⟧'
        self.delims2 = ['->', ':=', '==', '+=', '||']
        self.whitespaces = ' \r\n\t\xa0'
        self.line_ranges = []
        self.rx_alnum1 = re.compile(r'\w{1}')

    def init_source(self, s):
        self.source = s
        self.src_len = len(s)
        pos1 = 0
        while True:
            pos2 = s.find('\n', pos1)
            if pos2 != -1:
                self.line_ranges.append((pos1, pos2))
                pos1 = pos2+1
            else:
                pos2 = self.src_len
                self.line_ranges.append((pos1, pos2))
                break

    @staticmethod
    def from_str(s):
        r = JAICP_Tokenizer()
        r.init_source(s)
        return r

    @staticmethod
    def from_file(filepath):
        with io.open(filepath, 'r', encoding='utf-8') as rdr:
            s = rdr.read()
            r = JAICP_Tokenizer(filepath)
            r.init_source(s)
            return r

    def is_alnum(self, c):
        return len(c) == 1 and self.rx_alnum1.search(c) is not None

    def eof(self):
        if self.cursor >= self.src_len:
            return True
        if any((c not in self.whitespaces) for c in self.source[self.cursor:]):
            p = self.tell()
            t = self.read()
            self.seek(p)
            return t == ''
        else:
            # под курсором и справа от него - только пробельные символы
            return True

    def tell(self):
        return self.cursor

    def seek(self, state):
        self.cursor = state

    def getc(self):
        if self.cursor < self.src_len:
            c = self.source[self.cursor]
            self.cursor += 1
            return c
        else:
            return ''

    def ungetc(self):
        assert(self.cursor > 0)
        self.cursor -= 1

    def skip_white(self):
        while self.cursor < self.src_len:
            c = self.getc()
            if c in self.whitespaces:
                continue
            elif c == '#':
                # пропускаем комментарий до конца строки
                while self.cursor < self.src_len:
                    c = self.getc()
                    if c == '\n':
                        break
            else:
                self.ungetc()
                break

    def get_cur_pos(self):
        for iline, (start, end) in enumerate(self.line_ranges):
            if start <= self.cursor < end:
                return iline+1, self.cursor-start+1
        return 0, 0

    def print_error(self, msg):
        line_no, col_no = self.get_cur_pos()
        if self.filepath:
            print('Error in file "{}", line {}, position {}: {}'.format(self.filepath, line_no, col_no, msg))
        else:
            print('Error in line {}, position {}: {}'.format(line_no, col_no, msg))
        exit(0)

    def read_str(self, escapes=True):
        self.skip_white()
        start_c = self.getc()
        if start_c not in '\'"':
            self.ungetc()
            self.print_error('Quoted string expected')

        buf = [start_c]
        while self.cursor < self.src_len:
            c = self.getc()
            if c == '\\' and escapes:
                c = self.getc()
            buf.append(c)
            if c == start_c:
                break

        return ''.join(buf)

    def read(self):
        buf = []
        self.skip_white()

        while self.cursor < self.src_len:
            c = self.getc()
            if c:
                if c in self.whitespaces:
                    self.ungetc()
                    break
                elif c == '*':
                    if buf:
                        return ''.join(buf) + c

                    # После звездочки могут быть буквы или {} с указанием кол-ва слов
                    pos1 = self.tell()
                    c2 = self.getc()
                    if c2 == ' ':
                        self.seek(pos1)
                        return c  # одиночная звездочка как токен
                    elif c2 == '{':
                        # в фигурных скобочках идет кол-во повторений.
                        # для простоты загружаем все символы до }
                        token = c + c2
                        while not self.eof():
                            c3 = self.getc()
                            token += c3
                            if c3 == '}':
                                break

                        return token
                    elif self.is_alnum(c2):
                        # Токен с форматом *стем*
                        # Для простоты грузим все символы до пробела или до разделителя
                        token = c + c2
                        while not self.eof():
                            pos2 = self.tell()
                            c3 = self.getc()
                            if c3 == ' ' or c3 in self.delims1:
                                self.seek(pos2)
                                break
                            else:
                                token += c3
                        return token
                    else:
                        self.seek(pos1)
                        return c

                elif c in self.delims1:
                    if buf:
                        self.ungetc()
                        break
                    else:
                        buf.append(c)
                        if any(s.startswith(c) for s in self.delims2):
                            pos1 = self.tell()
                            c2 = self.getc()

                            # многосимвольный разделитель
                            for delim in sorted(self.delims2, key=lambda z: -len(z)):
                                if delim.startswith(c) and delim[1] == c2:
                                    return c + c2

                            self.seek(pos1)

                        # многосимвольный не подобрался, возвращаем односимвольный разделитель.
                        return c
                else:
                    buf.append(c)
            else:
                break

        return ''.join(buf)

    def read_it(self, req_str):
        pos = self.tell()
        t = self.read()
        if t != req_str:
            self.seek(pos)
            self.print_error('Expected "{}", got "{}"'.format(req_str, t))

    def probe(self, req_str):
        pos = self.tell()
        t = self.read()
        self.seek(pos)
        return t == req_str

    def probe_read(self, req_str):
        pos = self.tell()
        t = self.read()
        if t == req_str:
            return True
        else:
            self.seek(pos)
            return False

    def here_comes_text(self, text):
        pos = self.tell()
        for req_c in text:
            c = self.getc()
            if c != req_c:
                self.seek(pos)
                return False

        return True

    def read_str_until(self, boundary):
        buf = []
        while not self.eof():
            if self.here_comes_text(boundary):
                break

            c = self.getc()
            buf.append(c)

        return ''.join(buf)

    def read_tokens_untill_cparen(self):
        tokens = []
        n_open_paren = 1
        while not self.eof():
            t = self.getc()
            if t == '\\':
                tokens.append(t)
                t = self.getc()
                tokens.append(t)
                continue
            elif t in '[({⟦':
                n_open_paren += 1
                tokens.append(t)
                # НАЧАЛО ОТЛАДКИ
                #print('DEBUG@256 t={} n_open_paren={} len={}'.format(t, n_open_paren, len(''.join(tokens))))
                # КОНЕЦ ОТЛАДКИ
            elif t in '])}⟧':
                n_open_paren -= 1
                # НАЧАЛО ОТЛАДКИ
                #print('DEBUG@261 t={} n_open_paren={} len={}'.format(t, n_open_paren, len(''.join(tokens))))
                # КОНЕЦ ОТЛАДКИ

                if n_open_paren == 0:
                    return ''.join(tokens)
                else:
                    tokens.append(t)
            else:
                tokens.append(t)

        raise RuntimeError()

    def read_rx_str(self):
        c = self.getc()
        if c == ' ':
            while c == ' ':
                c = self.getc()

        if c != '<':
            print('Expecting "<" in the beginning of regular expression, got "{}"'.format(c))
            raise RuntimeError()
        buf = []
        while not self.eof():
            c = self.getc()
            if c == '>':
                break
            else:
                buf.append(c)

        return ''.join(buf)


if __name__ == '__main__':
    #ps = '* *{1} *стем* *{1,3} (кошка/собака) {много/*мало/конец*}'
    #ps = 'о/конец*'
    ps = '* ($which/в чем) * {$your (фича/фичи/~фишка/~фишечка/~особенность/~изюминка)} *'

    src = JAICP_Tokenizer.from_str(ps)
    while not src.eof():
        t = src.read()
        print(t)

    # TODO ...
