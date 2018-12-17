# -*- coding: utf-8 -*-

import codecs


class OrderComprehensionTable(object):
    def __init__(self):
        self.templates = []
        self.order2anchor = dict()

    def load_lists(self, lists):
        for group in lists:
            for sent in group:
                self.templates.append((sent, group[0]))
                self.order2anchor[sent] = group[0]

    def load_file(self, txt_path):
        """
        Из текстового файла загружаются группы опорных предложений
        для приказов. Каждая группа представляет из себя несколько строк,
        отделяемых от других групп пустой строкой. Первая строка в группе
        содержит опорный вариант приказа, к которому привязана обработка
        приказов. Остальные строки в группе содержат синонимичные варианты
        приказа.
        """
        with codecs.open(txt_path, 'r', 'utf-8') as rdr:
            group_lines = []
            for line in rdr:
                line = line.strip()
                if line:
                    group_lines.append(line)
                else:
                    for line in group_lines:
                        self.templates.append((group_lines[0], line))
                        self.order2anchor[line] = group_lines[0]
                    group_lines = []

    def get_templates(self):
        return list(self.templates)

    def get_order_anchor(self, order):
        return self.order2anchor.get(order, None)
