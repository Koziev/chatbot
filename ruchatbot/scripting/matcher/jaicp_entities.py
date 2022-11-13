""" Справочник именованной сущности - загрузка и преобразование во внутренее представление """

import os
import sys
import re
import io
import glob
import collections
import logging
import csv
import json

from .dsl_tools import is_empty_line, calc_indent


class JAICP_Entity:
    """ Справочник одной сущности """
    def __init__(self, csv_path, name, var, strict=False):
        self.name = name
        self.var = var
        self.strict = strict
        self.csv_path = csv_path
        self.items = []
        self.str2item = dict()
        self.min_len = 0
        self.max_len = 0

    def __repr__(self):
        return 'entities( "{}": {} items )'.format(self.name, len(self.items))

    def add_item(self, item_id, text, value):
        self.items.append(JAICP_EntityItem(item_id, text, value))

    def after_load(self):
        """ После загрузки из CSV вычислим некоторые параметры для ускорения сопоставления """
        self.min_len = 100
        self.max_len = 0
        for item in self.items:
            for v in item.variants:
                tokens = tuple(v.split(' '))
                nt = len(tokens)
                self.min_len = min(self.min_len, nt)
                self.max_len = max(self.max_len, nt)
                self.str2item[v.lower().replace('ё', 'е')] = item

    def find_item(self, probe_str):
        return self.str2item.get(probe_str.lower().replace('ё', 'е'))


class JAICP_EntityItem:
    """ Один элемент справочника """
    def __init__(self, item_id, src_text, value):
        self.item_id = item_id
        self.src_text = src_text
        self.variants = [z.strip() for z in src_text.split(',')]
        self.value = value

    def __repr__(self):
        return '[{}] {}'.format(self.item_id, self.src_text)


def load_ref(ref_path, ref_name, ref_var):
    ref = JAICP_Entity(ref_path, ref_name, ref_var)

    if os.path.exists(ref_path):
        with open(ref_path, newline='') as csvfile:
            rdr = csv.reader(csvfile, delimiter=';')
            for row in rdr:
                item_id = row[0]
                item_text = row[1]
                item_value = json.loads(row[2])
                ref.add_item(item_id, item_text, item_value)
    else:
        logging.error('File "%s" does not exist', ref_path)

    ref.after_load()
    return ref


def load_all_entities(dirs):
    """ Заходим в каждый из указанных каталогов, ищем в *.sc файлах
     упоминание csv-файлов справочников, загружаем. """
    refs = dict()

    for patterns_dir in dirs:
        for p in glob.glob(patterns_dir + '/**/*.sc', recursive=True):
            logging.debug('\nStart parsing file "{}"...'.format(p))
            with io.open(p, 'r', encoding='utf-8') as rdr:
                require_indent = 0
                ref_fname = None
                ref_name = None
                ref_var = None

                for line in rdr:
                    if 'require:' in line and '.csv' in line:
                        if ref_fname is not None:
                            ref = load_ref(ref_fname, ref_name, ref_var)
                            refs[ref_name] = ref
                            ref_name = None
                            ref_var = None

                        require_indent = calc_indent(line)
                        ref_fname = line[line.index(':')+1:].strip()
                        ref_fname = os.path.join(os.path.dirname(p), ref_fname)

                    elif is_empty_line(line):
                        if ref_fname is not None:
                            ref = load_ref(ref_fname, ref_name, ref_var)
                            refs[ref_name] = ref
                            ref_fname = None
                            ref_name = None
                            ref_var = None

                    elif ref_fname is not None and calc_indent(line) > require_indent:
                        if 'name' in line:
                            ref_name = line[line.index('=')+1:].strip()
                        elif 'var' in line:
                            ref_var = line[line.index('=')+1:].strip()
                        else:
                            raise NotImplementedError()

                if ref_fname is not None:
                    ref = load_ref(ref_fname, ref_name, ref_var)
                    refs[ref_name] = ref

    return refs


if __name__ == '__main__':
    dirs = ['/home/inkoziev/mts/chatbot/common/src',
                 '/home/inkoziev/mts/chatbot/skills/src',
                 '/home/inkoziev/mts/chatbot/chit-chat/src']

    refs = load_all_entities(dirs)

    print('{} referencies loaded'.format(len(refs)))
