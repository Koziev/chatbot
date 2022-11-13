import os
import sys
import re
import io
import glob
import logging

from .jaicp_pattern import JAICP_Pattern
from .global_rule import JAICP_GlobalRule
from .dsl_tools import is_empty_line, calc_indent, LineReader


def load_global_rules0(sc_path):
    pattern_lines = []

    with LineReader(sc_path) as rdr:
        while not rdr.eof():
            line = rdr.readline()
            if 'state:' in line:
                state_name = line.split(':')[1].strip()
                pattern_indent = calc_indent(line) + 4

                while not rdr.eof():
                    line = rdr.readline()
                    if is_empty_line(line) or line.strip().startswith('#'):
                        continue

                    i = calc_indent(line)
                    if i < pattern_indent:
                        rdr.back()
                        # Закончился блок паттернов в стейте "state:"
                        if pattern_lines:
                            pattern_str = ' '.join(pattern_lines)
                            yield state_name, pattern_str
                            pattern_lines.clear()
                        break
                    elif i == pattern_indent:
                        # Должно быть объявление нового паттерна

                        if pattern_lines:
                            # Обработаем предыдущий накопленный паттерн
                            pattern_str = ' '.join(pattern_lines)
                            yield state_name, pattern_str
                            pattern_lines.clear()

                        if 'q!:' in line:
                            s = line[line.index(':')+1:].strip()
                            pattern_lines.append(s)
                    elif i > pattern_indent:
                        # Продолжение тела паттерна
                        if pattern_lines:
                            pattern_lines.append(line.strip())
                        #else:
                        #    raise RuntimeError()
                    else:
                        raise RuntimeError()

    if pattern_lines:
        pattern_str = ' '.join(pattern_lines)
        yield state_name, pattern_str


def load_global_rules(patterns_dirs, named_patterns, entities):
    rules = []

    nb_errors = 0

    if True:
        for patterns_dir in patterns_dirs:
            for p in glob.glob(patterns_dir + '/**/*.sc', recursive=True):
                logging.debug('\nStart parsing file "{}"...'.format(p))
                for state, pattern_str in load_global_rules0(p):
                    try:
                        pattern = JAICP_Pattern.build(pattern_str, named_patterns=named_patterns, src_path=p)
                        pattern.bind_named_patterns(named_patterns)
                        pattern.bind_entities(entities)
                        pattern.optimize()

                        rule = JAICP_GlobalRule()
                        rule.src_path = p
                        rule.state = state
                        rule.pattern = pattern
                        rules.append(rule)

                    except (NotImplementedError, RuntimeError, re.error, ValueError, AssertionError, KeyError) as err:
                        logging.error('Global rule parsing error:')
                        logging.error('source file: %s', p)
                        logging.error('state name:  %s', state)
                        logging.error('pattern:     %s', pattern_str)
                        logging.error('error:       %s', err)
                        nb_errors += 1

    logging.debug('%d global rules loaded, %d skipped due to compilation errors', len(rules), nb_errors)

    return rules
