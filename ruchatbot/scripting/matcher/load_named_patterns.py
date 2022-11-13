import os
import sys
import re
import io
import glob
import logging

from .jaicp_pattern import JAICP_Pattern
from .dsl_tools import is_empty_line, calc_indent



def load_named_patterns0(sc_path):
    pattern_name = None
    pattern_lines = []

    with io.open(sc_path, 'r', encoding='utf-8') as rdr:
        for line in rdr:
            if 'patterns:' in line:
                pattern_indent = calc_indent(line) + 4

                for line in rdr:
                    if is_empty_line(line) or line.strip().startswith('#'):
                        continue

                    i = calc_indent(line)
                    if i < pattern_indent:
                        # Закончился блок паттернов "patterns:"
                        if pattern_lines:
                            pattern_str = ' '.join(pattern_lines)
                            yield pattern_name, pattern_str
                            pattern_lines.clear()
                        break
                    elif i == pattern_indent:
                        # Должно быть объявление нового именованного паттерна

                        if pattern_lines:
                            # Обработаем предыдущий накопленный паттерн
                            pattern_str = ' '.join(pattern_lines)
                            yield pattern_name, pattern_str
                            pattern_lines.clear()

                        # строка с именем паттерна и, возможно, его телом
                        mx = re.match(r'([ ]+)(\$\w+)\s*\=', line)
                        if mx is None:
                            raise RuntimeError()
                        else:
                            pattern_name = mx.group(2)

                            s = line.strip()
                            pattern_tail = s[s.index('=')+1:].strip()
                            pattern_lines.append(pattern_tail)
                    elif i > pattern_indent:
                        # Продолжение тела паттерна
                        pattern_lines.append(line.strip())
                    else:
                        raise RuntimeError()

        if pattern_lines:
            pattern_str = ' '.join(pattern_lines)
            yield pattern_name, pattern_str


def load_named_patterns(patterns_dirs, entities):
    named_patterns = dict()

    nb_errors = 0

    pattern_names_with_errors = set()

    for patterns_dir in patterns_dirs:
        for p in glob.glob(patterns_dir + '/**/*.sc', recursive=True):
            logging.debug('\nStart parsing file "{}"...'.format(p))
            for pattern_name, pattern_str in load_named_patterns0(p):
                #logging.debug('Loading pattern %s', pattern_name)

                # НАЧАЛО ОТЛАДКИ
                if pattern_name != '$specGeoPlaces':
                    continue
                # КОНЕЦ ОТЛАДКИ

                if pattern_name in named_patterns:
                    if pattern_str != named_patterns[pattern_name].src_str:
                        logging.error('Named pattern %s redefinition. First definition is in "%s", second one is in "%s"', pattern_name, named_patterns[pattern_name].src_path, p)
                        #exit(0)
                        nb_errors += 1
                    else:
                        continue

                try:
                    pattern = JAICP_Pattern.build(pattern_str, src_path=p, named_patterns=named_patterns)
                    named_patterns[pattern_name] = pattern
                except (NotImplementedError, RuntimeError, re.error, ValueError, AssertionError) as err:
                    logging.error('Named pattern parsing error:')
                    logging.error('source file:  %s', p)
                    logging.error('pattern name: %s', pattern_name)
                    logging.error('pattern body: %s', pattern_str)
                    logging.error('error: %s', err)
                    nb_errors += 1
                    pattern_names_with_errors.add(pattern_name)

    logging.debug('%d patterns loaded, %d compilation errors', len(named_patterns), nb_errors)


    # НАЧАЛО ОТЛАДКИ
    del named_patterns['$skAppointParamComb4']
    # КОНЕЦ ОТЛАДКИ

    logging.error('Binding...')
    nb_errors = 0
    for i, (pattern_name, pattern) in enumerate(named_patterns.items(), start=1):
        try:
            pattern.bind_named_patterns(named_patterns)
            pattern.bind_entities(entities)
        except:
            nb_errors += 1
            logging.error('Named pattern binding error: pattern name=%s pattern body=%s error=%s', pattern_name, str(pattern), str(sys.exc_info()))

    logging.error('Binding completed, %d error(s) occured', nb_errors)

    return named_patterns
