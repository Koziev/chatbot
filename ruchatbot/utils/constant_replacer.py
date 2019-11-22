# -*- coding: utf-8 -*-

import re


replacer_funcs = ['$chooseAdjByGender', '$chooseVByGender', '$chooseNByGender']


def replace_constant(string, constants, text_utils):
    if '$' in string:
        # Все макроподстановки начинаются с $
        for c_name, c_value in constants.items():
            c_name2 = '$'+c_name
            if c_name2 in string:
                string = string.replace(c_name2, c_value)

        # В строке могут еще быть функции замены типа @chooseAdjByGender.
        # Будем считать, что все они тоже начинаются с символа $
        for func in replacer_funcs:
            while func in string:
                pattern = func.replace('$', r'\$') + r'\((.+?)\)'
                mx = re.search(pattern, string)
                args_start = mx.start(1)
                args_end = mx.end(1)
                args_str = string[args_start : args_end]
                words = [w.strip() for w in args_str.split(',')]
                word = text_utils.apply_word_function(func, constants, words)
                string = string[:mx.start()] + word + string[mx.end():]

    return string