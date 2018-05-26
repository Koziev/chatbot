# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import colorama  # https://pypi.python.org/pypi/colorama
import platform


colorama.init()


class EvaluationMarkup:
    ok_color = u'\033[92m'
    fail_color = u'\033[91m'
    close_color = u'\033[0m'

    ok_bullet = u'☑ '
    fail_bullet = u'☒ '

    @staticmethod
    def print_ok():
        if platform.system() == 'Linux':
            print(colorama.Fore.GREEN + EvaluationMarkup.ok_bullet+colorama.Fore.RESET, end='')
        else:
            print(colorama.Fore.GREEN + u'(+) '+colorama.Fore.RESET, end='')


    @staticmethod
    def print_fail():
        if platform.system() == 'Linux':
            print(colorama.Fore.RED+EvaluationMarkup.fail_bullet+colorama.Fore.RESET, end='')
        else:
            print(colorama.Fore.RED+'(-) '+colorama.Fore.RESET, end='')
