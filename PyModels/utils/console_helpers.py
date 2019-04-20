# -*- coding: utf-8 -*-

import sys
import platform
import colorama  # https://pypi.python.org/pypi/colorama


def is_py2():
    return sys.version_info[0] < 3


def input_kbd(prompt):
    if is_py2():
        return raw_input(prompt.strip()+u' ').decode(sys.stdout.encoding).strip().lower()
    else:
        return input(prompt.strip()+u' ').strip().lower()


def print_red_line(msg):
    print(colorama.Fore.RED + msg + colorama.Fore.RESET)


def print_green_line(msg):
    print(colorama.Fore.GREEN + msg + colorama.Fore.RESET)


def get_ok_label():
    if platform.system() == 'Windows':
        return u'(+) '
    else:
        return u'☑ '


def get_fail_label():
    if platform.system() == 'Windows':
        return u'(-) '
    else:
        return u'☒ '
