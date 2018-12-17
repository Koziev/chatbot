# -*- coding: utf-8 -*-

import sys
import colorama  # https://pypi.python.org/pypi/colorama
import platform


def is_py2():
    return sys.version_info[0] < 3


def input_kbd(prompt):
    if is_py2():
        return raw_input(prompt.strip() + u' ').decode(sys.stdout.encoding).strip().lower()
    else:
        return input(prompt.strip() + u' ').strip().lower()


def print_error(error_msg):
    print(colorama.Fore.RED + u'{}'.format(error_msg) + colorama.Fore.RESET)


def print_answer(prompt, answer):
    print(prompt + u' ' + colorama.Fore.GREEN + u'{}'.format(answer) + colorama.Fore.RESET)


def print_tech_banner():
    print(colorama.Fore.LIGHTBLUE_EX +
          'Answering machine is running on ' +
          platform.platform() +
          colorama.Fore.RESET)
