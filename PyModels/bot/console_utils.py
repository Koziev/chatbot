# -*- coding: utf-8 -*-

import sys
import colorama  # https://pypi.python.org/pypi/colorama
import platform
import logging


def is_py2():
    return sys.version_info[0] < 3


def input_kbd(prompt):
    if is_py2():
        raw = raw_input(prompt.strip() + u' ')
        try:
            # Странные ошибки бывают при вводе в консоли с редактированием через backspace.
            # Из decode при этом вылетает UnicodeDecodeError.
            s8 = raw.decode(sys.stdout.encoding, errors='ignore').strip()
            return s8
        except Exception as ex:
            logging.exception('Error occured when decoding raw_input string')
            return u''
    else:
        return input(prompt.strip() + u' ').strip()


def print_error(error_msg):
    print(colorama.Fore.RED + u'{}'.format(error_msg) + colorama.Fore.RESET)


def print_answer(prompt, answer):
    print(prompt + u' ' + colorama.Fore.GREEN + u'{}'.format(answer) + colorama.Fore.RESET)


def print_tech_banner():
    print(colorama.Fore.LIGHTBLUE_EX +
          'Answering machine is running on ' +
          platform.platform() +
          colorama.Fore.RESET)
