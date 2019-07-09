# -*- coding: utf-8 -*-

import logging
import logging.handlers


def init_trainer_logging(logfile_path, debugging=True):
    # настраиваем логирование в файл и эхо-печать в консоль
    log_level = logging.DEBUG if debugging else logging.ERROR
    logging.basicConfig(level=log_level, format='%(asctime)s %(message)s')

    lf = logging.FileHandler(logfile_path, mode='w')
    lf.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    lf.setFormatter(formatter)
    logging.getLogger('').addHandler(lf)
